#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include <stack>
#include <unordered_map>
#include <cmath>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <ginac/ginac.h>
#include <iomanip>
using namespace GiNaC;
// #include <math_functions.h>
#ifdef USE_GAOL
#include <gaol/gaol.h>
#endif

// ---------------- CUDA CHECK ----------------
#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " " << file << " " << line << std::endl;
        exit(code);
    }
}

// ---------------- DATA STRUCTURES ----------------
struct Interval
{
    double lo, hi;
};

enum OpType
{
    OP_VAR,
    OP_CONST,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_POW,
    OP_EXP,
    OP_SIN,
    OP_COS,
    OP_TAN,
    OP_LOG
};

struct Node
{
    int op;
    int left;
    int right;
    double value;
    int var_idx;
};

// ---------------- DEVICE OPS ----------------

__device__ __forceinline__ double next_up(double x)
{
    if (isnan(x) || isinf(x)) return x;
    if (x == 0.0) return __longlong_as_double(1ULL);
    unsigned long long u = __double_as_longlong(x);
    if (x > 0) u++;
    else u--;
    return __longlong_as_double(u);
}

__device__ __forceinline__ double next_down(double x)
{
    if (isnan(x) || isinf(x)) return x;
    if (x == 0.0) return -__longlong_as_double(1ULL);
    unsigned long long u = __double_as_longlong(x);
    if (x > 0) u--;
    else u++;
    return __longlong_as_double(u);
}
__device__ Interval d_add(Interval a, Interval b)
{
    return {a.lo + b.lo, a.hi + b.hi};
}
__device__ Interval d_sub(Interval a, Interval b)
{
    return {a.lo - b.hi, a.hi - b.lo};
}
__device__ Interval d_mul(Interval a, Interval b)
{
    double v1 = a.lo * b.lo;
    double v2 = a.lo * b.hi;
    double v3 = a.hi * b.lo;
    double v4 = a.hi * b.hi;
    return {fmin(fmin(v1, v2), fmin(v3, v4)), fmax(fmax(v1, v2), fmax(v3, v4))};
}
__device__ Interval d_div(Interval a, Interval b)
{
    if (b.lo < 0 && b.hi > 0)
        return {-INFINITY, INFINITY};
    else if(b.lo == 0)
    {
        Interval inv {1/b.hi, INFINITY};
        return d_mul(a,inv);
    }
    else if(b.hi == 0)
    {
        Interval inv {-INFINITY, 1/b.lo};
        return d_mul(a,inv);
    }
    Interval inv = {1.0 / b.hi, 1.0 / b.lo};
    return d_mul(a, inv);
}
__device__ __forceinline__ Interval d_exp(Interval a)
{
    double lo = exp(a.lo);
    double hi = exp(a.hi);

    lo = __longlong_as_double(__double_as_longlong(lo) - 1);
    hi = __longlong_as_double(__double_as_longlong(hi) + 1);

    return {lo, hi};
}
__device__ __forceinline__ Interval d_pow_int(Interval a, int n)
{
    if (n == 0) return {1.0, 1.0};

    bool is_negative_exp = false;
    if (n < 0)
    {
        if (a.lo <= 0.0 && a.hi >= 0.0)
            return {-INFINITY, INFINITY};
            
        is_negative_exp = true;
        n = -n; // Make exponent positive for the calculation
    }

    double l = pow(a.lo, (double)n);
    double r = pow(a.hi, (double)n);

    Interval p;
    if (n % 2 == 0)
    {
        double lo = (a.lo <= 0.0 && a.hi >= 0.0) ? 0.0 : ((l < r) ? l : r);
        double hi = (l > r) ? l : r;
        p = {lo, hi};
    }
    else
    {
        double lo = (l < r) ? l : r;
        double hi = (l > r) ? l : r;
        p = {lo, hi};
    }

    if (is_negative_exp) {
        return d_div({1.0, 1.0}, p);
    }
    
    return p;
}
__device__ __forceinline__ Interval d_sin(Interval a)
{
    const double PI = 3.141592653589793238462643383279502884;
    const double TWO_PI = 2.0 * PI;

    if (a.hi - a.lo >= TWO_PI)
        return {-1.0, 1.0};

    double lo = fmin(sin(a.lo), sin(a.hi));
    double hi = fmax(sin(a.lo), sin(a.hi));

    double kmax = ceil((a.lo - PI / 2.0) / TWO_PI);
    double tmax = PI / 2.0 + kmax * TWO_PI;
    if (tmax >= a.lo && tmax <= a.hi)
        hi = 1.0;

    double kmin = ceil((a.lo - 3.0 * PI / 2.0) / TWO_PI);
    double tmin = 3.0 * PI / 2.0 + kmin * TWO_PI;
    if (tmin >= a.lo && tmin <= a.hi)
        lo = -1.0;

    return {next_down(lo), next_up(hi)};
}

__device__ __forceinline__ Interval d_cos(Interval a)
{
    const double PI = 3.141592653589793238462643383279502884;
    const double TWO_PI = 2.0 * PI;

    if (a.hi - a.lo >= TWO_PI)
        return {-1.0, 1.0};

    double lo = fmin(cos(a.lo), cos(a.hi));
    double hi = fmax(cos(a.lo), cos(a.hi));

    double kmax = ceil(a.lo / TWO_PI);
    double tmax = kmax * TWO_PI;
    if (tmax >= a.lo && tmax <= a.hi)
        hi = 1.0;

    double kmin = ceil((a.lo - PI) / TWO_PI);
    double tmin = PI + kmin * TWO_PI;
    if (tmin >= a.lo && tmin <= a.hi)
        lo = -1.0;

    return {next_down(lo), next_up(hi)};
}

__device__ __forceinline__ Interval d_tan(Interval a)
{
    const double PI = 3.141592653589793238462643383279502884;

    // If the interval is wider than pi, it definitely hits an asymptote.
    if (a.hi - a.lo >= PI)
        return {-INFINITY, INFINITY};

    double lo = tan(a.lo);
    double hi = tan(a.hi);

    // Safety check for NaN
    if (isnan(lo) || isnan(hi))
        return {-INFINITY, INFINITY};

    // Because tan(x) is strictly monotonically increasing between asymptotes,
    // if the left bound's tangent is >= the right bound's tangent,
    // the interval MUST have crossed a vertical asymptote.
    if (lo >= hi && a.lo != a.hi)
        return {-INFINITY, INFINITY};

    return {next_down(lo), next_up(hi)};
}

__device__ __forceinline__ Interval d_log(Interval a)
{
    if (a.hi <= 0.0)
        return {-INFINITY, INFINITY};

    double lo;
    if (a.lo <= 0.0) {
        lo = -INFINITY;
    } else {
        lo = next_down(log(a.lo));
    }

    double hi = next_up(log(a.hi));

    return {lo, hi};
}
// ---------------- DEVICE EVAL ----------------

__device__ Interval eval_function_device(const Node* nodes, int n, int offset, const Interval* vars)
{
    if (n <= 0)
        return {-INFINITY, INFINITY};

    Interval stack[256];

    for (int i = 0; i < n; i++)
    {
        Node nd = nodes[i];

        // ✅ convert GLOBAL → LOCAL indices
        int l = (nd.left  == -1) ? -1 : nd.left  - offset;
        int r = (nd.right == -1) ? -1 : nd.right - offset;

        if (nd.op == OP_VAR)
        {
            stack[i] = vars[nd.var_idx];
        }
        else if (nd.op == OP_CONST)
        {
            stack[i] = {nd.value, nd.value};
        }
        else if (nd.op == OP_ADD)
        {
            stack[i] = d_add(stack[l], stack[r]);
        }
        else if (nd.op == OP_SUB)
        {
            stack[i] = d_sub(stack[l], stack[r]);
        }
        else if (nd.op == OP_MUL)
        {
            if (l >= i || r >= i)
            {
                stack[i] = {-INFINITY, INFINITY};
                continue;
            }

            stack[i] = d_mul(stack[l], stack[r]);
        }
        else if (nd.op == OP_DIV)
        {
            if (l >= i || r >= i)
            {
                stack[i] = {-INFINITY, INFINITY};
                continue;
            }

            stack[i] = d_div(stack[l], stack[r]);
        }
        else if (nd.op == OP_POW)
        {
            if (l >= i || r >= i)
            {
                stack[i] = {-INFINITY, INFINITY};
                continue;
            }

            Interval base_val = stack[l];
            Interval expo = stack[r];

            if (expo.lo != expo.hi)
            {
                stack[i] = {-INFINITY, INFINITY};
            }
            else
            {
                int p = (int)llround(expo.lo);
                stack[i] = d_pow_int(base_val, p);
            }
        }
        else if (nd.op == OP_EXP)
        {
            if (l >= i)
            {
                stack[i] = {-INFINITY, INFINITY};
                continue;
            }

            stack[i] = d_exp(stack[l]);
        }
        else if (nd.op == OP_SIN)
        {
            if (l >= i)
            {
                stack[i] = {-INFINITY, INFINITY};
                continue;
            }

            stack[i] = d_sin(stack[l]);
        }
        else if (nd.op == OP_COS)
        {
            if (l >= i)
            {
                stack[i] = {-INFINITY, INFINITY};
                continue;
            }

            stack[i] = d_cos(stack[l]);
        }
        else if (nd.op == OP_TAN)
        {
            if (l >= i)
            {
                stack[i] = {-INFINITY, INFINITY};
                continue;
            }
            stack[i] = d_tan(stack[l]);
        }
        else if (nd.op == OP_LOG)
        {
            if (l >= i)
            {
                stack[i] = {-INFINITY, INFINITY};
                continue;
            }
            stack[i] = d_log(stack[l]);
        }
    }

    return stack[n - 1];
}
// ---------------- KERNEL ----------------
__global__ void evaluate_matrix_kernel(const Node *nodes, const int *offsets,
                                       const int *sizes, const Interval *vars,
                                       Interval *results, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (sizes[i] <= 0)
    {
        results[i] = {-INFINITY, INFINITY};
        return;
    }

    const Node *fn = &nodes[offsets[i]];
    results[i] = eval_function_device(fn, sizes[i], offsets[i], vars);
}

// ---------------- PARSER ----------------
struct Token
{
    int type;
    std::string val;
};

bool is_function_token(const std::string& op)
{
    return op == "exp" || op == "sin" || op == "cos";
}

int precedence(const std::string& op)
{
    if (op == "^") return 3;
    if (op == "*" || op == "/") return 2;
    if (op == "+" || op == "-") return 1;
    return 0;
}

bool is_binary_op(const std::string& op)
{
    return op == "+" || op == "-" || op == "*" || op == "/" || op == "^";
}

bool is_right_associative(const std::string& op)
{
    return op == "^";
}

// std::vector<Token> tokenize(const std::string &s)
// {
//     std::vector<Token> t;
//     for (int i = 0; i < s.size();)
//     {
//         if (isspace(s[i]))
//         {
//             i++;
//             continue;
//         }
//         if (isdigit(s[i]))
//         {
//             std::string num;
//             while (i < s.size() && (isdigit(s[i]) || s[i] == '.'))
//                 num += s[i++];
//             t.push_back({1, num});
//         }
//         else if (isalpha(s[i]))
//         {
//             std::string v;
//             while (i < s.size() && isalpha(s[i]))
//                 v += s[i++];
//             t.push_back({2, v});
//         }
//         else
//         {
//             t.push_back({3, std::string(1, s[i++])});
//         }
//     }
//     return t;
// }

// std::vector<Token> infix_to_postfix(const std::vector<Token> &t)
// {
//     std::vector<Token> out;
//     std::stack<Token> st;

//     for (const auto &x : t)
//     {
//         if (x.type == 1)
//         {
//             out.push_back(x);
//         }
//         else if (x.type == 2 && x.val != "exp" && x.val != "sin" && x.val != "cos")
//         {
//             out.push_back(x);
//         }
//         else if (x.val == "exp" || x.val == "sin" || x.val == "cos")
//         {
//             st.push(x);
//         }
//         else if (is_function_token(x.val))
//         {
//             st.push(x);
//         }
//         else if (x.val == "(")
//         {
//             st.push(x);
//         }
//         else if (x.val == ")")
//         {
//             while (!st.empty() && st.top().val != "(")
//             {
//                 out.push_back(st.top());
//                 st.pop();
//             }
//             if (!st.empty() && st.top().val == "(")
//                 st.pop();

//             if (!st.empty() && is_function_token(st.top().val))
//             {
//                 out.push_back(st.top());
//                 st.pop();
//             }
//         }
//         else if (is_binary_op(x.val))
//         {
//             while (!st.empty() && is_binary_op(st.top().val))
//             {
//                 int ptop = precedence(st.top().val);
//                 int pcur = precedence(x.val);

//                 if (ptop > pcur || (ptop == pcur && !is_right_associative(x.val)))
//                 {
//                     out.push_back(st.top());
//                     st.pop();
//                 }
//                 else
//                 {
//                     break;
//                 }
//             }
//             st.push(x);
//         }
//     }

//     while (!st.empty())
//     {
//         out.push_back(st.top());
//         st.pop();
//     }

//     return out;
// }

// std::vector<Node> build_nodes_from_postfix(
//     const std::vector<Token> &pf,
//     const std::unordered_map<std::string, int> &mp)
// {
//     std::vector<Node> nodes;
//     std::stack<int> st;

//     for (const auto &t : pf)
//     {
//         if (t.type == 1)
//         {
//             nodes.push_back({OP_CONST, -1, -1, std::stod(t.val), 0});
//             st.push((int)nodes.size() - 1);
//         }
//         else if (t.type == 2 && t.val != "exp" && t.val != "sin" && t.val != "cos")
//         {
//             nodes.push_back({OP_VAR, -1, -1, 0, mp.at(t.val)});
//             st.push((int)nodes.size() - 1);
//         }
//         else if (t.val == "exp" || t.val == "sin" || t.val == "cos")
//         {
//             if (st.empty())
//             {
//                 printf("Error: unary op stack empty\n");
//                 continue;
//             }

//             int a = st.top(); st.pop();

//             int op = (t.val == "exp") ? OP_EXP :
//                     (t.val == "sin") ? OP_SIN :
//                                         OP_COS;

//             nodes.push_back({op, a, -1, 0, 0});
//             st.push((int)nodes.size() - 1);
//         }
//         else
//         {
//             if (st.size() < 2)
//             {
//                 std::cerr << "Bad postfix for operator " << t.val << std::endl;
//                 nodes.clear();
//                 nodes.push_back({OP_CONST, -1, -1, 0.0, 0});
//                 return nodes;
//             }

//             int b = st.top();
//             st.pop();
//             int a = st.top();
//             st.pop();

//             int op = (t.val == "+") ? OP_ADD :
//                     (t.val == "-") ? OP_SUB :
//                     (t.val == "*") ? OP_MUL :
//                     (t.val == "/") ? OP_DIV :
//                                     OP_POW;

//             nodes.push_back({op, a, b, 0, 0});
//             st.push((int)nodes.size() - 1);
//         }
//     }

//     return nodes;
// }

#ifdef USE_GAOL
gaol::interval eval_gaol(const std::vector<Token> &pf,
                         const std::unordered_map<std::string, int> &mp,
                         const std::vector<Interval> &vars)
{
    std::stack<gaol::interval> st;

    for (auto &t : pf)
    {
        if (t.type == 1)
        {
            st.push(gaol::interval(std::stod(t.val)));
        }
        else if (t.type == 2 && t.val != "exp" && t.val != "sin" && t.val != "cos")
        {
            int i = mp.at(t.val);
            st.push(gaol::interval(vars[i].lo, vars[i].hi));
        }
        else if (t.val == "exp")
        {
            auto a = st.top(); st.pop();
            st.push(gaol::exp(a));
        }
        else if (t.val == "sin")
        {
            auto a = st.top(); st.pop();
            st.push(gaol::sin(a));
        }
        else if (t.val == "cos")
        {
            auto a = st.top(); st.pop();
            st.push(gaol::cos(a));
        }
        else
        {
            auto b = st.top();
            st.pop();
            auto a = st.top();
            st.pop();

            if (t.val == "+")
                st.push(a + b);
            else if (t.val == "-")
                st.push(a - b); 
            else if (t.val == "*")
                st.push(a * b);
            else if (t.val == "/")
                st.push(a / b);
            // else if (t.val == "exp")
            // {
            //     auto a = st.top(); st.pop();
            //     st.push(gaol::exp(a));
            // }
            // else if (t.val == "sin")
            // {
            //     auto a = st.top(); st.pop();
            //     st.push(gaol::sin(a));
            // }
            // else if (t.val == "cos")
            // {
            //     auto a = st.top(); st.pop();
            //     st.push(gaol::cos(a));
            // }
            else if (t.val == "^")
            {
                auto expv = st.top(); st.pop();
                auto base = st.top(); st.pop();

                if (expv.left() == expv.right())
                {
                    int p = (int)llround(expv.left());
                    st.push(gaol::pow(base, p));
                }
                else
                {
                    double inf = std::numeric_limits<double>::infinity();
                    st.push(gaol::interval(-inf, inf));
                }
            }
        }
    }

    return st.top();
}
#endif

int compile_ginac(const ex &e,
                  std::vector<Node> &nodes,
                  std::unordered_map<std::string,int> &mp)
{
    if (is_a<numeric>(e))
    {
        nodes.push_back({OP_CONST, -1, -1, ex_to<numeric>(e).to_double(), 0});
        return (int)nodes.size() - 1;
    }

    if (is_a<symbol>(e))
    {
        std::string name = ex_to<symbol>(e).get_name();
        nodes.push_back({OP_VAR, -1, -1, 0.0, mp[name]});
        return (int)nodes.size() - 1;
    }

    if (is_a<power>(e))
    {
        int l = compile_ginac(e.op(0), nodes, mp);
        int r = compile_ginac(e.op(1), nodes, mp);

        nodes.push_back({OP_POW, l, r, 0, 0});
        return nodes.size() - 1;
    }

    if (is_a<function>(e))
    {
        std::string name = ex_to<function>(e).get_name();
        int a = compile_ginac(e.op(0), nodes, mp);

        int op = OP_EXP;
        if (name == "sin") op = OP_SIN;
        else if (name == "cos") op = OP_COS;
        else if (name == "tan") op = OP_TAN;
        else if (name == "log") op = OP_LOG;

        nodes.push_back({op, a, -1, 0.0, 0});
        return (int)nodes.size() - 1;
    }

    if (is_a<add>(e))
    {
        int cur = compile_ginac(e.op(0), nodes, mp);

        for (int i = 1; i < e.nops(); i++)
        {
            int rhs = compile_ginac(e.op(i), nodes, mp);

            nodes.push_back({OP_ADD, cur, rhs, 0, 0});
            cur = nodes.size() - 1;
        }

        return cur;
    }

    if (is_a<mul>(e))
    {
        int cur = compile_ginac(e.op(0), nodes, mp);

        for (int i = 1; i < e.nops(); i++)
        {
            int rhs = compile_ginac(e.op(i), nodes, mp);

            nodes.push_back({OP_MUL, cur, rhs, 0, 0});
            cur = nodes.size() - 1;
        }

        return cur;
    }
    throw std::runtime_error("Unsupported GiNaC node");
}

gaol::interval evaluate_expression(const GiNaC::ex &expr,
                                   const std::map<std::string, gaol::interval> &variable_intervals)
{
    if (GiNaC::is_a<GiNaC::numeric>(expr))
    {
        double value = GiNaC::ex_to<GiNaC::numeric>(expr).to_double();
        return gaol::interval(value, value);
    }
    else if (GiNaC::is_a<GiNaC::symbol>(expr))
    {
        const GiNaC::symbol &sym = GiNaC::ex_to<GiNaC::symbol>(expr);
        auto it = variable_intervals.find(sym.get_name());
        if (it != variable_intervals.end())
            return it->second;
        throw std::runtime_error("Variable not found in interval map: " + sym.get_name());
    }
    else if (GiNaC::is_a<GiNaC::add>(expr))
    {
        gaol::interval result(0.0, 0.0);
        for (size_t i = 0; i < expr.nops(); ++i)
            result = result + evaluate_expression(expr.op(i), variable_intervals);
        return result;
    }
    else if (GiNaC::is_a<GiNaC::mul>(expr))
    {
        gaol::interval result = evaluate_expression(expr.op(0), variable_intervals);
        for (size_t i = 1; i < expr.nops(); ++i)
            result = result * evaluate_expression(expr.op(i), variable_intervals);
        return result;
    }
    else if (GiNaC::is_a<GiNaC::power>(expr))
    {
        gaol::interval base = evaluate_expression(expr.op(0), variable_intervals);
        gaol::interval exponent = evaluate_expression(expr.op(1), variable_intervals);

        if (exponent.left() == exponent.right())
        {
            int p = (int)llround(exponent.left());
            return gaol::pow(base, p);
        }

        double inf = std::numeric_limits<double>::infinity();
        return gaol::interval(-inf, inf);
    }
    else if (GiNaC::is_a<GiNaC::function>(expr))
    {
        const GiNaC::function &f = GiNaC::ex_to<GiNaC::function>(expr);
        gaol::interval arg = evaluate_expression(expr.op(0), variable_intervals);

        if (f.get_name() == "exp") return gaol::exp(arg);
        if (f.get_name() == "sin") return gaol::sin(arg);
        if (f.get_name() == "cos") return gaol::cos(arg);
        if (f.get_name() == "tan") return gaol::tan(arg);
        if (f.get_name() == "log") return gaol::log(arg);

        throw std::runtime_error("Unsupported function: " + f.get_name());
    }

    throw std::runtime_error("Unsupported GiNaC expression type.");
}

// ---------------- MAIN ----------------
int main()
{
    std::cout << std::fixed << std::setprecision(6);
    // std::vector<std::string> vars_name = {"x", "y"};

    // std::unordered_map<std::string, int> mp;
    // for (int i = 0; i < vars_name.size(); i++)
    //     mp[vars_name[i]] = i;

    // std::vector<Interval> vars = {{1, 3}, {0, 4}};
    // ---------------- FUNCTION INPUT ----------------
    // Write custom functions here. Keep the syntax compatible with tokenize().
    // Example functions:
    //   "x+y"
    //   "x/exp(y)+x"
    //   "x*x-y/exp(x)"
    // std::vector<std::string> funcs = {
    //     // "x+exp(y)/x*x*x/x/y-exp(x)/y-y/exp(y)",
    //     // "x/exp(x)+y+x*exp(x)/y*y*exp(y)/exp(x)/y+x",
    //     // "x+x-x/x-y+y/y*y/x-x",
    //     // "x-y-y-y-x-y*y/x-y*y+y",
    //     // "x+x-y+x",
    //     "x^3"
    //     // "sin(x)",
    //     // "cos(y)",
    //     // "x^3+sin(y)-cos(x)"
    // };


    int m = 100; // Number of random functions to generate
    int n = 100;     // Number of variables (x1, x2, x3)

    std::vector<std::string> funcs;
    std::mt19937 rng(42); // Fixed seed so you compare the exact same workload every time
    
    // We omit '/' here because dividing by zero intervals frequently yields [-inf, inf],
    // which short-circuits math operations and makes the benchmark artificially fast.
    std::vector<std::string> bin_ops = {"+", "-", "*", "/"}; 
    std::vector<std::string> un_funcs = {"sin", "cos", "tan", "exp"};

    for (int i = 0; i < m; i++)
    {
        // Start with a random base variable
        std::string f = "x" + std::to_string((rng() % n) + 1);
        
        // Append between 2 to 6 random terms
        int num_terms = (rng() % 5) + 2; 
        for (int j = 0; j < num_terms; j++)
        {
            std::string op = bin_ops[rng() % bin_ops.size()];
            std::string var = "x" + std::to_string((rng() % n) + 1);
            
            int term_type = rng() % 3;
            if (term_type == 0) {
                // Example: + sin(x2)
                std::string func_name = un_funcs[rng() % un_funcs.size()];
                f += op + func_name + "(" + var + ")";
            } else if (term_type == 1) {
                // Example: * x1^3
                int power = (rng() % 3) + 2; // Powers 2, 3, or 4
                f += op + var + "^" + std::to_string(power);
            } else {
                // Example: - x3
                f += op + var;
            }
        }
        funcs.push_back(f);
    }

    // int F = (int)funcs.size();
    // int n = 3;

    // std::vector<std::string> funcs = {
    //     "sin(x2)",
    //     "x1^4 + exp(x3) + x2^2 +cos(x1^2)",
    //     "x1+x2^-3",
    //     "x2 + log(x1)"
    // };

    int F = (int)funcs.size();

    GiNaC::parser reader;
    GiNaC::symtab table;

    std::unordered_map<std::string, int> mp;
    std::vector<std::string> var_names;

    for (int i = 1; i <= n; i++)
    {
        std::string name = "x" + std::to_string(i);
        GiNaC::symbol s(name);
        table[name] = s;
        mp[name] = i - 1;
        var_names.push_back(name);
    }

    reader.get_syms() = table;
    double L = 1.0;
    double H = 2.0;

    std::vector<Interval> vars_h(n);
    std::map<std::string, gaol::interval> gaol_env;

    for (int i = 0; i < n; i++)
    {
        vars_h[i] = {L, H};
        gaol_env[var_names[i]] = gaol::interval(L, H);
    }

    std::vector<std::vector<Token>> pf;
    std::vector<std::vector<Node>> np;

    // for (auto &f : funcs)
    // {
    //     auto t = tokenize(f);
    //     auto p = infix_to_postfix(t);
    //     for (auto &tok : p) std::cout << tok.val << " ";
    //     std::cout << "\n";
    //     pf.push_back(p);
    //     np.push_back(build_nodes_from_postfix(p, mp));
    // }

    // std::vector<Node> all;
    // std::vector<int> off, sz;

    // for (auto &v : np)
    // {
    //     off.push_back(all.size());
    //     sz.push_back(v.size());
    //     for (auto &n : v)
    //         all.push_back(n);
    // }

    std::vector<Node> all_nodes;
    std::vector<int> offsets, sizes;
    std::vector<GiNaC::ex> exprs;

    for (int i = 0; i < F; i++)
    {
        GiNaC::ex expr = reader(funcs[i]);
        exprs.push_back(expr);

        std::vector<Node> nodes;
        compile_ginac(expr, nodes, mp);

        int base = all_nodes.size();

        for (auto &nd : nodes)
        {
            if (nd.left != -1)  nd.left  += base;
            if (nd.right != -1) nd.right += base;
        }

        offsets.push_back(base);
        sizes.push_back(nodes.size());

        all_nodes.insert(all_nodes.end(), nodes.begin(), nodes.end());
    }

    Node *d_nodes;
    int *d_off, *d_sz;
    Interval *d_vars, *d_out;

    CUDA_CHECK(cudaMalloc(&d_nodes, all_nodes.size() * sizeof(Node)));
    CUDA_CHECK(cudaMalloc(&d_off, offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sz, sizes.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vars, vars_h.size() * sizeof(Interval)));
    CUDA_CHECK(cudaMalloc(&d_out, F * sizeof(Interval)));

    CUDA_CHECK(cudaMemcpy(d_nodes, all_nodes.data(),
                        all_nodes.size() * sizeof(Node), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_off, offsets.data(),
                        offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_sz, sizes.data(),
                        sizes.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_vars, vars_h.data(),
                        vars_h.size() * sizeof(Interval), cudaMemcpyHostToDevice));
    auto t1 = std::chrono::high_resolution_clock::now();

    int threads = 256;
    int blocks = (F + threads - 1) / threads;

    evaluate_matrix_kernel<<<blocks, threads>>>(d_nodes, d_off, d_sz, d_vars, d_out, F);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    std::vector<Interval> gpu(F);
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_out, F * sizeof(Interval), cudaMemcpyDeviceToHost));
    
    double gpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "GPU Time (ms): " << gpu_time << "\n";

    std::cout << "\nGPU Results:\n";
    for (int i = 0; i < F; i++)
    {
        std::cout << "f" << i << " = " << funcs[i] << "  ->  [" << gpu[i].lo << ", " << gpu[i].hi << "]\n";
    }

#ifdef USE_GAOL
    auto c1 = std::chrono::high_resolution_clock::now();

    std::cout << "\nGAOL Results:\n";
    for (int i = 0; i < F; i++)
    {
        auto r = evaluate_expression(exprs[i], gaol_env);
        std::cout << funcs[i] << " -> [" << r.left() << ", " << r.right() << "]\n";
    }
    auto c2 = std::chrono::high_resolution_clock::now();

    double cpu_time = std::chrono::duration<double, std::milli>(c2 - c1).count();

    std::cout << "\nGAOL CPU Time (ms): " << cpu_time << "\n";
#endif
    std::cout << "GPU Time (ms): " << gpu_time << "\n";
    return 0;
}
