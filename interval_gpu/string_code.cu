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
    OP_COS
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
    if (n == 0)
        return {1.0, 1.0};

    if (n < 0)
    {
        if (a.lo <= 0.0 && a.hi >= 0.0)
            return {-INFINITY, INFINITY};
        Interval p = d_pow_int(a, -n);
        return d_div({1.0, 1.0}, p);
    }

    if (n % 2 == 0)
    {
        double l = pow(a.lo, n);
        double r = pow(a.hi, n);
        double hi = fmax(l, r);
        double lo = (a.lo <= 0.0 && a.hi >= 0.0) ? 0.0 : fmin(l, r);
        return {next_down(lo), next_up(hi)};
    }
    else
    {
        double l = pow(a.lo, n);
        double r = pow(a.hi, n);
        return {next_down(fmin(l, r)), next_up(fmax(l, r))};
    }
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

// ---------------- DEVICE EVAL ----------------
__device__ Interval eval_function_device(const Node *nodes, int n, const Interval *vars)
{
    if (n <= 0)
        return {-INFINITY, INFINITY};

    Interval stack[256];
    for (int i = 0; i < n; i++)
    {
        Node nd = nodes[i];
        if (nd.op == OP_VAR)
            stack[i] = vars[nd.var_idx];
        else if (nd.op == OP_CONST)
            stack[i] = {nd.value, nd.value};
        else if (nd.op == OP_ADD)
            stack[i] = d_add(stack[nd.left], stack[nd.right]);
        else if (nd.op == OP_SUB)
            stack[i] = d_sub(stack[nd.left], stack[nd.right]);
        else if (nd.op == OP_MUL)
            stack[i] = d_mul(stack[nd.left], stack[nd.right]);
        else if (nd.op == OP_DIV)
            stack[i] = d_div(stack[nd.left], stack[nd.right]);
        else if (nd.op == OP_POW)
        {
            if (nd.left < 0 || nd.right < 0 || nd.left >= i || nd.right >= i)
            {
                stack[i] = {-INFINITY, INFINITY};
                continue;
            }

            Interval base = stack[nd.left];
            Interval expo = stack[nd.right];

            if (expo.lo != expo.hi)
            {
                stack[i] = {-INFINITY, INFINITY};
            }
            else
            {
                int p = (int)llround(expo.lo);
                stack[i] = d_pow_int(base, p);
            }
        }
        else if (nd.op == OP_EXP)
            stack[i] = d_exp(stack[nd.left]);
        else if (nd.op == OP_SIN)
            stack[i] = d_sin(stack[nd.left]);
        else if (nd.op == OP_COS)
            stack[i] = d_cos(stack[nd.left]);
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
    results[i] = eval_function_device(fn, sizes[i], vars);
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

std::vector<Token> tokenize(const std::string &s)
{
    std::vector<Token> t;
    for (int i = 0; i < s.size();)
    {
        if (isspace(s[i]))
        {
            i++;
            continue;
        }
        if (isdigit(s[i]))
        {
            std::string num;
            while (i < s.size() && (isdigit(s[i]) || s[i] == '.'))
                num += s[i++];
            t.push_back({1, num});
        }
        else if (isalpha(s[i]))
        {
            std::string v;
            while (i < s.size() && isalpha(s[i]))
                v += s[i++];
            t.push_back({2, v});
        }
        else
        {
            t.push_back({3, std::string(1, s[i++])});
        }
    }
    return t;
}

std::vector<Token> infix_to_postfix(const std::vector<Token> &t)
{
    std::vector<Token> out;
    std::stack<Token> st;

    for (const auto &x : t)
    {
        if (x.type == 1)
        {
            out.push_back(x);
        }
        else if (x.type == 2 && x.val != "exp" && x.val != "sin" && x.val != "cos")
        {
            out.push_back(x);
        }
        else if (x.val == "exp" || x.val == "sin" || x.val == "cos")
        {
            st.push(x);
        }
        else if (is_function_token(x.val))
        {
            st.push(x);
        }
        else if (x.val == "(")
        {
            st.push(x);
        }
        else if (x.val == ")")
        {
            while (!st.empty() && st.top().val != "(")
            {
                out.push_back(st.top());
                st.pop();
            }
            if (!st.empty() && st.top().val == "(")
                st.pop();

            if (!st.empty() && is_function_token(st.top().val))
            {
                out.push_back(st.top());
                st.pop();
            }
        }
        else if (is_binary_op(x.val))
        {
            while (!st.empty() && is_binary_op(st.top().val))
            {
                int ptop = precedence(st.top().val);
                int pcur = precedence(x.val);

                if (ptop > pcur || (ptop == pcur && !is_right_associative(x.val)))
                {
                    out.push_back(st.top());
                    st.pop();
                }
                else
                {
                    break;
                }
            }
            st.push(x);
        }
    }

    while (!st.empty())
    {
        out.push_back(st.top());
        st.pop();
    }

    return out;
}

std::vector<Node> build_nodes_from_postfix(
    const std::vector<Token> &pf,
    const std::unordered_map<std::string, int> &mp)
{
    std::vector<Node> nodes;
    std::stack<int> st;

    for (const auto &t : pf)
    {
        if (t.type == 1)
        {
            nodes.push_back({OP_CONST, -1, -1, std::stod(t.val), 0});
            st.push((int)nodes.size() - 1);
        }
        else if (t.type == 2 && t.val != "exp" && t.val != "sin" && t.val != "cos")
        {
            nodes.push_back({OP_VAR, -1, -1, 0, mp.at(t.val)});
            st.push((int)nodes.size() - 1);
        }
        else if (t.val == "exp" || t.val == "sin" || t.val == "cos")
        {
            if (st.empty())
            {
                printf("Error: unary op stack empty\n");
                continue;
            }

            int a = st.top(); st.pop();

            int op = (t.val == "exp") ? OP_EXP :
                    (t.val == "sin") ? OP_SIN :
                                        OP_COS;

            nodes.push_back({op, a, -1, 0, 0});
            st.push((int)nodes.size() - 1);
        }
        else
        {
            if (st.size() < 2)
            {
                std::cerr << "Bad postfix for operator " << t.val << std::endl;
                nodes.clear();
                nodes.push_back({OP_CONST, -1, -1, 0.0, 0});
                return nodes;
            }

            int b = st.top();
            st.pop();
            int a = st.top();
            st.pop();

            int op = (t.val == "+") ? OP_ADD :
                    (t.val == "-") ? OP_SUB :
                    (t.val == "*") ? OP_MUL :
                    (t.val == "/") ? OP_DIV :
                                    OP_POW;

            nodes.push_back({op, a, b, 0, 0});
            st.push((int)nodes.size() - 1);
        }
    }

    return nodes;
}

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

// ---------------- MAIN ----------------
int main()
{

    std::vector<std::string> vars_name = {"x", "y"};
    std::unordered_map<std::string, int> mp;
    for (int i = 0; i < vars_name.size(); i++)
        mp[vars_name[i]] = i;

    std::vector<Interval> vars = {{1, 3}, {0, 4}};
    // ---------------- FUNCTION INPUT ----------------
    // Write custom functions here. Keep the syntax compatible with tokenize().
    // Example functions:
    //   "x+y"
    //   "x/exp(y)+x"
    //   "x*x-y/exp(x)"
    std::vector<std::string> funcs = {
        // "x+exp(y)/x*x*x/x/y-exp(x)/y-y/exp(y)",
        // "x/exp(x)+y+x*exp(x)/y*y*exp(y)/exp(x)/y+x",
        // "x+x-x/x-y+y/y*y/x-x",
        // "x-y-y-y-x-y*y/x-y*y+y",
        // "x+x-y+x",
        "x^3"
        // "sin(x)",
        // "cos(y)",
        // "x^3+sin(y)-cos(x)"
    };

    // If you want random functions instead, comment out the manual list above
    // and uncomment this block.
    
    // std::vector<std::string> funcs;
    // std::mt19937 rng(42);
    // std::vector<std::string> ops = {"+", "-", "*", "/"};
    // int n = 100;
    // for (int i = 0; i < n; i++)
    // {
    //     std::string f = "x";
    //     for (int j = 0; j < 10; j++)
    //     {
    //         if (rng() % 3 == 0)
    //             f += ops[rng() % 4] + "exp(" + vars_name[rng() % 2] + ")";
    //         else
    //             f += ops[rng() % 4] + vars_name[rng() % 2];
    //     }
    //     funcs.push_back(f);
    // }
    

    int N = (int)funcs.size();

    std::vector<std::vector<Token>> pf;
    std::vector<std::vector<Node>> np;

    for (auto &f : funcs)
    {
        auto t = tokenize(f);
        auto p = infix_to_postfix(t);
        for (auto &tok : p) std::cout << tok.val << " ";
        std::cout << "\n";
        pf.push_back(p);
        np.push_back(build_nodes_from_postfix(p, mp));
    }

    std::vector<Node> all;
    std::vector<int> off, sz;

    for (auto &v : np)
    {
        off.push_back(all.size());
        sz.push_back(v.size());
        for (auto &n : v)
            all.push_back(n);
    }

    Node *d_nodes;
    int *d_off, *d_sz;
    Interval *d_vars, *d_res;
    
    CUDA_CHECK(cudaMalloc(&d_nodes, all.size() * sizeof(Node)));
    CUDA_CHECK(cudaMalloc(&d_off, off.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sz, sz.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vars, vars.size() * sizeof(Interval)));
    CUDA_CHECK(cudaMalloc(&d_res, funcs.size() * sizeof(Interval)));

    CUDA_CHECK(cudaMemcpy(d_nodes, all.data(), all.size() * sizeof(Node), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_off, off.data(), off.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sz, sz.data(), sz.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vars, vars.data(), vars.size() * sizeof(Interval), cudaMemcpyHostToDevice));

    auto t1 = std::chrono::high_resolution_clock::now();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    evaluate_matrix_kernel<<<blocks, threads>>>(d_nodes, d_off, d_sz, d_vars, d_res, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t2 = std::chrono::high_resolution_clock::now();

    std::vector<Interval> gpu(funcs.size());
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_res, funcs.size() * sizeof(Interval), cudaMemcpyDeviceToHost));

    double gpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "GPU Time (ms): " << gpu_time << "\n";

    std::cout << "\nGPU Results:\n";
    for (int i = 0; i < N; i++)
    {
        std::cout << "f" << i << " = " << funcs[i] << "  ->  [" << gpu[i].lo << ", " << gpu[i].hi << "]\n";
    }

#ifdef USE_GAOL
    auto c1 = std::chrono::high_resolution_clock::now();

    std::cout << "\nGAOL Results:\n";
    for (int i = 0; i < N; i++)
    {
        auto r = eval_gaol(pf[i], mp, vars);
        std::cout << "f" << i << " = " << funcs[i] << "  ->  " << r << "\n";
    }

    auto c2 = std::chrono::high_resolution_clock::now();

    double cpu_time = std::chrono::duration<double, std::milli>(c2 - c1).count();

    std::cout << "\nGAOL CPU Time (ms): " << cpu_time << "\n";
#endif
    std::cout << "GPU Time (ms): " << gpu_time << "\n";
    return 0;
}
