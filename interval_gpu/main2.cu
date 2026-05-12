// File: main.cu
// GPU + auto-parser + optional GAOL hybrid example
// Build with CMakeLists.txt provided in the canvas.

#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <cctype>
#include <cmath>
#include <unordered_map>
#include <cuda_runtime.h>

#ifdef USE_GAOL
#include <gaol/gaol.h>
#endif

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " " << file << " " << line << std::endl;
        exit(code);
    }
}

// -----------------------------
// Device/Host shared types
// -----------------------------
struct Interval {
    double lo, hi;
};

// Use a small enum for op types; keep layout simple
enum OpType { OP_VAR = 0, OP_CONST = 1, OP_ADD = 2, OP_SUB = 3, OP_MUL = 4, OP_DIV = 5, OP_EXP = 6 };

struct Node {
    int op;     // OpType
    int left;   // index of left child (or var/const index)
    int right;  // index of right child (or -1 for unary)
    double value; // for CONST
    int var_idx;  // for VAR
};

// -----------------------------
// Device interval arithmetic
// -----------------------------
__device__ Interval d_add(Interval a, Interval b) {
    return {a.lo + b.lo, a.hi + b.hi};
}
__device__ Interval d_sub(Interval a, Interval b) {
    return {a.lo - b.hi, a.hi - b.lo};
}
__device__ Interval d_mul(Interval a, Interval b) {
    double v1 = a.lo * b.lo;
    double v2 = a.lo * b.hi;
    double v3 = a.hi * b.lo;
    double v4 = a.hi * b.hi;
    double lo = fmin(fmin(v1, v2), fmin(v3, v4));
    double hi = fmax(fmax(v1, v2), fmax(v3, v4));
    return {lo, hi};
}
__device__ Interval d_div(Interval a, Interval b) {
    if (b.lo <= 0.0 && b.hi >= 0.0) {
        return {-INFINITY, INFINITY};
    }
    Interval inv = {1.0 / b.hi, 1.0 / b.lo};
    return d_mul(a, inv);
}
__device__ Interval d_exp(Interval a) {
    // exp is monotonic
    return {exp(a.lo), exp(a.hi)};
}

// Evaluate single function represented by `nodes` (num_nodes long)
__device__ Interval eval_function_device(const Node* nodes, int num_nodes, const Interval* vars) {
    // simple stack: allocate per-thread fixed size
    Interval stack_[128];

    for (int i = 0; i < num_nodes; ++i) {
        Node n = nodes[i];
        switch (n.op) {
            case OP_VAR:
                stack_[i] = vars[n.var_idx];
                break;
            case OP_CONST:
                stack_[i] = {n.value, n.value};
                break;
            case OP_ADD: {
                stack_[i] = d_add(stack_[n.left], stack_[n.right]);
                break;
            }
            case OP_SUB: {
                stack_[i] = d_sub(stack_[n.left], stack_[n.right]);
                break;
            }
            case OP_MUL: {
                stack_[i] = d_mul(stack_[n.left], stack_[n.right]);
                break;
            }
            case OP_DIV: {
                stack_[i] = d_div(stack_[n.left], stack_[n.right]);
                break;
            }
            case OP_EXP: {
                stack_[i] = d_exp(stack_[n.left]);
                break;
            }
            default:
                stack_[i] = {0.0, 0.0};
        }
    }

    return stack_[num_nodes - 1];
}

// Kernel: each thread computes one function
__global__ void evaluate_matrix_kernel(const Node* all_nodes, const int* offsets, const int* sizes,
                                       const Interval* vars, Interval* results, int num_funcs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_funcs) return;

    const Node* func_nodes = &all_nodes[offsets[idx]];
    int size = sizes[idx];
    results[idx] = eval_function_device(func_nodes, size, vars);
}

// -----------------------------
// Host: parser (tokenize, shunting-yard, postfix -> nodes)
// -----------------------------

enum TokenType { T_VAR, T_NUM, T_OP, T_LPAREN, T_RPAREN, T_FUNC };
struct Token { TokenType type; std::string value; };

static bool is_name_char(char c) {
    return std::isalpha((unsigned char)c) || (c == '_');
}

std::vector<Token> tokenize(const std::string& s) {
    std::vector<Token> out;
    size_t i = 0;
    while (i < s.size()) {
        char c = s[i];
        if (std::isspace((unsigned char)c)) { ++i; continue; }
        if (std::isdigit((unsigned char)c) || (c == '.')) {
            size_t j = i;
            while (j < s.size() && (std::isdigit((unsigned char)s[j]) || s[j] == '.')) ++j;
            out.push_back({T_NUM, s.substr(i, j - i)});
            i = j;
            continue;
        }
        if (is_name_char(c)) {
            size_t j = i;
            while (j < s.size() && (is_name_char(s[j]) || std::isdigit((unsigned char)s[j]))) ++j;
            std::string name = s.substr(i, j - i);
            // treat single-letter names as variables, multi-letter as function or variable
            if (name == "exp") out.push_back({T_FUNC, name});
            else out.push_back({T_VAR, name});
            i = j;
            continue;
        }
        // operators and parens
        if (c == '+' || c == '-' || c == '*' || c == '/') {
            out.push_back({T_OP, std::string(1, c)});
            ++i; continue;
        }
        if (c == '(') { out.push_back({T_LPAREN, "("}); ++i; continue; }
        if (c == ')') { out.push_back({T_RPAREN, ")"}); ++i; continue; }

        // unknown char
        ++i;
    }
    return out;
}

int prec_op(const std::string& op) {
    if (op == "+" || op == "-") return 1;
    if (op == "*" || op == "/") return 2;
    return 0;
}

std::vector<Token> infix_to_postfix(const std::vector<Token>& tokens) {
    std::vector<Token> out;
    std::stack<Token> ops;

    for (auto t : tokens) {
        if (t.type == T_NUM || t.type == T_VAR) {
            out.push_back(t);
        } else if (t.type == T_FUNC) {
            ops.push(t);
        } else if (t.type == T_OP) {
            while (!ops.empty() && ops.top().type == T_OP &&
                   prec_op(ops.top().value) >= prec_op(t.value)) {
                out.push_back(ops.top()); ops.pop();
            }
            ops.push(t);
        } else if (t.type == T_LPAREN) {
            ops.push(t);
        } else if (t.type == T_RPAREN) {
            while (!ops.empty() && ops.top().type != T_LPAREN) {
                out.push_back(ops.top()); ops.pop();
            }
            if (!ops.empty() && ops.top().type == T_LPAREN) ops.pop();
            if (!ops.empty() && ops.top().type == T_FUNC) { out.push_back(ops.top()); ops.pop(); }
        }
    }
    while (!ops.empty()) { out.push_back(ops.top()); ops.pop(); }
    return out;
}

// Build function-local Node vector from postfix tokens. Node indices are local (0..n-1)
std::vector<Node> build_nodes_from_postfix(const std::vector<Token>& postfix,
                                           const std::unordered_map<std::string,int>& var_map) {
    std::vector<Node> nodes;
    std::stack<int> st;
    for (auto t : postfix) {
        if (t.type == T_NUM) {
            int idx = (int)nodes.size();
            nodes.push_back({OP_CONST, -1, -1, std::stod(t.value), 0});
            st.push(idx);
        } else if (t.type == T_VAR) {
            auto it = var_map.find(t.value);
            if (it == var_map.end()) {
                std::cerr << "Unknown variable: " << t.value << std::endl;
                exit(1);
            }
            int idx = (int)nodes.size();
            nodes.push_back({OP_VAR, -1, -1, 0.0, it->second});
            st.push(idx);
        } else if (t.type == T_OP) {
            if (st.size() < 2) { std::cerr << "Bad postfix expression (binary op)\n"; exit(1); }
            int r = st.top(); st.pop();
            int l = st.top(); st.pop();
            int op = OP_ADD;
            if (t.value == "+") op = OP_ADD;
            else if (t.value == "-") op = OP_SUB;
            else if (t.value == "*") op = OP_MUL;
            else if (t.value == "/") op = OP_DIV;
            int idx = (int)nodes.size();
            nodes.push_back({op, l, r, 0.0, 0});
            st.push(idx);
        } else if (t.type == T_FUNC) {
            // unary function, pop one
            if (st.empty()) { std::cerr << "Bad postfix (func)\n"; exit(1); }
            int a = st.top(); st.pop();
            int idx = (int)nodes.size();
            if (t.value == "exp") nodes.push_back({OP_EXP, a, -1, 0.0, 0});
            else { std::cerr << "Unknown func: " << t.value << std::endl; exit(1); }
            st.push(idx);
        }
    }
    if (st.size() != 1) { std::cerr << "Postfix did not reduce to single value\n"; exit(1); }
    return nodes;
}

#ifdef USE_GAOL
// Evaluate postfix tokens using gaol intervals on CPU (rigorous)
Interval eval_postfix_gaol(const std::vector<Token>& postfix,
                           const std::unordered_map<std::string,int>& var_map,
                           const std::vector<Interval>& vars) {
    std::stack<gaol::interval> st;
    for (auto t : postfix) {
        if (t.type == T_NUM) {
            double v = std::stod(t.value);
            st.push(gaol::interval(v, v));
        } else if (t.type == T_VAR) {
            int vid = var_map.at(t.value);
            st.push(gaol::interval(vars[vid].lo, vars[vid].hi));
        } else if (t.type == T_OP) {
            auto r = st.top(); st.pop();
            auto l = st.top(); st.pop();
            if (t.value == "+") st.push(l + r);
            else if (t.value == "-") st.push(l - r);
            else if (t.value == "*") st.push(l * r);
            else if (t.value == "/") st.push(l / r);
        } else if (t.type == T_FUNC) {
            auto a = st.top(); st.pop();
            if (t.value == "exp") st.push(gaol::interval(exp(a.lower()), exp(a.upper())));
        }
    }
    gaol::interval res = st.top();
    Interval out; out.lo = res.lower(); out.hi = res.upper();
    return out;
}
#endif

// -----------------------------
// Host: main wiring everything
// -----------------------------
int main(int argc, char** argv) {
    // variable order / mapping
    std::vector<std::string> var_names = {"x", "y"};
    std::unordered_map<std::string,int> var_map;
    for (int i = 0; i < (int)var_names.size(); ++i) var_map[var_names[i]] = i;

    // functions as strings (matrix 2x2 => 4 functions)
    std::vector<std::string> funcs = {
        "x*x + exp(y)",
        "x*y - 3*y",
        "x - 4*y",
        "4"
    };

    // parse all functions
    std::vector<std::vector<Token>> postfix_funcs;
    std::vector<std::vector<Node>> nodes_per_func;

    for (auto &fs : funcs) {
        auto toks = tokenize(fs);
        auto pf = infix_to_postfix(toks);
        auto nodes_local = build_nodes_from_postfix(pf, var_map);
        postfix_funcs.push_back(pf);
        nodes_per_func.push_back(nodes_local);
    }

    // flatten nodes into one array and record offsets/sizes
    std::vector<Node> all_nodes;
    std::vector<int> offsets;
    std::vector<int> sizes;

    for (auto &v : nodes_per_func) {
        offsets.push_back((int)all_nodes.size());
        sizes.push_back((int)v.size());

        // 🔥 DO NOT shift indices
        for (int i = 0; i < (int)v.size(); ++i) {
            all_nodes.push_back(v[i]);
        }
    }

    int num_funcs = (int)funcs.size();

    // input variable intervals
    std::vector<Interval> vars = { {1.0, 2.0}, {0.0, 1.0} }; // x, y

    // Upload to GPU
    Node* d_nodes = nullptr;   int* d_offsets = nullptr; int* d_sizes = nullptr;
    Interval* d_vars = nullptr; Interval* d_results = nullptr;

    CUDA_CHECK(cudaMalloc(&d_nodes, all_nodes.size() * sizeof(Node)));
    CUDA_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sizes, sizes.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vars, vars.size() * sizeof(Interval)));
    CUDA_CHECK(cudaMalloc(&d_results, num_funcs * sizeof(Interval)));

    CUDA_CHECK(cudaMemcpy(d_nodes, all_nodes.data(), all_nodes.size() * sizeof(Node), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes, sizes.data(), sizes.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vars, vars.data(), vars.size() * sizeof(Interval), cudaMemcpyHostToDevice));

    // Launch kernel
    int threads = 256;
    int blocks = (num_funcs + threads - 1) / threads;
    evaluate_matrix_kernel<<<blocks, threads>>>(d_nodes, d_offsets, d_sizes, d_vars, d_results, num_funcs);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Copy back
    std::vector<Interval> results(num_funcs);
    CUDA_CHECK(cudaMemcpy(results.data(), d_results, num_funcs * sizeof(Interval), cudaMemcpyDeviceToHost));

    // Print as 2x2
    std::cout << "\nGPU Results (2x2):\n";
    for (int i = 0; i < num_funcs; ++i) {
        std::cout << "[" << results[i].lo << ", " << results[i].hi << "] ";
        if ((i+1) % 2 == 0) std::cout << "\n";
    }

#ifdef USE_GAOL
    std::cout << "\nRefinement with GAOL (CPU):\n";
    for (int i = 0; i < num_funcs; ++i) {
        Interval g = eval_postfix_gaol(postfix_funcs[i], var_map, vars);
        std::cout << "[" << g.lo << ", " << g.hi << "] ";
        if ((i+1) % 2 == 0) std::cout << "\n";
    }
#else
    std::cout << "\n(GAOL not enabled) - build with -DUSE_GAOL=ON and point GAOL_ROOT in CMake to enable rigorous refinement.\n";
#endif

    // free GPU
    cudaFree(d_nodes); cudaFree(d_offsets); cudaFree(d_sizes); cudaFree(d_vars); cudaFree(d_results);

    return 0;
}


// File: CMakeLists.txt
// Place next to main.cu. Configure GAOL with -DUSE_GAOL=ON and -DGAOL_ROOT=/path/to/gaol

