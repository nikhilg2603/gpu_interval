#include "gpu_interface.h"
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

void launch_gpu_kernel(Node* d_nodes, int* d_off, int* d_sz, Interval* d_vars, Interval* d_out, int F, int blocks, int threads) {
    // 1. Create CUDA timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 2. Start timer
    cudaEventRecord(start);

    // 3. Run the kernel
    evaluate_matrix_kernel<<<blocks, threads>>>(d_nodes, d_off, d_sz, d_vars, d_out, F);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for GPU to finish

    // 4. Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 5. Calculate and print time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "[GPU Kernel] Math Execution Time: " << milliseconds << " ms\n";

    // 6. Cleanup timers
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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


