// Libraries
//    - Ibex : Interval Arithmetic
//        - Gaol : Interval estimation from a function
//    - Eigen : Cholesky Decomposition
// References
//    - [0] C.A. Floudas! * C.S. Adjiman!, S. Dallwig” and A. Neumaier”. A global optimization method,
//      abb, for general twice-differentiable constrained nlps — i. theoretical advances. pages 1137–
//      1158. Department of Chemical Engineering, Princeton University, Princeton, NJ 08544, U.S.A
//      and Institut fur Mathematik, Universitat Wien, Strudlhofgasse 4, A-1090 Wien, Austria, 1997.
//    - [1] Jorge Nocedal and Stephen J. Wright. Numerical Optimization. Second Edition. Springer
//      Science+Business Media, LLC, 233 Spring Street, New York, NY 10013, USA, 2006
// Algorithms Implemented
//    - Diagonal Shift
//    - Gerschgorin at a point
//    - Optimization with Hessian modfication using Interval approximation(Alpha - O(n^2))
//    - Optimization with Hessian modfication using Interval approximation(Alpha - O(n^3))

// import libraries
#include "gpu_interface.h"
#include <cuda_runtime.h> // Allows g++ to see cudaMalloc and cudaMemcpy#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <vector>
#include <regex>
#include <map>
#include <limits>
#include <cmath>
#include "omp.h"
#include <cfenv>

#pragma STDC FENV_ACCESS ON

// #include "ibex.h"
#include <ginac/ginac.h>
#include <gaol/gaol.h>

#define EIGEN_NO_CUDA 1 
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/QR"


using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;
using namespace GiNaC;
// using namespace ibex;

// Converts Ibex Matrix to std c++ 2d vector
// vector<vector<double>> ConvertIbexMatrixTo2DVector(ibex::Matrix m)
// {
//     int n = m.nb_rows();
//     vector<vector<double>> dummyHessian(n, vector<double>(n));
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             dummyHessian[i][j] = m[i][j];
//         }
//     }
//     return dummyHessian;
// }
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        exit(code);
    }
}

int compile_ginac(const GiNaC::ex &e,
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

    if (is_a<GiNaC::function>(e))
    {
        std::string name = ex_to<GiNaC::function>(e).get_name();
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


// GAOL Modifications
std::string replaceVariables(const std::string &function, int num_variables)
{
    std::string modified_function = function;
    for (int i = 1; i <= num_variables; ++i)
    {
        std::ostringstream old_var, new_var;
        old_var << "x(" << i << ")";
        new_var << "x" << i;

        size_t pos = 0;
        while ((pos = modified_function.find(old_var.str(), pos)) != std::string::npos)
        {
            modified_function.replace(pos, old_var.str().length(), new_var.str());
            pos += new_var.str().length();
        }
    }
    return modified_function;
}

std::map<GiNaC::symbol, gaol::interval> createIntervalMap(const std::vector<gaol::interval> &xy,
                                                          const std::vector<GiNaC::realsymbol> &vars)
{
    std::map<GiNaC::symbol, gaol::interval> map;
    for (size_t i = 0; i < xy.size(); ++i)
    {
        map[vars[i]] = xy[i];
    }
    return map;
}

// Function to evaluate a symbolic expression with GAOL intervals
// gaol::interval evaluate_expression(const GiNaC::ex &expr, const std::map<GiNaC::symbol, gaol::interval> &variable_intervals)
// {
//     // Define a stack item that holds an expression and its evaluated result
//     struct StackItem {
//         GiNaC::ex expression;
//         gaol::interval result;
//         bool evaluated;
//         std::vector<gaol::interval> operand_results;
//         size_t next_operand_index;

//         StackItem(const GiNaC::ex &e) : expression(e), evaluated(false), next_operand_index(0) {}
//     };

//     std::stack<StackItem> eval_stack;
//     eval_stack.push(StackItem(expr));

//     while (!eval_stack.empty()) {
//         StackItem &current = eval_stack.top();

//         if (current.evaluated) {
//             // If this item is already evaluated, return its result
//             gaol::interval result = current.result;
//             eval_stack.pop();

//             if (!eval_stack.empty()) {
//                 // Store the result in the parent's operand results
//                 eval_stack.top().operand_results.push_back(result);
//             } else {
//                 // This is the final result
//                 return result;
//             }
//             continue;
//         }

//         // Handle different expression types
//         if (GiNaC::is_a<GiNaC::numeric>(current.expression)) {
//             // Convert numeric constant to an interval
//             double value = GiNaC::ex_to<GiNaC::numeric>(current.expression).to_double();
//             current.result = gaol::interval(value, value);
//             current.evaluated = true;
//         }
//         else if (GiNaC::is_a<GiNaC::symbol>(current.expression)) {
//             // Lookup the symbol in the variable intervals map
//             const GiNaC::symbol &sym = GiNaC::ex_to<GiNaC::symbol>(current.expression);
//             auto it = variable_intervals.find(sym);
//             if (it != variable_intervals.end()) {
//                 current.result = it->second;
//                 current.evaluated = true;
//             } else {
//                 throw std::runtime_error("Variable not found in the interval map.");
//             }
//         }
//         else if (GiNaC::is_a<GiNaC::add>(current.expression) ||
//                  GiNaC::is_a<GiNaC::mul>(current.expression) ||
//                  GiNaC::is_a<GiNaC::power>(current.expression) ||
//                  GiNaC::is_a<GiNaC::function>(current.expression)) {

//             // Check if all operands have been processed
//             if (current.next_operand_index < current.expression.nops()) {
//                 // Push the next operand onto the stack
//                 eval_stack.push(StackItem(current.expression.op(current.next_operand_index)));
//                 current.next_operand_index++;
//             } else {
//                 // All operands have been evaluated, compute the result
//                 if (GiNaC::is_a<GiNaC::add>(current.expression)) {
//                     // Handle addition
//                     gaol::interval result(0.0, 0.0);
//                     for (const auto& operand_result : current.operand_results) {
//                         result = result + operand_result;
//                     }
//                     current.result = result;
//                 }
//                 else if (GiNaC::is_a<GiNaC::mul>(current.expression)) {
//                     // Handle multiplication
//                     if (!current.operand_results.empty()) {
//                         gaol::interval result = current.operand_results[0];
//                         for (size_t i = 1; i < current.operand_results.size(); ++i) {
//                             result = result * current.operand_results[i];
//                         }
//                         current.result = result;
//                     } else {
//                         current.result = gaol::interval(1.0, 1.0); // Empty product is 1
//                     }
//                 }
//                 else if (GiNaC::is_a<GiNaC::power>(current.expression)) {
//                     // Handle power (base^exponent)
//                     if (current.operand_results.size() == 2) {
//                         gaol::interval base = current.operand_results[0];
//                         gaol::interval exponent = current.operand_results[1];

//                         // Ensure the exponent is constant
//                         if (exponent.left() == exponent.right()) {
//                             double exp_value = exponent.left();
//                             current.result = gaol::pow(base, exp_value);
//                         } else {
//                             throw std::runtime_error("Non-constant interval exponents are not supported.");
//                         }
//                     } else {
//                         throw std::runtime_error("Power operation requires exactly 2 operands.");
//                     }
//                 }
//                 else if (GiNaC::is_a<GiNaC::function>(current.expression)) {
//                     // Handle functions
//                     const GiNaC::function &f = GiNaC::ex_to<GiNaC::function>(current.expression);
//                     if (current.operand_results.size() == 1) {
//                         gaol::interval arg = current.operand_results[0];

//                         if (f.get_name() == "exp") {
//                             current.result = gaol::exp(arg);
//                         }
//                         else if (f.get_name() == "sin") {
//                             current.result = gaol::sin(arg);
//                         }
//                         else if (f.get_name() == "cos") {
//                             current.result = gaol::cos(arg);
//                         }
//                         else if (f.get_name() == "tan") {
//                             current.result = gaol::tan(arg);
//                         }
//                         else {
//                             throw std::runtime_error("Unsupported function: " + f.get_name());
//                         }
//                     } else {
//                         throw std::runtime_error("Function requires exactly 1 argument.");
//                     }
//                 }

//                 current.evaluated = true;
//             }
//         }
//         else {
//             throw std::runtime_error("Unsupported operation encountered in the expression.");
//         }
//     }

//     // This should never be reached if the algorithm is correct
//     throw std::runtime_error("Unexpected end of evaluation.");
// }

gaol::interval evaluate_expression(const GiNaC::ex &expr, const std::map<std::string, gaol::interval> &variable_intervals)
{
    // std::cout << "expression is" << expr << "\n";
    if (GiNaC::is_a<GiNaC::numeric>(expr))
    {
        // Convert numeric constant to an interval
        double value = GiNaC::ex_to<GiNaC::numeric>(expr).to_double();
        return gaol::interval(value, value);
    }
    else if (GiNaC::is_a<GiNaC::symbol>(expr))
    {
        // Lookup the symbol in the variable intervals map
        const GiNaC::symbol &sym = GiNaC::ex_to<GiNaC::symbol>(expr);
        // for (const auto &entry : variable_intervals)
        // {
        //     cout << entry.first << " " << entry.second << " " << sym.get_name() << "\n";
        //     if (entry.first.get_name() == sym.get_name())
        //         return entry.second;
        // }
        // throw std::runtime_error("Variable " + sym.get_name() + " not found in the interval map.");
        auto it = variable_intervals.find(sym.get_name());
        // cout << it->first << " ";
        if (it != variable_intervals.end())
        {
            return it->second;
        }
        else
        {
            throw std::runtime_error("Variable not found in the interval map.");
        }
    }
    else if (GiNaC::is_a<GiNaC::add>(expr))
    {
        // Handle addition
        gaol::interval result(0.0, 0.0);
        for (size_t i = 0; i < expr.nops(); ++i)
        {
            result = result + evaluate_expression(expr.op(i), variable_intervals);
        }
        return result;
    }
    else if (GiNaC::is_a<GiNaC::mul>(expr))
    {
        // Handle multiplication
        gaol::interval result = evaluate_expression(expr.op(0), variable_intervals);
        for (size_t i = 1; i < expr.nops(); ++i)
        {
            result = result * evaluate_expression(expr.op(i), variable_intervals); // Correct multiplication
        }
        return result;
    }
    else if (GiNaC::is_a<GiNaC::power>(expr))
    {
        // Handle power (expr^exp)
        gaol::interval base = evaluate_expression(expr.op(0), variable_intervals);
        gaol::interval exponent = evaluate_expression(expr.op(1), variable_intervals);

        // Ensure the exponent is constant
        if (std::fabs(exponent.right() - exponent.left()) < 1e-5)
        {
            double exp_value = exponent.left();
            return gaol::pow(base, exp_value); // Compute power
        }
        else
        {
            throw std::runtime_error("Non-constant interval exponents are not supported.");
        }
    }
    else if (GiNaC::is_a<GiNaC::function>(expr))
    {
        // Handle functions like exp, sin, cos, tan
        const GiNaC::function &f = GiNaC::ex_to<GiNaC::function>(expr);
        if (f.get_name() == "exp")
        {
            return gaol::exp(evaluate_expression(expr.op(0), variable_intervals));
        }
        else if (f.get_name() == "sin")
        {
            return gaol::sin(evaluate_expression(expr.op(0), variable_intervals));
        }
        else if (f.get_name() == "cos")
        {
            return gaol::cos(evaluate_expression(expr.op(0), variable_intervals));
        }
        else if (f.get_name() == "tan")
        {
            return gaol::tan(evaluate_expression(expr.op(0), variable_intervals));
        }
        else
        {
            throw std::runtime_error("Unsupported function: " + f.get_name());
        }
    }
    else
    {
        throw std::runtime_error("Unsupported operation encountered in the expression.");
    }
}

// Functions to convert expression into a hessian and gradient

double evaluate_function_at_point(const GiNaC::ex &expr,
                                  const Eigen::VectorXd &point,
                                  const std::vector<GiNaC::realsymbol> &vars)
{
    GiNaC::lst substitutions;
    for (int i = 0; i < point.size(); ++i)
    {
        substitutions.append(vars[i] == point[i]);
    }
    return GiNaC::ex_to<GiNaC::numeric>(expr.subs(substitutions)).to_double();
}

Eigen::VectorXd evaluate_gradient_at_point(const std::vector<GiNaC::ex> &grad_exprs,
                                           const Eigen::VectorXd &point,
                                           const std::vector<GiNaC::realsymbol> &vars)
{
    int n = point.size();
    Eigen::VectorXd grad(n);
    GiNaC::lst substitutions;
    for (int i = 0; i < n; ++i)
    {
        substitutions.append(vars[i] == point[i]);
    }
    for (int i = 0; i < n; ++i)
    {
        grad[i] = GiNaC::ex_to<GiNaC::numeric>(grad_exprs[i].subs(substitutions)).to_double();
    }
    return grad;
}

Eigen::MatrixXd evaluate_hessian_at_point(const Eigen::Matrix<GiNaC::ex, Eigen::Dynamic, Eigen::Dynamic> &hess_exprs,
                                          const Eigen::VectorXd &point,
                                          const std::vector<GiNaC::realsymbol> &vars)
{
    int n = point.size();
    Eigen::MatrixXd hess(n, n);
    GiNaC::lst substitutions;
    for (int i = 0; i < n; ++i)
    {
        substitutions.append(vars[i] == point[i]);
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            hess(i, j) = GiNaC::ex_to<GiNaC::numeric>(hess_exprs(i, j).subs(substitutions)).to_double();
            // cout << hess(i, j) << " ";
        }
    }
    return hess;
}

gaol::interval evaluate_function_on_interval(const GiNaC::ex &expr,
                                             const std::map<string, gaol::interval> &intervals)
{
    return evaluate_expression(expr, intervals);
}

std::vector<gaol::interval> evaluate_gradient_on_interval(const std::vector<GiNaC::ex> &grad_exprs,
                                                          const std::map<std::string, gaol::interval> &intervals)
{
    int n = grad_exprs.size();
    std::vector<gaol::interval> grad(n);
    for (int i = 0; i < n; ++i)
    {
        grad[i] = evaluate_expression(grad_exprs[i], intervals);
    }
    return grad;
}

std::vector<std::vector<gaol::interval>> evaluate_hessian_on_interval(const Eigen::Matrix<GiNaC::ex, Eigen::Dynamic, Eigen::Dynamic> &hess_exprs,
                                                                      const std::map<std::string, gaol::interval> &intervals)
{
    int n = hess_exprs.rows();
    std::vector<std::vector<gaol::interval>> hess(n, std::vector<gaol::interval>(n));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            hess[i][j] = evaluate_expression(hess_exprs(i, j), intervals);
        }
    }
    return hess;
}

// Converts std c++ 2d vector to Eigen Matrix
Eigen::MatrixXd ConvertToEigenMatrix(std::vector<std::vector<double>> data)
{
    Eigen::MatrixXd eMatrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i)
        eMatrix.row(i) = Eigen::VectorXd::Map(&data[i][0], data[0].size());
    return eMatrix;
}

// Calculates distance between old_vec and new_vec
// double normOfVector(Vector vec, int n)
// {
//     double nrm = 0;
//     for (int i = 0; i < n; i++)
//     {
//         nrm += ((vec[i]) * (vec[i]));
//     }
//     return sqrt(nrm);
// }

// Use Eigen's built-in norm() method:
double normOfVector(const Eigen::VectorXd &vec)
{
    return vec.norm();
}

// Returs Direction Vector from hessian and gradient (-1*Hessian*grad)
// Vector DirectionVector(Eigen::MatrixXd hessian, Vector grad)
// {
//     int n = grad.size();

//     Vector direcV(Vector ::zeros(n));
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             direcV[i] += (-1 * hessian(i, j)) * grad[j];
//         }
//     }
//     return direcV;
// }

Eigen::VectorXd DirectionVector(const Eigen::MatrixXd &hessian, const Eigen::VectorXd &grad)
{
    return -hessian * grad;
}

// Returns Gradient vector from gradient function(gradient) and point(xk)
// Vector gradVector(Function gradient, Vector xk)
// {
//     int n = xk.size();

//     Vector gradV(n);
//     for (int i = 0; i < n; i++)
//     {
//         IntervalVector result = gradient[i].eval(xk);
//         gradV[i] = result.lb()[0];
//     }
//     return gradV;
// }

// Eigen::VectorXd gradVector(const std::function<std::vector<gaol::interval>(const Eigen::VectorXd&)>& gradient,
//                              const Eigen::VectorXd &xk)
// {
//     int n = xk.size();
//     Eigen::VectorXd gradV(n);
//     std::vector<gaol::interval> result = gradient(xk);
//     for (int i = 0; i < n; i++)
//     {
//         gradV[i] = result[i].left(); // use left() as lower bound (or whichever you need)
//     }
//     return gradV;
// }

// Returns Hessian Matrix from double differentiation function(dff) and point(xk)
// Matrix HessianMatrix(Function dff, Vector xk)
// {
//     int n = xk.size();
//     Matrix hessian(n, n);
//     for (int i = 0; i < n; i++)
//     {
//         // Function new_f(gradient[i], Function::DIFF);
//         for (int j = i; j < n; j++)
//         {
//             IntervalVector result = dff[i][j].eval(xk);
//             hessian[i][j] = hessian[j][i] = result.lb()[0];
//         }
//     }
//     return hessian;
// }

// Compute Lower bound on minimim eigen value of Interval Matrix(Im)
// double minEigenValueIntervalMatrix(IntervalMatrix &Im)
// {
//     int sz = Im.nb_rows();
//     double min_eigen = POS_INFINITY;
//     for (int i = 0; i < sz; i++)
//     {
//         double sum = 0;
//         for (int j = 0; j < sz; j++)
//         {
//             if (j != i)
//             {
//                 sum += max(abs(Im[i][j].lb()), abs(Im[i][j].ub()));
//             }
//         }
//         min_eigen = min(min_eigen, Im[i][i].lb() - sum);
//     }
//     return min_eigen;
// }
double minEigenValueIntervalMatrix(const std::vector<std::vector<gaol::interval>> &Im)
{
    int sz = Im.size();
    double min_eigen = std::numeric_limits<double>::infinity();
    for (int i = 0; i < sz; i++)
    {
        double sum = 0;
        for (int j = 0; j < sz; j++)
        {
            if (j != i)
            {
                double a = std::abs(Im[i][j].left());
                double b = std::abs(Im[i][j].right());
                sum += std::max(a, b);
            }
        }
        double diag = Im[i][i].left();
        min_eigen = std::min(min_eigen, diag - sum);
    }
    return min_eigen;
}

// [0] Pg 1145
// Matrix CalculateModifiedMidPointMatrix(IntervalMatrix &Im)
// {
//     int sz = Im.nb_rows();
//     Matrix ModifiedMatrix(sz, sz);
//     for (int i = 0; i < sz; i++)
//     {
//         for (int j = 0; j < sz; j++)
//         {
//             if (i == j)
//             {
//                 ModifiedMatrix[i][j] = Im[i][j].lb();
//             }
//             else
//             {
//                 ModifiedMatrix[i][j] = (Im[i][j].lb() + Im[i][j].ub()) / 2;
//             }
//         }
//     }
//     return ModifiedMatrix;
// }
// Im is now an Eigen matrix with scalar type gaol::interval.
Eigen::MatrixXd CalculateModifiedMidPointMatrix(const std::vector<std::vector<gaol::interval>> &Im)
{
    int sz = Im.size();
    Eigen::MatrixXd ModifiedMatrix(sz, sz);
    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            if (i == j)
            {
                ModifiedMatrix(i, j) = Im[i][j].left();
            }
            else
            {
                ModifiedMatrix(i, j) = (Im[i][j].left() + Im[i][j].right()) / 2.0;
            }
        }
    }
    return ModifiedMatrix;
}

// [0] Pg 1145
// Matrix CalculateEMatrix(IntervalMatrix &Im)
// {
//     int sz = Im.nb_rows();
//     Matrix ModifiedMatrix(sz, sz);
//     for (int i = 0; i < sz; i++)
//     {
//         for (int j = 0; j < sz; j++)
//         {
//             if (i == j)
//             {
//                 ModifiedMatrix[i][j] = (Im[i][j].ub() - Im[i][j].lb()) / 2;
//             }
//             else
//             {
//                 ModifiedMatrix[i][j] = 0;
//             }
//         }
//     }
//     return ModifiedMatrix;
// }

Eigen::MatrixXd CalculateEMatrix(const std::vector<std::vector<gaol::interval>> &Im)
{
    int sz = Im.size();
    Eigen::MatrixXd ModifiedMatrix(sz, sz);
    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            if (i == j)
            {
                ModifiedMatrix(i, j) = (Im[i][j].right() - Im[i][j].left()) / 2.0;
            }
            else
            {
                ModifiedMatrix(i, j) = 0.0;
            }
        }
    }
    return ModifiedMatrix;
}

// [0] Pg 1145
// Matrix CalculateModifiedRadiusMatrix(IntervalMatrix &Im)
// {
//     int sz = Im.nb_rows();
//     Matrix ModifiedMatrix(sz, sz);
//     for (int i = 0; i < sz; i++)
//     {
//         for (int j = 0; j < sz; j++)
//         {
//             if (i == j)
//             {
//                 ModifiedMatrix[i][j] = 0;
//             }
//             else
//             {
//                 ModifiedMatrix[i][j] = (Im[i][j].ub() - Im[i][j].lb()) / 2;
//             }
//         }
//     }
//     return ModifiedMatrix;
// }

Eigen::MatrixXd CalculateModifiedRadiusMatrix(const std::vector<std::vector<gaol::interval>> &Im)
{
    int sz = Im.size();
    Eigen::MatrixXd ModifiedMatrix(sz, sz);
    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            if (i == j)
            {
                ModifiedMatrix(i, j) = 0.0;
            }
            else
            {
                ModifiedMatrix(i, j) = (Im[i][j].right() - Im[i][j].left()) / 2.0;
            }
        }
    }
    return ModifiedMatrix;
}

// Compute Lower bound on minimim eigen value of Matrix(mt) - https://www.scirp.org/journal/paperinformation.aspx?paperid=103295
// double calcLowerBoundEigenValue(Matrix mt)
// {
//     int sz = mt.nb_rows();
//     double min_eigen = POS_INFINITY;
//     for (int i = 0; i < sz; i++)
//     {
//         double sum = 0;
//         for (int j = 0; j < sz; j++)
//         {
//             if (j != i)
//             {
//                 sum += abs(mt[i][j]);
//             }
//         }
//         min_eigen = min(min_eigen, mt[i][i] - sum);
//     }
//     return min_eigen;
// }

double UpdateDelta(double delta, double direction_sum, double direction_sq_sum, int n)
{
    return min(10.0, max(1e-3, delta * 2 * (direction_sum) / (sqrt(n) * sqrt(direction_sq_sum * direction_sq_sum + 1))));
}

double calcLowerBoundEigenValue(const Eigen::MatrixXd &mt)
{
    int sz = mt.rows();
    double min_eigen = std::numeric_limits<double>::infinity();
    for (int i = 0; i < sz; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < sz; j++)
        {
            if (j != i)
            {
                sum += std::abs(mt(i, j));
            }
        }
        min_eigen = std::min(min_eigen, mt(i, i) - sum);
    }
    return min_eigen;
}

// Compute Upper bound on max eigen value of Matrix(mt) - https://www.scirp.org/journal/paperinformation.aspx?paperid=103295
// double calcUpperBoundEigenValue(Matrix mt)
// {
//     int sz = mt.nb_rows();
//     double max_eigen = POS_INFINITY;
//     for (int i = 0; i < sz; i++)
//     {
//         double sum = 0;
//         for (int j = 0; j < sz; j++)
//         {
//             if (j != i)
//             {
//                 sum += abs(mt[i][j]);
//             }
//         }
//         max_eigen = max(max_eigen, mt[i][i] + sum);
//     }
//     return max_eigen;
// }

double calcUpperBoundEigenValue(const Eigen::MatrixXd &mt)
{
    int sz = mt.rows();
    double max_eigen = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < sz; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < sz; j++)
        {
            if (j != i)
            {
                sum += std::abs(mt(i, j));
            }
        }
        max_eigen = std::max(max_eigen, mt(i, i) + sum);
    }
    return max_eigen;
}

// https://en.wikipedia.org/wiki/Spectral_radius#:~:text=In%20mathematics%2C%20the%20spectral%20radius,denoted%20by%20%CF%81(%C2%B7).
// double spectralRadius(Matrix mt)
// {
//     double lower_bound = calcLowerBoundEigenValue(mt);
//     double upper_bound = calcUpperBoundEigenValue(mt);

//     return max(abs(lower_bound), abs(upper_bound));
// }

double spectralRadius(const Eigen::MatrixXd &mt)
{
    double lower_bound = calcLowerBoundEigenValue(mt);
    double upper_bound = calcUpperBoundEigenValue(mt);
    return std::max(std::abs(lower_bound), std::abs(upper_bound));
}

// [0] Pg 1145
// double On3MinEigenValueIntervalMatrix(IntervalMatrix &Im)
// {
//     Matrix ModifiedMPMatrix = CalculateModifiedMidPointMatrix(Im);
//     Matrix EMatrix = CalculateEMatrix(Im);
//     Matrix ModifiedRadiusMatrix = CalculateModifiedRadiusMatrix(Im);
//     double sp = spectralRadius(ModifiedRadiusMatrix + EMatrix);
//     double lb = calcLowerBoundEigenValue(ModifiedMPMatrix + EMatrix);

//     return lb - sp;
// }

double On3MinEigenValueIntervalMatrix(const std::vector<std::vector<gaol::interval>> &Im)
{
    Eigen::MatrixXd ModifiedMPMatrix = CalculateModifiedMidPointMatrix(Im);
    Eigen::MatrixXd EMatrix = CalculateEMatrix(Im);
    Eigen::MatrixXd ModifiedRadiusMatrix = CalculateModifiedRadiusMatrix(Im);
    double sp = spectralRadius(ModifiedRadiusMatrix + EMatrix);
    double lb = calcLowerBoundEigenValue(ModifiedMPMatrix + EMatrix);
    return lb - sp;
}

// Inverse of Hessian Matrix
// Eigen::MatrixXd InverseMatrix(Matrix hessian)
// {
//     int n = hessian.nb_rows();
//     Eigen::MatrixXd hessianInverse(n, n);

//     for (int i = 0; i < n; i++)
//     {
//         for (int j = i; j < n; j++)
//         {
//             hessianInverse(i, j) = hessianInverse(j, i) = hessian[i][j];
//         }
//     }
//     // cout << "Inverse: " << hessianInverse << "\n";
//     hessianInverse = hessianInverse.inverse();
//     // cout << "Inverse: " << hessianInverse << "\n";
//     return hessianInverse;
// }

// Compute inverse using Eigen's inverse() method:
Eigen::MatrixXd InverseMatrix(const Eigen::MatrixXd &hessian)
{
    return hessian.inverse();
}

//
// bool checkCholesky(Matrix hessian)
// {
//     vector<vector<double>> dummyHessian = ConvertIbexMatrixTo2DVector(hessian);
//     Eigen::LLT<Eigen::MatrixXd> lltOfA(ConvertToEigenMatrix(dummyHessian)); // compute the Cholesky decomposition of Hessian
//     Eigen::ComputationInfo m_info = lltOfA.info();
//     if (m_info == 1)
//     {
//         return true;
//         // return 0;
//     }
//     else
//     {
//         return false;
//     }
// }

// Matrix ModifyHessianDS(Matrix hessian)
// {
//     int n = hessian.nb_rows();
//     Matrix IDMatrix = Matrix::eye(n);
//     int cnt = 0;
//     while (checkCholesky(hessian) && cnt < 100000)
//     {
//         cnt++;
//         hessian = hessian + IDMatrix;
//     }
//     return hessian;
// }
bool checkCholesky(const Eigen::MatrixXd &hessian)
{
    Eigen::LLT<Eigen::MatrixXd> lltOfA(hessian); // compute the Cholesky decomposition of Hessian
    Eigen::ComputationInfo m_info = lltOfA.info();
    if (m_info == 1)
    {
        return true;
        // return 0;
    }
    else
    {
        return false;
    }
}

Eigen::MatrixXd ModifyHessianDS(Eigen::MatrixXd hessian)
{
    int n = hessian.rows();
    Eigen::MatrixXd IDMatrix = Eigen::MatrixXd::Identity(n, n);
    int cnt = 0;
    // Continue adding the identity matrix until the Hessian becomes positive definite,
    // or until the maximum iteration count is reached.
    while (!checkCholesky(hessian) && cnt < 100000)
    {
        cnt++;
        hessian = hessian + IDMatrix;
    }
    return hessian;
}

// Matrix ModifyHessianGerschgorin(Matrix hessian)
// {
//     double min_eigen = POS_INFINITY;
//     int n = hessian.nb_rows();
//     for (int i = 0; i < n; i++)
//     {
//         double sum = 0;
//         for (int j = 0; j < n; j++)
//         {
//             if (j != i || j == 0)
//             {
//                 sum += abs(hessian[i][j]);
//             }
//         }
//         min_eigen = min(min_eigen, hessian[i][i] - sum);
//     }

//     if (min_eigen < 0)
//     {
//         hessian = hessian + (-1 * min_eigen) * (Matrix::eye(n));
//     }

//     return hessian;
// }

Eigen::MatrixXd ModifyHessianGerschgorin(const Eigen::MatrixXd &hessian_input)
{
    // Make a copy of the input Hessian.
    Eigen::MatrixXd hessian = hessian_input;
    int n = hessian.rows();
    double min_eigen = std::numeric_limits<double>::infinity();

    // For each row, compute the Gerschgorin bound.
    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            if (j != i)
            {
                sum += std::abs(hessian(i, j));
            }
        }
        // Compute the lower bound for the i-th row.
        double row_bound = hessian(i, i) - sum;
        min_eigen = std::min(min_eigen, row_bound);
    }

    // If the minimum eigenvalue bound is negative, apply a diagonal shift.
    if (min_eigen < 0)
    {
        hessian = hessian + (-min_eigen) * Eigen::MatrixXd::Identity(n, n);
    }

    return hessian;
}

// Matrix FindTestVectors(int p, int n) no use in this file
// {
//     return ibex::Matrix::rand(p, n);
// }

// Helper: Create an Eigen matrix of GAOL intervals from a point vector xk and step alp.
// Here, each row i gets an interval [ xk[i] - alp*(range_ul - range_ll), xk[i] + alp*(range_ul - range_ll) ]
std::vector<gaol::interval> createIntervalVector(const Eigen::VectorXd &xk, double alp, double range_ll, double range_ul)
{
    int m = xk.size();
    std::vector<gaol::interval> intervals(m);
    for (int i = 0; i < m; ++i)
    {
        double lower = std::max(range_ll, xk[i] - alp * (range_ul - range_ll));
        double upper = std::min(range_ul, xk[i] + alp * (range_ul - range_ll));
        intervals[i] = gaol::interval(lower, upper);
    }
    return intervals;
}

// Create an Eigen matrix of GAOL intervals (for Hessian evaluation)
// We assume that 'dff' is provided as an Eigen::Matrix<GiNaC::ex, Dynamic, Dynamic> of second derivative expressions.
// 'intervals_map' is a mapping from each variable (GiNaC::symbol) to its GAOL interval.
std::vector<std::vector<gaol::interval>>
computeIntervalHessian(const Eigen::Matrix<GiNaC::ex, Eigen::Dynamic, Eigen::Dynamic> &dff,
                       const std::map<std::string, gaol::interval> &intervals)
{
    int m = dff.rows();
    std::vector<std::vector<gaol::interval>> im(m, std::vector<gaol::interval>(m));
    for (int i = 0; i < m; ++i)
    {
        for (int j = i; j < m; ++j)
        {
            im[i][j] = im[j][i] = evaluate_expression(dff(i, j), intervals);
        }
    }
    return im;
}


// Eigen::MatrixXd ModifyHessianOn2(const std::string &func,
//                                  const Eigen::VectorXd &xk, double alp,
//                                  const Eigen::MatrixXd &hessian,
//                                  const Eigen::Matrix<GiNaC::ex, Eigen::Dynamic, Eigen::Dynamic> &hessian_expressions,
//                                  double range_ll, double range_ul,
//                                  const std::vector<GiNaC::realsymbol> &vars,
//                                  const Eigen::VectorXd &grad) // <-- Added grad input
// {
//     int m = xk.size();
//     std::vector<gaol::interval> xy(m);
//     for (int i = 0; i < m; i++)
//     {
//         double lower = xk[i] - alp / 2.0;
//         double upper = xk[i] + alp / 2.0;
//         xy[i] = gaol::interval(lower, upper);
//         // cout << xy[i] << endl;
//     }

//     std::map<std::string, gaol::interval> interval_map;
//     for (int i = 0; i < m; i++)
//     {
//         interval_map[vars[i].get_name()] = xy[i];
//     }

//     Eigen::Matrix<gaol::interval, Eigen::Dynamic, Eigen::Dynamic> im(m, m);
//     for (int i = 0; i < m; i++)
//     {
//         for (int j = i; j < m; j++)
//         {
//             im(i, j) = im(j, i) = evaluate_expression(hessian_expressions(i, j), interval_map);
//         }
//     }

//     double lowerBoundEigenValue = minEigenValueIntervalMatrix(im);
//     double alpha = std::max(0.0, -0.5 * lowerBoundEigenValue);
//    // Modification for positive definite_KK
//     double sigma = 1;
//     double coef_grad = 0.001;
//     double g_til = std::pow(grad.norm(), sigma);

// static bool printed_once = false;
// if(!printed_once){
//     std::cout << "\n=== PARAMETERS USED ===\n";
//     std::cout << "coef_grad = " << coef_grad << "\n";
//     std::cout << "sigma     = " << sigma     << "\n";
//     printed_once = true;
// }

//     // cout << "alpha :" << alpha << "\n";
//     Eigen::MatrixXd modified_hessian = hessian +  (2 * alpha + coef_grad * g_til) * Eigen::MatrixXd::Identity(xk.size(), xk.size());
//     // std::cout << ">>> DEBUG TEST <<<" << std::endl;
//     // std::cout << ">>> USING ON3 + g_til modification <<< g_til = " << g_til << std::endl;
//     // KK
//     return modified_hessian;
// }

Eigen::MatrixXd ModifyHessianOn2(const std::string &func,
                                 const Eigen::VectorXd &xk, double alp,
                                 const Eigen::MatrixXd &hessian,
                                 const Eigen::Matrix<GiNaC::ex, Eigen::Dynamic, Eigen::Dynamic> &hessian_expressions,
                                 double range_ll, double range_ul,
                                 const std::vector<GiNaC::realsymbol> &vars,
                                 const Eigen::VectorXd &grad,
                                 Node* d_nodes, int* d_off, int* d_sz, Interval* d_vars, Interval* d_out,
                                 double &total_gpu_time) 
{
    int m = xk.size();
    
    // 1. Prepare Interval array for the current iteration
    std::vector<Interval> host_vars(m);
    for (int i = 0; i < m; i++)
    {
        double lower = xk[i] - alp / 2.0;
        double upper = xk[i] + alp / 2.0;
        host_vars[i] = {lower, upper};
    }
    auto t_start = std::chrono::high_resolution_clock::now();
    // 2. Copy dynamic variables to GPU
    CUDA_CHECK(cudaMemcpy(d_vars, host_vars.data(), m * sizeof(Interval), cudaMemcpyHostToDevice));

    // 3. Launch Kernel
    int F = m * m; // Total functions in the nxn Hessian matrix
    int threads = 256;
    int blocks = (F + threads - 1) / threads;

    // auto t1 = std::chrono::high_resolution_clock::now();
    launch_gpu_kernel(d_nodes, d_off, d_sz, d_vars, d_out, F, blocks, threads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // auto t2 = std::chrono::high_resolution_clock::now();

    // 4. Retrieve Results
    std::vector<Interval> gpu_res(F);
    CUDA_CHECK(cudaMemcpy(gpu_res.data(), d_out, F * sizeof(Interval), cudaMemcpyDeviceToHost));
    
    auto t_end = std::chrono::high_resolution_clock::now();
    total_gpu_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
    // std::cout << "[Total Offload] Memcpy + Kernel Time: " << total_gpu_time << " ms\n";

    // 5. Map 1D flat results back to a standard 2D vector Interval Matrix
    std::vector<std::vector<gaol::interval>> im(m, std::vector<gaol::interval>(m));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            im[i][j] = gaol::interval(gpu_res[i * m + j].lo, gpu_res[i * m + j].hi);
        }
    }

    double lowerBoundEigenValue = minEigenValueIntervalMatrix(im);
    double alpha = std::max(0.0, -0.5 * lowerBoundEigenValue);
    
    double sigma = 1;
    double coef_grad = 0.001;
    double g_til = std::pow(grad.norm(), sigma);

    Eigen::MatrixXd modified_hessian = hessian + (2 * alpha + coef_grad * g_til) * Eigen::MatrixXd::Identity(xk.size(), xk.size());
    return modified_hessian;
}

// Eigen::MatrixXd ModifyHessian(Eigen::MatrixXd hessian, int algo, std::string func,
//                               Eigen::VectorXd xk, double alp,
//                               Eigen::Matrix<GiNaC::ex, Eigen::Dynamic, Eigen::Dynamic> &hessian_expressions,
//                               double range_ll, double range_ul,
//                               std::vector<GiNaC::realsymbol> &vars,
//                               Eigen::VectorXd grad)
// {
//     switch (algo)
//     {
//     case 0:
//         return ModifyHessianGerschgorin(hessian);
//     case 1:
//         return ModifyHessianDS(hessian);
//     case 2:
//         return ModifyHessianOn2(func, xk, alp, hessian, hessian_expressions, range_ll, range_ul, vars, grad); // Pass grad
//     // case 3:
//     //     return ModifyHessianOn3(xk, alp, hessian, hessian_expressions, range_ll, range_ul, vars);
//     default:
//         return hessian;
//     }
// }

Eigen::MatrixXd ModifyHessian(Eigen::MatrixXd hessian, int algo, std::string func,
                              Eigen::VectorXd xk, double alp,
                              Eigen::Matrix<GiNaC::ex, Eigen::Dynamic, Eigen::Dynamic> &hessian_expressions,
                              double range_ll, double range_ul,
                              std::vector<GiNaC::realsymbol> &vars,
                              Eigen::VectorXd grad,
                              Node* d_nodes, int* d_off, int* d_sz, Interval* d_vars, Interval* d_out,
                              double &total_gpu_time)
{
    switch (algo)
    {
    case 0:
        return ModifyHessianGerschgorin(hessian);
    case 1:
        return ModifyHessianDS(hessian);
    case 2:
        return ModifyHessianOn2(func, xk, alp, hessian, hessian_expressions, range_ll, range_ul, vars, grad, d_nodes, d_off, d_sz, d_vars, d_out, total_gpu_time);
    default:
        return hessian;
    }
}

// Generates random number between fMin and fMax
double random(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

// bool CheckCacheValidity(double range_ll, double range_ul, double alp, Eigen::VectorXd xk, Eigen::VectorXd xkn)
// {
//     int m = xk.size();
//     double _x[m][2];
//     for (int i = 0; i < m; i++)
//     {
//         _x[i][0] = xk[i] - alp * 0.5;
//         _x[i][1] = xk[i] + alp * 0.5;
//     }
//     IntervalVector currentInterval(m, _x);
//     for (int i = 0; i < xk.size(); i++)
//     {
//         if (!currentInterval[i].contains(xkn[i]))
//         {
//             return false;
//         }
//     }
//     return true;
// }

bool CheckCacheValidity(double range_ll, double range_ul, double alp,
                        const Eigen::VectorXd &xk, const Eigen::VectorXd &xkn,
                        const Eigen::VectorXd &grad)
{
    int m = xk.size();
    std::vector<gaol::interval> currentInterval(m);

    for (int i = 0; i < m; i++)
    {
        double lower = xk[i] - alp * 0.5;
        double upper = xk[i] + alp * 0.5;
        currentInterval[i] = gaol::interval(lower, upper);
    }

    for (int i = 0; i < m; i++)
    {
        if (xkn[i] < currentInterval[i].left() || xkn[i] > currentInterval[i].right())
        {
            return false;
        }
    }
    return true;
}

bool numericAwareCompare(const std::string &a, const std::string &b)
{
    auto is_digit = [](char c)
    { return std::isdigit(c) != 0; };
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size())
    {
        if (is_digit(a[i]) && is_digit(b[j]))
        {
            size_t start_i = i, start_j = j;
            while (i < a.size() && is_digit(a[i]))
                i++;
            while (j < b.size() && is_digit(b[j]))
                j++;
            std::string num_a = a.substr(start_i, i - start_i);
            std::string num_b = b.substr(start_j, j - start_j);
            // Strip leading zeros and compare numbers
            num_a.erase(0, num_a.find_first_not_of('0'));
            num_b.erase(0, num_b.find_first_not_of('0'));

            if (num_a.size() != num_b.size())
            {
                return num_a.size() < num_b.size();
            }
            if (num_a != num_b)
            {
                return num_a < num_b;
            }
        }
        else if (a[i] != b[j])
        {
            return a[i] < b[j];
        }
        else
        {
            i++;
            j++;
        }
    }
    return a.size() < b.size();
}

void replaceAll(string &source, const string &from, const string &to)
{
    if (from.empty())
    {
        return;
    }
    size_t start_pos = 0;
    while ((start_pos = source.find(from, start_pos)) != string::npos)
    {
        source.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "Usage: " << argv[0] << " <directory_path> " << "<problems_directory_path> " << endl;
        return 1;
    }

    vector<string> test_probs;
    vector<string> test_probs_set;
    const filesystem::path dir_path{argv[1]};

    if (!filesystem::exists(dir_path) || !filesystem::is_directory(dir_path))
    {
        cout << "Provided path does not exist or is not a directory." << endl;
        return 1;
    }

    for (auto const &dir_entry : filesystem::directory_iterator{dir_path})
    {
        if (dir_entry.is_regular_file())
        {
            test_probs.push_back(dir_entry.path().filename().string());
        }
    }

    const filesystem::path parent{argv[2]};

    for (auto const &dir_entry : filesystem::directory_iterator{parent})
    {
        if (dir_entry.is_regular_file())
        {
            test_probs_set.push_back(dir_entry.path().filename().string());
        }
    }
    sort(test_probs_set.begin(), test_probs_set.end(), numericAwareCompare);

    sort(test_probs.begin(), test_probs.end(), numericAwareCompare);
    for (const auto &filename : test_probs)
    {
        cout << "Processing file: " << filename << endl;
        // Add the code here to process each file
    }

    fstream fout;
    string filename = "RandomInitalGuess.csv";
    fout.open(filename, ios::in);
    string temp;
    getline(fout, temp);
    std::unordered_map<std::string, std::vector<double>> initalGuessMap;
    for (const string &test_problems_file : test_probs_set)
    {
        string initalGuess;
        getline(fout, initalGuess);
        size_t pos = initalGuess.find(',');

        initalGuess = initalGuess.substr(pos + 1);
        // cout << initalGuess << '\n';
        initalGuess.erase(std::remove(initalGuess.begin(), initalGuess.end(), '('), initalGuess.end());
        initalGuess.erase(std::remove(initalGuess.begin(), initalGuess.end(), ')'), initalGuess.end());

        std::replace(initalGuess.begin(), initalGuess.end(), ';', ' ');

        stringstream ss(initalGuess);
        vector<double> numbers;
        double number;
        // cout << "Guess for " << test_problems_file << " : ";
        while (ss >> number)
        {
            numbers.push_back(number);
            // cout << number << " ";
        }
        // cout << "\n";
        initalGuessMap[test_problems_file] = numbers;
    }
    fout.close();

    // ofstream fout;
    // string filename = "result_GGN.csv";
    // fout.open(filename, std::ofstream::out | std::ofstream::trunc);
    // fout << "ProblemName" << "," << "Algorithm" << "," << "Alpha" << "," << "Iterations" << "," << "FuncValue" << "," << "Gradient" << "," << "TimeTaken(s)" << "," << "HessianModified" << "," << "Solved" << "," << "Max_Iterations" << "," << "Time_Limit" << "," << "Descent_Direction" << "\n";
    // fout << "Alpha" << "," << "Function Value" << "," << "Point" << "\n";
    // #pragma omp parallel for
    for (const string &test_problems_file : test_probs)
    {
        ofstream fout;
        fstream test_problems;
        // string test_problems_file = "on3test.txt";
        filesystem::path full_path = dir_path / test_problems_file;
        // open the test_problems_file file to perform read operation using file object.
        test_problems.open(full_path, ios::in);

        if (!test_problems.is_open())
        {
            cout << "Unable to open file: " << full_path << endl;
            continue; // Skip to the next file
        }
        string csv_name = test_problems_file.substr(0, test_problems_file.length() - 4);

        // Open new csv file to save data for this problem
        string filename = csv_name + ".csv";
        string directory = "btp_test_problems/cpp_solutions/Gaol_Results/GGN_A1_Modified/";

        if (!fs::exists(directory))
        {
            fs::create_directories(directory); // Create the directory if it does not exist
        }

        fout.open(directory + filename, std::ofstream::out | std::ofstream::trunc);
       //KK fout << "Time" << "," << "Function Value" << "," << "Grad Norm" << "," << "Hessian Evaluatiom" << "," << "Function Evaluations" << "," << "Gradient Evaluation" << "," << "Delta" << "\n";
        // fout << "Time" << "," << "Function Value" << "," << "Grad Norm" << "," << "Hessian Evaluatiom" << "," << "Function Evaluations" << "," << "Gradient Evaluation" << "\n";
        // fout << "ProblemName" << "," << "Algorithm" << "," << "Interval_Size" << "," << "Iterations" << ","
        //      << "FuncValue" << "," << "Gradient" << "," << "TimeTaken(s)" << "," << "Initial Guess" << "," << "HessianModified" << "," << "FunctionEval" << "," << "GradientEval" << "," << "Min_Eigen_Value" << ","
        //      << "Solved" << "," << "Cholesky_Result" << "," << "Max_Iterations" << "," << "Time_Limit" << "," << "Descent_Direction" << "," << "Small Step Size" << "\n";
        // Alpha option for calculating Interval Hessian
        double alp_options[1] = {0.1};
        // double alp_options[2] = {0.1, 0.5};
        string problem_name;
        while (getline(test_problems, problem_name))
        {
            try
            {
                // Extraxting the problem name
                problem_name = problem_name.substr(0, problem_name.length() - 1);

                // Open new csv file to save data for this problem
                // string filename = problem_name + ".csv";
                // fout.open(filename, std::ofstream::out | std::ofstream::trunc);

                // fout << "IterationGG" << "," << "IterationON2" << "," << "IterationDS" << "," << "IterationON3" << "," << "FuncValueGG" << "," << "FuncValueON2" << "," << "FuncValueDS" << "," << "FuncValueON3" << "," << "Alp" << "\n";

                string n_str, range_str;
                int n;
                double range_ll, range_ul;
                string func;

                // n = Variables in the problem
                getline(test_problems, n_str);
                n = stoi(n_str);
                // Input function of the problem
                getline(test_problems, func);

                // LowerBound(ll) and UpperBound(ul) for testing initial guess over this range
                getline(test_problems, range_str);
                range_ll = stod(range_str);
                getline(test_problems, range_str);
                range_ul = stod(range_str);

                std::string modified_function = replaceVariables(func, n);

                // Create symbolic variables
                GiNaC::parser reader;
                std::vector<GiNaC::realsymbol> vars;
                GiNaC::symtab table;
                for (int i = 1; i <= n; ++i)
                {
                    std::string var_name = "x" + std::to_string(i);
                    GiNaC::realsymbol var(var_name);
                    vars.push_back(var);
                    table[var_name] = var;
                }

                // Parse function into GiNaC expression
                reader.get_syms() = table;

                GiNaC::ex expr = reader(modified_function);
                int num_variables = n;
                std::vector<GiNaC::ex> gradient_expressions;
                for (int i = 0; i < num_variables; ++i)
                {
                    gradient_expressions.push_back(expr.diff(vars[i]));
                }
                // Evaluate function using GAOL
                Eigen::Matrix<GiNaC::ex, Eigen::Dynamic, Eigen::Dynamic> hessian_expressions(num_variables, num_variables);
                for (int i = 0; i < num_variables; ++i)
                {
                    for (int j = 0; j < num_variables; ++j)
                    {
                        hessian_expressions(i, j) = expr.diff(vars[i]).diff(vars[j]);
                    }
                }

                std::unordered_map<std::string, int> mp;
                for (int i = 0; i < n; i++) {
                    mp[vars[i].get_name()] = i;
                }

                std::vector<Node> all_nodes;
                std::vector<int> offsets, sizes;
                int F_hessian = n * n; // Flat size of the matrix

                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        std::vector<Node> nodes;
                        compile_ginac(hessian_expressions(i, j), nodes, mp);
                        
                        int base = all_nodes.size();
                        for (auto &nd : nodes) {
                            if (nd.left != -1) nd.left += base;
                            if (nd.right != -1) nd.right += base;
                        }
                        offsets.push_back(base);
                        sizes.push_back(nodes.size());
                        all_nodes.insert(all_nodes.end(), nodes.begin(), nodes.end());
                    }
                }

                Node *d_nodes;
                int *d_off, *d_sz;
                Interval *d_vars, *d_out;

                CUDA_CHECK(cudaMalloc(&d_nodes, all_nodes.size() * sizeof(Node)));
                CUDA_CHECK(cudaMalloc(&d_off, offsets.size() * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_sz, sizes.size() * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_vars, n * sizeof(Interval)));
                CUDA_CHECK(cudaMalloc(&d_out, F_hessian * sizeof(Interval)));

                CUDA_CHECK(cudaMemcpy(d_nodes, all_nodes.data(), all_nodes.size() * sizeof(Node), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_off, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_sz, sizes.data(), sizes.size() * sizeof(int), cudaMemcpyHostToDevice));
                // ---> INSERT END <---

                Eigen::MatrixXd hessian(n, n); // Hessian matrix using Eigen
                Eigen::MatrixXd hessian_mod(n, n);
                Eigen::MatrixXd inverseHessian(n, n);
                Eigen::MatrixXd hessian_cache(n, n);
                Eigen::VectorXd xk(n), xkn(n), inixk(n), xp(n); // Using Eigen::VectorXd for vectors
                Eigen::VectorXd fs_our(n), fs_ger(n);
                double diff_hess_norm = 0.0;
                int num_algorithms = 5; // only using Gershgorin Algo
                string algorithms[num_algorithms] = {"Steepest Descent", "Gerschgorin", "Diagonal Shift", "On2 Interval hessian", "On3 Interval Hessian"};
                // double func_value_ger, func_value_on2, func_value_ds, func_value_on3; // Function value of the four problems
                double func_value[num_algorithms];
                int algo_iterations[num_algorithms];
                // int total_test = 0; // Total tests(initial guesses) in which hessian was modified atleast one time.
                int func_eval = 0;
                int grad_eval = 0;

                const vector<double> &numbers = initalGuessMap[test_problems_file];
                // cout << n << " " << numbers.size() << "\n";
                // cout << "Processing: " << test_problems_file << "\n";
                int pmax = 1;
                // generating random initial guess 10 time to start with the minimum function value
                for (int j = 0; j < n; j++)
                {
                    inixk[j] = numbers[j];
                }
                for (int p = 0; p < pmax; p++)
                {
                    // inixk[0] = 2;
                    // inixk[1] = 2;
                    xk = inixk;
                    xkn = xk;

                    for (int i = 0; i < n; i++)
                    {
                        cout << inixk[i] << " ";
                    }
                    cout << endl;

                    Eigen::VectorXd grad(n), direcV(n); // Gradient and Direction Vector
                    double a_init = 1;
                    double a = a_init; // a: alpha to be modified with the wolfe conditions

                    // double delta_min = 1e-6;
                    double direction_sum = 0;
                    double direction_sq_sum = 0;
                    double delta = 0;

                    // c constant used for check strong wolf condition
                    double c = 0.001;

                    double a_min = 1e-20; // min value for a. Note: If a becomes less than a_min, then we have reached the solution

                    // rho value to update alpha value
                    double rh = 0.5;

                    int iter = 1;                 // iteration count
                    int iter_max = 10000;         // max iteration allowed.(If iter goes beyond iter_max, we might have diverted exceptionally from our solution)
                    int duration_max = 3600;      // max amount of time allowed for a problem
                    int hessian_modify_count = 0; // Number of times the hessian is modified.
                    // IntervalVector result;        // Value of the function at point xk
                    double results; // Value of the function at point xk, using GAOL intervals in a vector

                    for (int l = 0; l < num_algorithms; l++)
                    {
                        func_value[l] = 0;
                        algo_iterations[l] = 0;
                    }

                    for (int l = 0; l < num_algorithms; l++)
                    {
                        bool hessianValid = false;
                        bool algo_success = true;
                        double norm_grad = 0;
                        // double norm_pk = 0;
                        // double norm_xk = 0;
                        double norm_vec = 0;
                        double gfpk = 0;
                        bool small_step = false;
                        std::chrono::duration<double> time_taken;

                        if (l != 3)
                        {
                            continue;
                        }
                        for (int k = 0; k < sizeof(alp_options) / sizeof(alp_options[0]); k++)
                        {
                            delta = alp_options[k];
                            hessianValid = false;
                            norm_grad = 0;
                            // norm_pk = 0;
                            // norm_xk = 0;
                            norm_vec = 0;
                            small_step = 0;

                            algo_success = true; // Algorithms solves the problem successfull
                            auto start = chrono::high_resolution_clock::now();
                            if (l <= 2 && algo_iterations[l] != 0)
                            {
                                //  Iterations for Gerschgorin and DS are calculated
                                // Gerschgorin and Diagonal Shift is independent of alpha(interval hessian)
                                break;
                            }
                            xkn = inixk;
                            a = a_init;
                            iter = 1;                 // iteration count
                            hessian_modify_count = 0; // Number of times the hessian is modified.
                            double accumulated_gpu_time = 0.0;
                            while (true)
                            {
                                if (iter > iter_max)
                                {
                                    algo_success = false;
                                    break;
                                }
                                if (iter != 1)
                                {
                                    // check if we have reached the solution(norm of xk-xkn is less than tolerance)
                                    norm_vec = normOfVector(xkn - xk);
                                }
                                else
                                {
                                    norm_vec = 1;
                                }
                                // result = f.eval(xkn);
                                results = evaluate_function_at_point(expr, xkn, vars);
                                // cout << "Function Value :" << results << "\n";
                                // if (alp_options[k] == 0.5)
                                // {
                                //     cout << "Function Val :" << results << "\n";
                                // }
                                if (iter == 1)
                                {
                                    //KK fout << (duration_cast<std::chrono::duration<double>>(chrono::high_resolution_clock::now() - start)).count() << "," << results << "," << xk[0] << xk[1] << "\n";
                                    grad = evaluate_gradient_at_point(gradient_expressions, xkn, vars); // gradient vector at pt. xk
                                    grad_eval++;
                                    // grad = gradVector(df, xkn);
                                    norm_grad = normOfVector(grad);
                                    // direction_sum = 0;
                                    // for (int i = 0; i < n; i++)
                                    // {
                                    //     direction_sum += abs(grad[i]);
                                    // }
                                    // direction_sq_sum = norm_grad;
                                    // delta = UpdateDelta(delta, direction_sum, direction_sq_sum, n);
                                    // fout << (duration_cast<std::chrono::duration<double>>(chrono::high_resolution_clock::now() - start)).count() << "," << results << "," << "(" << xk[0] << " ; " << xk[1] << ")" << "\n";
                                    // fout << (duration_cast<std::chrono::duration<double>>(chrono::high_resolution_clock::now() - start)).count() << "," << results
                                    //      << "," << norm_grad << "," << hessian_modify_count << "," << func_eval << "," << grad_eval << "\n";
                                }
                                // if (alp_options[k] == 0.5)
                                // {
                                //     cout << "Gradient :" << "(" << grad[0] << ";" << grad[1] << ")" << "\n";
                                // }
                                if (norm_grad < 1e-3)
                                {
                                    break;
                                }
                                // cout << "Norm Vector :" << norm_vec << "\n";
                                if (norm_vec < 1e-6)
                                {
                                    small_step = true;
                                    // if (norm_grad > 1e-3)
                                    // {
                                    //     algo_success = false;
                                    // }
                                    // break;
                                    hessianValid = false;
                                }
                                // cout << "Hessian_vaid :" << hessianValid << "\n";
                                // cout << "Iteration :" << iter << "\n";
                                // cout << "Delta :" << delta << "\n";

                                auto time_check = chrono::high_resolution_clock::now();
                                time_taken = duration_cast<std::chrono::duration<double>>(time_check - start);
                                if (time_taken.count() > duration_max)
                                {
                                    algo_success = false;
                                    break;
                                }
                                // cout << "Iteration : " << iter << "\n";
                                iter++;

                                if (!((iter == 2) || (small_step)))
                                {
                                    hessianValid = CheckCacheValidity(range_ll, range_ul, delta, xp, xkn, grad);
                                }
                                small_step = false;
                                for (int i = 0; i < n; i++)
                                {
                                    xk[i] = xkn[i];
                                }
                                hessian = evaluate_hessian_at_point(hessian_expressions, xk, vars);
                                if (!hessianValid)
                                {
                                    // cout << "Hessian not valid" << "\n";
                                    if (iter != 2)
                                    {
                                        direction_sum = 0;
                                        for (int i = 0; i < n; i++)
                                        {
                                            direction_sum += abs(direcV[i]);
                                        }
                                        direction_sq_sum = normOfVector(direcV);
                                        // cout << "Direction Sum: " << direction_sum << " Direction Sq Sum: " << direction_sq_sum << "\n";
                                        delta = UpdateDelta(delta, direction_sum, direction_sq_sum, n);
                                        // cout << "Delta: " << delta << "\n";
                                    }
                                    // hessian matrix at pt. xk
                                    // cout << "Hessian: " << hessian << "\n";
                                    // hessian = HessianMatrix(dff, xk);
                                    // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                                    // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                                    hessian_modify_count += 1;
                                    xp = xk;
                                    // hessian_mod = ModifyHessian(hessian, l - 1, func, xp, delta, hessian_expressions, range_ll, range_ul, vars, grad); // IBEX
                                    hessian_mod = ModifyHessian(hessian, l - 1, func, xp, delta, hessian_expressions, range_ll, range_ul, vars, grad, d_nodes, d_off, d_sz, d_vars, d_out, accumulated_gpu_time);
                                    
                                    // cout << "Hessian: " << hessian << "\n";
                                    inverseHessian = InverseMatrix(hessian_mod);
                                    hessian_cache = inverseHessian;
                                }
                                else
                                {
                                    inverseHessian = hessian_cache;
                                }
                                diff_hess_norm += (hessian - hessian_mod).norm();
                                direcV = DirectionVector(inverseHessian, grad);
                                // if (alp_options[k] == 0.5)
                                // {
                                //     cout << "Direction Vector :" << "(" << direcV[0] << ";" << direcV[1] << ")" << "\n";
                                // }

                                // norm_pk = normOfVector(direcV, n);
                                gfpk = 0.0000;
                                for (int i = 0; i < n; i++)
                                {
                                    gfpk += direcV[i] * grad[i];
                                }
                                // cout << "GfPk: " << gfpk << "\n";
                                if ((gfpk > 0) || isnan(gfpk))
                                {
                                    cout << "Couldn't find descent direction after hessian modification" << "\n";
                                    algo_success = false;
                                    break;
                                }

                                for (int i = 0; i < n; i++)
                                {
                                    xkn[i] = xk[i] + a * direcV[i];
                                }
                                // if (alp_options[k] == 0.5)
                                // {
                                //     cout << "New Point:" << "(" << xkn[0] << ";" << xkn[1] << ")" << "\n";
                                // }
                                // IntervalVector func_value_xkn = f.eval(xkn);
                                double func_value_xkn = evaluate_function_at_point(expr, xkn, vars);
                                func_eval++;
                                // check wolfe condition and update xkn
                                while (func_value_xkn > results + c * a * gfpk)
                                {
                                    a = a * rh;
                                    if (a < a_min)
                                    {
                                        break;
                                    }

                                    for (int i = 0; i < n; i++)
                                    {
                                        xkn[i] = xk[i] + a * direcV[i];
                                        // xkn[i] = xkn[i]*domain_size + minx;
                                    }
                                    func_eval++;
                                    func_value_xkn = evaluate_function_at_point(expr, xkn, vars);
                                }

                                // cout << "Inversehessian :" << inverseHessian << " grad :" << grad << " direcV :" << direcV << " Function Value :" << results << " New Point:" << xk << " GfPk:" << gfpk << "Alpha: " << a << "\n";
                                // cout << "Iteration :" << iter << "\n";
                                // cout << "Gradient :" << "(" << grad[0] << ";" << grad[1] << ")" << "\n";
                                // cout << "Point :" << "(" << xkn[0] << ";" << xkn[1] << ")" << "\n";
                                // cout << "Function Value :" << func_value_xkn << "\n";
                                // cout << "Direction Vector :" << "(" << direcV[0] << ";" << direcV[1] << ")" << "\n";

                                grad = evaluate_gradient_at_point(gradient_expressions, xkn, vars); // gradient vector at pt. xk
                                grad_eval++;
                                // grad = gradVector(df, xkn); // gradient vector at pt. xk
                                norm_grad = normOfVector(grad);
                                if (a < a_min)
                                {
                                    if (norm_grad > 1e-3)
                                    {
                                        algo_success = false;
                                    }
                                    break;
                                }
                                a = a_init;
                                // fout << iter << "," << func_value_xkn.lb()[0] << "\n";
                                func_value[l] = func_value_xkn;
                                // fout << (duration_cast<std::chrono::duration<double>>(chrono::high_resolution_clock::now() - start)).count() << "," << func_value_xkn
                                //      << "," << norm_grad << "," << hessian_modify_count << "," << func_eval << "," << grad_eval << "\n";
                              //KK  fout << (duration_cast<std::chrono::duration<double>>(chrono::high_resolution_clock::now() - start)).count() << "," << results
                                //KK     << "," << norm_grad << "," << hessian_modify_count << "," << func_eval << "," << grad_eval << "," << delta << "\n";
                                // grad = gradVector(df, xkn);
                                // fout << (duration_cast<std::chrono::duration<double>>(chrono::high_resolution_clock::now() - start)).count() << "," << func_value_xkn << "," << xkn[0] << xkn[1] << "\n";
                            }
                            auto finish = chrono::high_resolution_clock::now();
                            if (l == 3) { // Only print for the On2 Interval Hessian algorithm
                            std::cout << "\n====================================\n";
                            std::cout << "  TOTAL GPU OFFLOAD TIME: " << accumulated_gpu_time << " ms\n";
                            std::cout << "====================================\n";
    }
                            // hessian = HessianMatrix(dff, xkn);
                            hessian = evaluate_hessian_at_point(hessian_expressions, xkn, vars);
                            inverseHessian = InverseMatrix(hessian);
                            direcV = DirectionVector(inverseHessian, grad);
                            gfpk = 0.0000;
                            for (int i = 0; i < n; i++)
                            {
                                gfpk += direcV[i] * grad[i];
                            }
                            if (!algo_success)
                            {
                                if (gfpk > 0)
                                {
                                    algo_success = false;
                                }
                                else
                                {
                                    algo_success = true;
                                }
                            }
                            Eigen::EigenSolver<Eigen::MatrixXd> es(n);
                            es.compute((hessian), /* computeEigenvectors = */ false);
                            double eigen_min = __DBL_MAX_10_EXP__;
                            for (int eig_idx = 0; eig_idx < n; eig_idx++)
                            {
                                if (es.eigenvalues().transpose().col(eig_idx).real().value() < eigen_min)
                                {
                                    eigen_min = es.eigenvalues().transpose().col(eig_idx).real().value();
                                }
                            }
                            cout << "Min EigenValue: " << eigen_min << "\n";
                            cout << "Gfpk: " << gfpk << "\n";
                            cout << "Cholesky: " << (checkCholesky(hessian) ? "Positive Definite" : "Indefinite") << "\n";

                            algo_iterations[l] = iter;
                            cout << "Algorithm : " << algorithms[l] << " and Interval_Size : " << round(alp_options[k] * 100) << "%" << endl;
                            std ::cout << "Iterations: " << algo_iterations[l] << " & Hessian Modified " << hessian_modify_count << " times." << endl;

                            for (int i = 0; i < n; i++)
                            {
                                std ::cout << xkn[i] << " ";
                            }
                            std ::cout << endl;

                            std::chrono::duration<double> elapsed_seconds = duration_cast<std::chrono::duration<double>>(finish - start);
                            fout << "ProblemName" << "," << "Algorithm" << "," << "Interval_Size" << "," << "Iterations" << ","
                                 << "FuncValue" << "," << "Gradient" << "," << "TimeTaken(s)" << "," << "Initial Guess" << "," << "HessianModified" << "," << "FunctionEval" << "," << "GradientEval" << "," << "Min_Eigen_Value" << ","
                                 << "Solved" << "," << "Cholesky_Result" << "," << "Max_Iterations" << "," << "Time_Limit" << "," << "Descent_Direction" << "," << "Small Step Size" << "Difference in Hessian" << "\n";

                            fout
                                << test_problems_file << "," << algorithms[l] << "," << alp_options[k] << ","
                                << algo_iterations[l] << "," << func_value[l] << "," << norm_grad << ","
                                << elapsed_seconds.count() << "," << inixk[0] << "," << hessian_modify_count << "," << func_eval + 1 << "," << iter << "," << eigen_min << ","
                                << (((norm_grad < 1e-3) && (eigen_min > -1e-3)) ? "Yes" : "No") << "," << (checkCholesky(hessian) ? "Positive Definite" : "Indefinite") << "," << ((iter > iter_max) ? "Yes" : "No") << ","
                                << ((time_taken.count() > duration_max) ? "Yes" : "No") << "," << (((gfpk > 0) || isnan(gfpk)) ? "Yes" : "No") << "," << (small_step ? "Yes" : "No") << "," << diff_hess_norm / iter << "\n";
                            fout.flush();
                            if (iter > iter_max)
                            {
                                cout << "Algorithm : " << algorithms[l] << " took more than max iterations\n";
                                continue;
                            }

                            if (!algo_success)
                            {
                                continue;
                            }
                        }
                        if (!algo_success)
                            continue;
                    }
                }
                cudaFree(d_nodes);
                cudaFree(d_off);
                cudaFree(d_sz);
                cudaFree(d_vars);
                cudaFree(d_out);

                string dummy;
                getline(test_problems, dummy);
            }
            catch (const std::exception &e)
            {
                fout << test_problems_file << "," << "Exception" << ",0,0,0,0,0,0,0,0,0,0,No,No,No,No,Yes,No\n";
                // fout << test_problems_file << "," << "Error in Function(stoi)" << ",0,0,0,0,0,0,No,No,Yes,No\n";
                // std::cerr << "Error: " << e.what() << std::endl;
                // return 1;
                break;
            }
        }
        test_problems.close();
        fout.close();
    }
    return 0;
}