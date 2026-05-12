#ifndef GPU_INTERFACE_H
#define GPU_INTERFACE_H
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
void launch_gpu_kernel(Node* d_nodes, int* d_off, int* d_sz, Interval* d_vars, Interval* d_out, int F, int blocks, int threads);

#endif
