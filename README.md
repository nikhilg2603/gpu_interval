# GPU-Accelerated GAOL Interval Optimization Solver

This project implements a hybrid CPU/GPU numerical optimization solver. It offloads highly parallelizable tasks—specifically the evaluation of dense $n \times n$ Interval Hessian matrices—to NVIDIA GPUs using CUDA, while maintaining complex symbolic differentiation and interval arithmetic on the host CPU using GiNaC, GAOL, and Eigen.

By physically separating the compilation domains (`g++` for host C++ libraries, `nvcc` for device kernels), this setup bypasses standard vector/CUDA integration conflicts and embeds native SASS binaries to prevent runtime PTX JIT driver issues.

## Prerequisites & Dependencies

Ensure the following libraries and toolkits are installed on your system:

* **C++ Compiler:** `g++` supporting C++17 and OpenMP (`-fopenmp`).
* **CUDA Toolkit:** `nvcc` compiler (targeting Pascal `sm_60`, Turing `sm_75`, or Ampere `sm_80` architectures).
* **Symbolic & Interval Libraries:**
  * [GAOL](https://sourceforge.net/projects/gaol/) (Interval Arithmetic)
  * [GiNaC](https://www.ginac.de/) & CLN (Symbolic Math & Differentiation)
  * [IBEX](http://www.ibex-lib.org/) (Used for package configurations)
* **Linear Algebra:** [Eigen3](https://eigen.tuxfamily.org/) (Headers must be accessible in the include path).

## Project Structure

* `gaol_gpu.cpp`: The main host execution file. Handles file parsing, GiNaC symbolic expression generation, GAOL environment setup, optimization loops, CPU timing, and Eigen matrix operations.
* `gpu_kernels.cu`: The pure CUDA implementation. Contains standard interval math routines optimized for device execution (`__device__` add, mul, sin, cos, pow, etc.) and the global kernel to map multi-variable interval points across the AST nodes.
* `gpu_interface.h`: The header file providing the handshake between host code and device kernels without mixing compiler domains.
* `Makefile`: Custom build configuration managing independent object compilation and unified NVCC linking.

---

## Build Instructions

To compile the project, open your terminal in the project directory and use the included `Makefile`.

### Clean Old Builds
Always clean the directory before a fresh build to ensure object files (`.o`) are correctly generated:
```bash
make clean
```

> **Note:** Running `make gaol_gpu` also works perfectly.

This generates the final executable named `gaol_gpu`.

---

# Running the Solver

The executable requires two directory paths passed as command-line arguments to locate the profile sets and test problems.

## Command Syntax

```bash
./gaol_gpu <path_to_data_profile_set> <path_to_all_probs>
```

## Example Usage

```bash
./gaol_gpu ./data_profile_set ./all_probs
```

### Arguments

- `./data_profile_set`  
  Directory containing the initial guess configurations or specific profile definitions.

- `./all_probs`  
  Directory containing the raw benchmark problem files (e.g., `.txt` files specifying variable counts, function strings, and bounding domains).

---

# Output & Performance Metrics

During execution, the solver outputs real-time console metrics tracking:

- Initial starting vectors.
- Matrix Definiteness checks (Cholesky results).

## Execution Timers

Separated timing outputs are provided to compare CPU vs. GPU bottlenecks:

- `[GPU Kernel] Math Execution Time`  
  Pure execution speed on the device.

- `[Total Offload] Memcpy + Kernel Time`  
  Full round-trip PCIe memory transfer + compute overhead.

- `[CPU] GAOL Hessian Evaluation Time`  
  Native host evaluation speed for comparison.

---

# Result Logging

Results are fully logged to dynamically generated `.csv` files inside:

```text
btp_test_problems/cpp_solutions/Gaol_Results/GGN_A1_Modified/
```