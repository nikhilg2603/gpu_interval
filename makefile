CXX = g++
NVCC = nvcc

# External package configurations
CXXFLAGS_IBEX := $(shell pkg-config --cflags ibex) 
LIBS_IBEX     := $(shell pkg-config --libs  ibex)

# C++ Flags (Used ONLY for your main cpp file to protect Eigen)
CXXFLAGS = -std=c++17 -fopenmp -U__STRICT_ANSI__ -O3 -DNDEBUG $(CXXFLAGS_IBEX)

# NVCC Flags: Embed native SASS for Pascal(60), Turing(75), and Ampere(80) GPUs.
# This completely bypasses the driver's PTX JIT compiler and prevents the toolchain error!
NVCCFLAGS = -std=c++17 -O3 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -Xcompiler -fopenmp

# Libraries to link
LIBS = $(LIBS_IBEX) -lginac -lcln -lgaol -lm -lgomp

TARGET = gaol_gpu

all: $(TARGET)

# 1. Compile pure CUDA code into an object file
gpu_kernels.o: gpu_kernels.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 2. Compile C++ code with standard g++ (Using INTHOP-A1-GGN.cpp)
gaol_gpu.o: gaol_gpu.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 3. LINK using NVCC
$(TARGET): gaol_gpu.o gpu_kernels.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f *.o $(TARGET)