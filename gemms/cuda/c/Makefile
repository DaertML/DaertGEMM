# Makefile for CUDA GEMM Example

# --- Configuration ---
# CUDA compiler (nvcc)
NVCC        = nvcc
# Source file
SRC         = gemm_cuda_naive.cu
# Executable name
TARGET      = gemm_cuda_naive
# CUDA compute capability (e.g., sm_75 for Turing, sm_86 for Ampere)
# Adjust this based on your GPU's architecture.
# You can find your GPU's compute capability here:
# https://developer.nvidia.com/cuda-gpus
# If unsure, you can often omit this or use a lower common one like sm_35
# For modern GPUs, sm_70 or higher is a good starting point.
ARCH        = sm_75
# Common CUDA flags
# -arch=$(ARCH): Specifies the target GPU architecture.
# -O3: Optimization level.
# --ptxas-options=-v: Verbose PTXAS (assembler) output, useful for seeing resource usage.
# -lcudart: Link to the CUDA Runtime library (usually implied by nvcc but good to be explicit).
# -lcublas: Link to the cuBLAS library (for more advanced GEMM, not strictly needed for naive).
NVCC_FLAGS  = -arch=$(ARCH) -O3 --ptxas-options=-v -lcudart -lcublas

# --- Targets ---

# Default target: builds and then runs the executable
all: build run

# Build target: compiles the CUDA source file
build: $(SRC)
	@echo "Compiling $(SRC) to $(TARGET)..."
	$(NVCC) $(SRC) -o $(TARGET) $(NVCC_FLAGS)
	@echo "Compilation complete."

# Run target: executes the compiled program
run: $(TARGET)
	@echo "Running $(TARGET)..."
	./$(TARGET)

# Clean target: removes compiled files
clean:
	@echo "Cleaning up..."
	rm -f $(TARGET)

.PHONY: all build run clean

