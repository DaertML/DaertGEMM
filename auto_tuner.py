import subprocess
import time
from ollama import chat
from ollama import ChatResponse
from agents.optimization_agent import OptimizationAgent
from tools.hardware_description import HardwareDescription
from tools.performance_pruning import PerformancePruning

def load_hardware_description(architecture):
    if architecture == "3090":
        with open("prompts/hwdesc/3090.txt", "r") as file:
            return file.read()
    elif architecture == "i7":
        with open("prompts/hwdesc/i7.txt", "r") as file:
            return file.read()
    else:
        raise ValueError("Unsupported architecture")

def load_hardware_hint(architecture):
    if architecture == "3090":
        with open("prompts/hwhint/3090.txt", "r") as file:
            return file.read()
    elif architecture == "i7":
        with open("prompts/hwhint/i7.txt", "r") as file:
            return file.read()
    else:
        raise ValueError("Unsupported architecture")

def load_optimization_methods(architecture):
    methods = {}
    for method in ["layout", "prefetch", "reordering", "tiling", "vectorization"]:
        with open(f"prompts/optim/{architecture}/{method}.txt", "r") as file:
            methods[method] = file.read()
    return methods

def benchmark_code(code, M=1024, N=1024, K=1024, num_runs=5):
    """
    Benchmarks the given GEMM code by running it multiple times and measuring its performance.

    Args:
        code (str): The GEMM code to benchmark.
        M (int): Matrix M dimension.
        N (int): Matrix N dimension.
        K (int): Matrix K dimension.
        num_runs (int): Number of runs for averaging the performance.

    Returns:
        float: Average execution time in seconds, or None if an error occurs.
    """
    # Write the code to a temporary Python file
    temp_file = "temp_gemm_code.py"
    with open(temp_file, 'w') as f:
        f.write(code)

    try:
        # Run the benchmark using run_python_benchmark from bench.py
        import bench
        module_name = temp_file[:-3]  # Remove .py extension for import
        func_name = "gemm_func"  # Assuming the generated code defines a function named gemm_func
        run_times = bench.run_python_benchmark(module_name, func_name, M, N, K, num_runs)
        if not run_times:
            print("No successful runs recorded during benchmarking.")
            return None
        avg_time = sum(run_times) / len(run_times)
        return avg_time
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        return None
    finally:
        # Clean up the temporary file
        import os
        os.remove(temp_file)

import langgraph

def auto_tuning(C, pA, k, T, architecture, model_name='llama3.2'):
    C = open(C, "r").read()

    optimization_agent = OptimizationAgent(model_name)
    hardware_description = load_hardware_description(architecture)
    hardware_hint = load_hardware_hint(architecture)
    optimization_methods = load_optimization_methods(architecture)

    for t in range(T):
        print(f"Iteration {t + 1}/{T}")
        Cnew = []
        for code in C:
            for mp in optimization_agent.suggest_primitives(code, hardware_description, hardware_hint, optimization_methods):
                c_t = optimization_agent.generate_code(code, mp)
                performance = evaluate_performance(c_t)  # Placeholder for actual performance evaluation
                print(f"Generated Code:\n{c_t}\nPerformance: {performance}")
                Cnew.append((c_t, performance))
        C = PerformancePruning.prune(Cnew, k)

    return max(C, key=lambda x: x[1])[0]

def evaluate_performance(code):
    """
    Evaluates the performance of the given GEMM code by benchmarking it and checking its validity.

    Args:
        code (str): The GEMM code to evaluate.

    Returns:
        float: Average execution time in seconds, or None if the kernel is invalid.
    """
    # Check for basic validity of the kernel
    if not isinstance(code, str) or "def gemm_func" not in code:
        print("Invalid GEMM code provided.")
        return None

    # Benchmark the code
    avg_time = benchmark_code(code)
    print(f"Benchmarked Code Performance: {avg_time}")
    return avg_time

if __name__ == "__main__":
    pA = "meta-prompts"
    k = 5
    T = 10
    C = "gemms/cuda/c/gemm_cuda_naive.cu"
    architecture = "3090"  # or "i7"
    best_code = auto_tuning(C, pA, k, T, architecture)
    print("Best GEMM code:", best_code)
