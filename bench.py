import json
import subprocess
import time
import importlib
import numpy as np
import statistics
import os

def load_experiments(json_path):
    """
    Loads experiment configurations from a JSON file.

    Args:
        json_path (str): The path to the JSON configuration file.

    Returns:
        list: A list of experiment dictionaries, or an empty list if an error occurs.
    """
    try:
        with open(json_path, 'r') as f:
            experiments_config = json.load(f)
        if "experiments" not in experiments_config:
            print(f"Warning: 'experiments' key not found in {json_path}. Please ensure the JSON is correctly structured.")
            return []
        return experiments_config["experiments"]
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_path}. Details: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading {json_path}: {e}")
        return []

def generate_matrices(M, K, N):
    """
    Generates two random matrices A (MxK) and B (KxN) for GEMM.
    These matrices are used as input for the GEMM functions to ensure
    consistent input for each benchmark run.

    Args:
        M (int): Number of rows in matrix A and result matrix C.
        K (int): Number of columns in matrix A and rows in matrix B.
        N (int): Number of columns in matrix B and result matrix C.

    Returns:
        tuple: A tuple containing two numpy arrays (matrix A, matrix B).
    """
    A = np.random.rand(M, K)
    B = np.random.rand(K, N)
    return A, B

def run_python_benchmark(module_name, func_name, M, N, K, num_runs):
    """
    Runs a specified Python GEMM function multiple times and measures its performance.

    Args:
        module_name (str): The name of the Python module containing the function.
        func_name (str): The name of the GEMM function to call.
        M (int): Matrix M dimension.
        N (int): Matrix N dimension.
        K (int): Matrix K dimension.
        num_runs (int): The number of times to execute the function for averaging.

    Returns:
        list: A list of run times in seconds for each successful execution.
    """
    run_times = []
    try:
        # Dynamically import the specified module
        module = importlib.import_module(module_name)
        # Get the GEMM function object from the module
        gemm_func = getattr(module, func_name)
    except (ImportError, AttributeError) as e:
        print(f"    Error loading Python function '{module_name}.{func_name}': {e}")
        return []

    for i in range(num_runs):
        A, B = generate_matrices(M, K, N) # Generate new matrices for each run to avoid caching effects
        start_time = time.perf_counter()
        try:
            # Execute the GEMM function
            gemm_func(A, B)
        except Exception as e:
            print(f"    Error during Python function execution '{func_name}' (Run {i+1}): {e}")
            continue # Continue to next run even if one fails
        end_time = time.perf_counter()
        run_times.append(end_time - start_time)
        print(f"      Run {i+1}: {run_times[-1]:.6f} s")
    return run_times

def run_external_benchmark(command_template, M, N, K, num_runs):
    """
    Runs an external command (e.g., a C, Rust, or compiled executable) multiple times
    and measures its wall-clock performance.

    Args:
        command_template (str): A string template for the command to execute.
                                It should contain placeholders like {M}, {N}, {K}.
        M (int): Matrix M dimension.
        N (int): Matrix N dimension.
        K (int): Matrix K dimension.
        num_runs (int): The number of times to execute the command for averaging.

    Returns:
        list: A list of run times in seconds for each successful command execution.
    """
    run_times = []
    # Check if the command executable exists and is executable
    # The first part of the command template is assumed to be the executable path
    executable_path = command_template.split()[0]
    if not os.path.exists(executable_path):
        print(f"    Error: Executable '{executable_path}' not found. Please ensure it exists and is in the correct directory.")
        return []
    if not os.access(executable_path, os.X_OK):
        print(f"    Error: Executable '{executable_path}' is not executable. Please run 'chmod +x {executable_path}'.")
        return []

    for i in range(num_runs):
        # Format the command with current matrix dimensions
        command = command_template.format(M=M, N=N, K=K)
        print(f"    Executing command (Run {i+1}): {command}")
        start_time = time.perf_counter()
        try:
            # subprocess.run executes the command.
            # shell=True is used for convenience with command strings, but for security,
            # it's better to pass command as a list of arguments and set shell=False for production.
            # capture_output=True captures stdout and stderr.
            # text=True decodes stdout/stderr as text.
            # check=True raises a CalledProcessError if the command returns a non-zero exit code.
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout:
                # print(f"      Command stdout:\n{result.stdout.strip()}")
                pass # Suppress verbose stdout from external commands unless explicitly needed
            if result.stderr:
                print(f"      Command stderr:\n{result.stderr.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"    Error executing command (Run {i+1}): Command returned non-zero exit code {e.returncode}")
            print(f"      Stdout: {e.stdout.strip()}")
            print(f"      Stderr: {e.stderr.strip()}")
            continue
        except FileNotFoundError:
            print(f"    Error: Command '{command.split()[0]}' not found. This should have been caught earlier.")
            continue
        except Exception as e:
            print(f"    An unexpected error occurred during external command execution (Run {i+1}): {e}")
            continue
        end_time = time.perf_counter()
        run_times.append(end_time - start_time)
        print(f"      Run {i+1}: {run_times[-1]:.6f} s")
    return run_times

def main(json_file_path="experiments.json"):
    """
    Main function to orchestrate the GEMM benchmarking process.
    It loads configurations, runs benchmarks, and prints a summary of results.
    """
    experiments = load_experiments(json_file_path)
    if not experiments:
        print("No experiments loaded or found in the configuration. Exiting.")
        return

    # List to store results of all experiments for final summary
    all_results = []

    print("--- Starting GEMM Benchmarking ---\n")

    for exp in experiments:
        exp_name = exp.get("name", "Unnamed Experiment")
        exp_type = exp.get("type")
        matrix_sizes = exp.get("matrix_sizes", [])
        num_runs = exp.get("num_runs", 1)

        if not exp_type or not matrix_sizes:
            print(f"Skipping experiment '{exp_name}': Missing 'type' or 'matrix_sizes' configuration.\n")
            continue

        print(f"--- Running Experiment: {exp_name} (Type: {exp_type}) ---")

        for M, N, K in matrix_sizes:
            if not all(isinstance(dim, int) and dim > 0 for dim in [M, N, K]):
                print(f"    Skipping matrix size [{M},{N},{K}]: Invalid dimensions. Must be positive integers.")
                continue

            print(f"  Benchmarking M={M}, N={N}, K={K} (Number of runs: {num_runs})")

            current_run_times = []
            if exp_type == "python_function":
                module_name = exp.get("module")
                func_name = exp.get("function")
                if not module_name or not func_name:
                    print(f"    Skipping: 'module' or 'function' not specified for Python experiment '{exp_name}'.")
                    continue
                current_run_times = run_python_benchmark(module_name, func_name, M, N, K, num_runs)
            elif exp_type == "external_command":
                command_template = exp.get("command_template")
                if not command_template:
                    print(f"    Skipping: 'command_template' not specified for external command experiment '{exp_name}'.")
                    continue
                current_run_times = run_external_benchmark(command_template, M, N, K, num_runs)
            else:
                print(f"    Skipping: Unknown experiment type '{exp_type}' for '{exp_name}'.")
                continue

            if current_run_times:
                avg_time = statistics.mean(current_run_times)
                # Standard deviation requires at least 2 data points
                std_dev = statistics.stdev(current_run_times) if len(current_run_times) > 1 else 0.0
                all_results.append({
                    "name": exp_name,
                    "type": exp_type,
                    "M": M,
                    "N": N,
                    "K": K,
                    "avg_time_s": avg_time,
                    "std_dev_s": std_dev,
                    "num_runs": len(current_run_times) # Actual number of successful runs
                })
                print(f"  Summary for M={M}, N={N}, K={K}:")
                print(f"    Average time: {avg_time:.6f} s")
                print(f"    Standard Deviation: {std_dev:.6f} s\n")
            else:
                print(f"  No successful runs recorded for M={M}, N={N}, K={K} in '{exp_name}'.\n")

    print("\n--- Benchmarking Complete ---")
    print("\n--- Detailed Results Summary ---")
    if not all_results:
        print("No results were generated. Please check your JSON configuration and ensure executables are present and runnable.")
        return

    # Print results in a formatted table
    print("{:<35} {:<8} {:<8} {:<8} {:<10} {:<15} {:<15}".format(
        "Experiment Name", "M", "N", "K", "Runs", "Avg Time (s)", "Std Dev (s)"
    ))
    print("-" * 110)
    for res in all_results:
        print("{:<35} {:<8} {:<8} {:<8} {:<10} {:<15.6f} {:<15.6f}".format(
            res["name"], res["M"], res["N"], res["K"], res["num_runs"],
            res["avg_time_s"], res["std_dev_s"]
        ))

    # Optional: Save results to CSV for easier analysis
    try:
        import csv
        csv_file_name = "gemm_benchmark_results.csv"
        with open(csv_file_name, 'w', newline='') as csvfile:
            fieldnames = ["name", "type", "M", "N", "K", "num_runs", "avg_time_s", "std_dev_s"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults also saved to {csv_file_name}")
    except Exception as e:
        print(f"\nWarning: Could not save results to CSV due to an error: {e}")

if __name__ == "__main__":
    # You can pass the JSON file path as an argument if needed, e.g.,
    # python benchmark_gemm.py my_custom_experiments.json
    # For now, it defaults to 'experiments.json'
    main("experiments.json") # Ensure this file is in the same directory as the script
