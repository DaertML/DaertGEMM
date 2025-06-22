import numpy as np
import time

# --- Python GEMM Implementations ---

def gemm_naive(A, B):
    """
    Naive General Matrix Multiplication (GEMM) implementation using triple nested loops.
    C = A @ B
    A is an M x K matrix
    B is a K x N matrix
    C will be an M x N matrix
    """
    M, K = A.shape
    K_b, N = B.shape

    if K != K_b:
        raise ValueError(f"Matrix A columns ({K}) must match Matrix B rows ({K_b})")

    C = np.zeros((M, N), dtype=A.dtype)

    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
    return C

def gemm_numpy(A, B):
    """
    GEMM implementation using NumPy's highly optimized dot product.
    This uses underlying optimized BLAS libraries (e.g., OpenBLAS, MKL).
    """
    return A @ B # or np.dot(A, B)

# --- Benchmarking and Validation ---

def verify_gemm(C_ref, C_test, rtol=1e-5, atol=1e-8):
    """
    Verifies if two matrices are approximately equal.
    Args:
        C_ref: The reference matrix (e.g., from NumPy's implementation).
        C_test: The matrix to test.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    Returns:
        True if matrices are approximately equal, False otherwise.
    """
    if C_ref.shape != C_test.shape:
        print(f"Shapes do not match: Reference {C_ref.shape}, Test {C_test.shape}")
        return False
    return np.allclose(C_ref, C_test, rtol=rtol, atol=atol)

def benchmark_gemm(gemm_func, A, B, name):
    """
    Benchmarks a GEMM implementation.
    Args:
        gemm_func: The GEMM function to benchmark.
        A: First input matrix.
        B: Second input matrix.
        name: Name of the implementation for printing.
    Returns:
        The computed result matrix and the execution time.
    """
    start_time = time.perf_counter()
    C = gemm_func(A, B)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"{name} took {elapsed_time:.6f} seconds.")
    return C, elapsed_time

if __name__ == "__main__":
    # Define matrix dimensions
    M = 100
    K = 150
    N = 120

    print(f"Benchmarking GEMM for matrices of size A({M}x{K}) and B({K}x{N})\n")

    # Generate random input matrices
    # Using float32 for consistency with potential CUDA examples
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)

    # --- Run NumPy benchmark (as reference) ---
    C_numpy, time_numpy = benchmark_gemm(gemm_numpy, A, B, "NumPy GEMM")

    # --- Run Naive GEMM benchmark ---
    print("\nRunning Naive GEMM (may take a while for larger matrices)...")
    C_naive, time_naive = benchmark_gemm(gemm_naive, A, B, "Naive GEMM")

    # --- Validate Results ---
    print("\n--- Validation ---")
    if verify_gemm(C_numpy, C_naive):
        print("Naive GEMM result matches NumPy GEMM result (validation successful!).")
    else:
        print("Naive GEMM result DOES NOT match NumPy GEMM result (validation FAILED!).")

    # --- Performance Comparison ---
    print("\n--- Performance Summary ---")
    if time_naive > 0: # Avoid division by zero if time is negligible
        speedup_numpy_over_naive = time_naive / time_numpy
        print(f"NumPy GEMM is {speedup_numpy_over_naive:.2f}x faster than Naive GEMM.")
    else:
        print("Naive GEMM time was negligible, cannot calculate speedup.")

    # You can try larger matrices to see the performance difference more clearly,
    # but be aware that naive GEMM will become very slow.
    # For example:
    # M_large, K_large, N_large = 500, 500, 500
    # A_large = np.random.rand(M_large, K_large).astype(np.float32)
    # B_large = np.random.rand(K_large, N_large).astype(np.float32)
    #
    # print(f"\nBenchmarking larger matrices A({M_large}x{K_large}) and B({K_large}x{N_large})")
    # C_numpy_large, time_numpy_large = benchmark_gemm(gemm_numpy, A_large, B_large, "NumPy GEMM (Large)")
    # C_naive_large, time_naive_large = benchmark_gemm(gemm_naive, A_large, B_large, "Naive GEMM (Large)")
    # if time_naive_large > 0:
    #     speedup_numpy_over_naive_large = time_naive_large / time_numpy_large
    #     print(f"NumPy GEMM (Large) is {speedup_numpy_over_naive_large:.2f}x faster than Naive GEMM (Large).")
