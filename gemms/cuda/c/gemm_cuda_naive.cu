// gemm_cuda_naive.cu
#include <stdio.h>
#include <stdlib.h> // For EXIT_FAILURE
#include <cuda_runtime.h> // For CUDA API calls

// Macro to check for CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA Kernel for naive matrix multiplication (C = A * B)
// A is M x K, B is K x N, C is M x N
__global__ void matrixMul(const float* A, const float* B, float* C, int M, int K, int N) {
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // Define matrix dimensions
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A, *h_B, *h_C_gpu, *h_C_ref;

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_gpu = (float*)malloc(size_C);
    h_C_ref = (float*)malloc(size_C); // For reference CPU computation

    if (!h_A || !h_B || !h_C_gpu || !h_C_ref) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = rand() / (float)RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    // A common block size is 16x16 or 32x32 for 2D kernels
    // Using a 16x16 thread block
    int THREADS_PER_BLOCK = 16;
    dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    printf("Executing CUDA kernel with Grid %dx%d, Block %dx%d\n",
           dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

    // Launch the kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("CUDA Matrix Multiplication took %.3f ms\n", milliseconds);

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));

    // --- CPU Reference Calculation for Verification ---
    printf("\nRunning CPU reference calculation for verification...\n");
    double cpu_start_time = clock();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
    double cpu_end_time = clock();
    double cpu_time_taken = ((double)(cpu_end_time - cpu_start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Reference took %.3f ms\n", cpu_time_taken);

    // --- Verification ---
    printf("\nVerifying results...\n");
    bool correct = true;
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(h_C_ref[i] - h_C_gpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        // Using a simple absolute difference check for float comparison
        if (diff > 1e-3) { // A higher tolerance might be needed depending on precision
            correct = false;
            // printf("Mismatch at index %d: CPU=%.6f, GPU=%.6f, Diff=%.6f\n", i, h_C_ref[i], h_C_gpu[i], diff);
            // break; // uncomment to find first mismatch
        }
    }

    if (correct) {
        printf("Verification successful! Results match (max diff: %.6f).\n", max_diff);
    } else {
        printf("Verification FAILED! Max difference: %.6f\n", max_diff);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_ref);

    return 0;
}