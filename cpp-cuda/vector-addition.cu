#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Thread %d (block %d, thread %d) processing element %d\n",
           i, blockIdx.x, threadIdx.x, i);

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 6;  // Small example
    size_t size = N * sizeof(float);

    // CPU arrays
    float h_A[] = {1, 2, 3, 4, 5, 6};
    float h_B[] = {10, 20, 30, 40, 50, 60};
    float h_C[6];

    // GPU arrays
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);    // Allocates GPU memory
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy to GPU // Transfer through PCIe bus
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);    // CPU→GPU
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);    // CPU→GPU

    // Launch kernel - 2 blocks, 4 threads/block (8 threads total)
    // Kernel runs on GPU, using GPU memory
    int threadsPerBlock = 4;
    int blocksPerGrid = 2;
    printf("Launching kernel with %d blocks, %d threads/block\n",
           blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();  // Wait for GPU

    // Copy back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("\nResults:\n");
    for (int i = 0; i < N; i++) {
        printf("C[%d] = %.1f + %.1f = %.1f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}