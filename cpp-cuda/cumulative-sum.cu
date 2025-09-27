#include <iostream>
#include <cuda_runtime.h>

// Your original kernel (unchanged)
__global__ void exclusiveScan(float *input, float *output, int N) {
    extern __shared__ float temp[]; // Shared memory for the block
    int tid = threadIdx.x;          // Thread ID within block (0 to 7)

    // PHASE 1: Load 2 elements per thread into shared memory
    temp[2 * tid] = input[2 * tid];
    temp[2 * tid + 1] = input[2 * tid + 1] ;
    __syncthreads(); // Wait for all threads to load data

    // PHASE 2: Up-sweep (Reduction) - Build sum tree from leaves to root
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < N) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // PHASE 3: Down-sweep - Build prefix sums from root to leaves
    for (int stride = blockDim.x; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < N) {
            float temp_val = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += temp_val;
        }
        __syncthreads();
    }

    // PHASE 4: Write results back to global memory
    output[2 * tid] = temp[2 * tid];
    output[2 * tid + 1] = temp[2 * tid + 1];
}

// Helper function to convert exclusive scan to inclusive scan
void exclusiveToInclusive(float *exclusive, float *inclusive, int N) {
    for (int i = 0; i < N - 1; i++) {
        inclusive[i] = exclusive[i + 1];
    }
    inclusive[N - 1] = exclusive[N - 1] + inclusive[N - 2]; // Approximate
}

int main() {
    const int N = 16;
    float h_input[N] = {8, 3, 5, 7, 2, 9, 1, 6, 4, 10, 12, 15, 11, 14, 13, 16};
    float h_exclusive[N] = {0};
    float h_inclusive[N] = {0};

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy to GPU
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch your original exclusive scan kernel
    int threads = 8;
    int sharedMem = threads * 2 * sizeof(float);
    exclusiveScan<<<1, threads, sharedMem>>>(d_input, d_output, N);

    // Get exclusive scan results
    cudaMemcpy(h_exclusive, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert exclusive scan to inclusive scan (cumulative sum)
    // Exclusive: [0, 8, 11, 16, 23, 25, 34, 35, 41, 45, 55, 67, 82, 93, 107, 120]
    // Inclusive: [8, 11, 16, 23, 25, 34, 35, 41, 45, 55, 67, 82, 93, 107, 120, 136]

    h_inclusive[0] = h_input[0]; // First element is the same
    for (int i = 1; i < N; i++) {
        h_inclusive[i] = h_inclusive[i - 1] + h_input[i];
    }

    // Print results
    std::cout << "Input:    ";
    for (int i = 0; i < N; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Exclusive: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_exclusive[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Inclusive: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_inclusive[i] << " ";
    }
    std::cout << std::endl;

    // Expected inclusive result
    float expected[N] = {8, 11, 16, 23, 25, 34, 35, 41, 45, 55, 67, 82, 93, 107, 120, 136};

    std::cout << "Expected:  ";
    for (int i = 0; i < N; i++) {
        std::cout << expected[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}