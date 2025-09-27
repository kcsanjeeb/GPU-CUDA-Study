#include <iostream>
#include <cuda_runtime.h>

// CORRECT adjacent-pair reduction
__global__ void reduceSumCorrect(float *input, float *output, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load data
    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    // Adjacent-pair reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            if (index + stride < blockDim.x) {
                sdata[index] += sdata[index + stride];
            }
        }
        __syncthreads();
    }

    // Output
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 16;
    float h_input[] = {8, 3, 5, 7, 2, 9, 1, 6, 4, 10, 12, 15, 11, 14, 13, 16};
    float h_output[1];

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy to GPU
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 16;
    int sharedMem = threads * sizeof(float);

    reduceSumCorrect<<<1, threads, sharedMem>>>(d_input, d_output, N);

    // Get result
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Array: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << "\nCorrect Sum: " << h_output[0] << std::endl;
    std::cout << "Expected: 136" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}