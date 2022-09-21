#include <iostream>
#include <limits>
#include <cstdio>

inline void CUDA_CHECK (cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

#define NUM_POINTS 4194304
#define TOLERANCE  0.0001
#define MAX_ITERS  1000

__global__ void jacobi (const float* f_old, float* f, float* l2_norm, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > 0 && idx < N - 1) {
        f[idx] = 0.5f * (f_old[idx+1] + f_old[idx-1]);

        float l2 = (f[idx] - f_old[idx]) * (f[idx] - f_old[idx]);

        atomicAdd(l2_norm, l2);
    }
}

__global__ void initialize (float* f, float T_left, float T_right, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx == 0) {
        f[idx] = T_left;
    }
    else if (idx == N - 1) {
        f[idx] = T_right;
    }
    else if (idx < N - 1) {
        f[idx] = 0.0f;
    }
}

int main() {
    // Allocate space for the grid data, and the temporary buffer
    // for the "old" data.
    float* f_old;
    float* f;

    CUDA_CHECK(cudaMalloc(&f_old, NUM_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&f, NUM_POINTS * sizeof(float)));

    // Allocate memory for the L2 norm, on both the host and device.
    float* l2_norm = (float*) malloc(sizeof(float));
    float* d_l2_norm;
    CUDA_CHECK(cudaMalloc(&d_l2_norm, sizeof(float)));

    // Initialize the error to a large number.
    float error = std::numeric_limits<float>::max();

    // Initialize the data.
    int threads_per_block = 256;
    int blocks = (NUM_POINTS + threads_per_block - 1) / threads_per_block;

    float T_left = 5.0f;
    float T_right = 10.0f;
    initialize<<<blocks, threads_per_block>>>(f_old, T_left, T_right, NUM_POINTS);
    initialize<<<blocks, threads_per_block>>>(f, T_left, T_right, NUM_POINTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Now iterate until the error is sufficiently small.
    // As a safety mechanism, cap the number of iterations.
    int num_iters = 0;

    while (error > TOLERANCE && num_iters < MAX_ITERS) {
        // Initialize the norm data to zero
        CUDA_CHECK(cudaMemset(d_l2_norm, 0, sizeof(float)));

        // Launch kernel to do the calculation
        jacobi<<<blocks, threads_per_block>>>(f_old, f, d_l2_norm, NUM_POINTS);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap f_old and f
        std::swap(f_old, f);

        // Update norm on host; calculate error by normalizing by number of zones and take square root
        CUDA_CHECK(cudaMemcpy(l2_norm, d_l2_norm, sizeof(float), cudaMemcpyDeviceToHost));

        if (*l2_norm == 0.0f) {
            // Deal with first iteration
            error = 1.0f;
        }
        else {
            error = std::sqrt(*l2_norm / NUM_POINTS);
        }

        if (num_iters % 10 == 0) {
            std::cout << "Iteration = " << num_iters << " error = " << error << std::endl;
        }

        ++num_iters;
    }

    if (error <= TOLERANCE && num_iters < MAX_ITERS) {
        std::cout << "Success!\n";
    }
    else {
        std::cout << "Failure!\n";
    }

    // Clean up
    free(l2_norm);
    CUDA_CHECK(cudaFree(d_l2_norm));
    CUDA_CHECK(cudaFree(f_old));
    CUDA_CHECK(cudaFree(f));

    return 0;
}
