#include <iostream>
#include <curand_kernel.h>

#include <nvshmem.h>
#include <nvshmemx.h>

inline void CUDA_CHECK (cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

#define N 1024*1024

__global__ void calculate_pi(int* hits, int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize random number state (unique for every thread in the grid)
    int offset = 0;
    curandState_t curand_state;
    curand_init(seed, idx, offset, &curand_state);

    // Generate random coordinates within (0.0, 1.0]
    float x = curand_uniform(&curand_state);
    float y = curand_uniform(&curand_state);

    // Increment hits counter if this point is inside the circle
    if (x * x + y * y <= 1.0f) {
        atomicAdd(hits, 1);
    }
}


int main(int argc, char** argv) {
    // Initialize NVSHMEM
    nvshmem_init();

    // Obtain our NVSHMEM processing element ID and number of PEs
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // Each PE (arbitrarily) chooses the GPU corresponding to its ID
    int device = my_pe;
    CUDA_CHECK(cudaSetDevice(device));

    // Allocate host and device values
    int* hits;
    hits = (int*) malloc(sizeof(int));

    int* d_hits;
    CUDA_CHECK(cudaMalloc((void**) &d_hits, sizeof(int)));

    // Initialize number of hits and copy to device
    *hits = 0;
    CUDA_CHECK(cudaMemcpy(d_hits, hits, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel to do the calculation
    int threads_per_block = 256;
    int blocks = (N / n_pes + threads_per_block - 1) / threads_per_block;

    int seed = my_pe;
    calculate_pi<<<blocks, threads_per_block>>>(d_hits, seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy final result back to the host
    CUDA_CHECK(cudaMemcpy(hits, d_hits, sizeof(int), cudaMemcpyDeviceToHost));

    // Calculate final value of pi
    float pi_est = (float) *hits / (float) (N / n_pes) * 4.0f;
    //No aggregation yet :) that we will see soon.
    // Print out result
    std::cout << "Estimated value of pi on PE " << my_pe << " = " << pi_est << std::endl;
    std::cout << "Relative error on PE " << my_pe << " = " << std::abs((M_PI - pi_est) / pi_est) << std::endl;

    free(hits);
    CUDA_CHECK(cudaFree(d_hits));

    // Finalize nvshmem
    nvshmem_finalize();

    return 0;
}
