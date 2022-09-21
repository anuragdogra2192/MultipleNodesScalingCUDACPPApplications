#include <iostream>
#include <curand_kernel.h>

#define N 1024*1024

__global__ void calculate_pi(int* hits, int device) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize random number state (unique for every thread in the grid)
    int seed = device;
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
    // Determine number of GPUs
    int device_count;
    cudaGetDeviceCount(&device_count);

    std::cout << "Using " << device_count << " GPUs" << std::endl;

    // Allocate host and device values (one per GPU)
    int** hits = (int**) malloc(device_count * sizeof(int*));
    for (int i = 0; i < device_count; ++i) {
        hits[i] = (int*) malloc(sizeof(int));
    }

    int** d_hits = (int**) malloc(device_count * sizeof(int*));
    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        cudaMalloc((void**) &d_hits[i], sizeof(int));
    }

    // Initialize number of hits and copy to device
    for (int i = 0; i < device_count; ++i) {
        *hits[i] = 0;
        cudaSetDevice(i);
        cudaMemcpy(d_hits[i], hits[i], sizeof(int), cudaMemcpyHostToDevice);
    }

    // Launch kernel to do the calculation
    int threads_per_block = 256;
    int blocks = (N / device_count + threads_per_block - 1) / threads_per_block;

    // Allow for asynchronous execution by launching all kernels first
    // and then synchronizing on all devices after.
    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        calculate_pi<<<blocks, threads_per_block>>>(d_hits[i], i);
    }

    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    // Copy final result back to the host
    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        cudaMemcpy(hits[i], d_hits[i], sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Sum number of hits over all devices
    int hits_total = 0;
    for (int i = 0; i < device_count; ++i) {
        hits_total += *hits[i];
    }

    // Calculate final value of pi
    float pi_est = (float) hits_total / (float) (N) * 4.0f;

    // Print out result
    std::cout << "Estimated value of pi = " << pi_est << std::endl;
    std::cout << "Error = " << std::abs((M_PI - pi_est) / pi_est) << std::endl;

    // Clean up
    for (int i = 0; i < device_count; ++i) {
        free(hits[i]);
        cudaFree(d_hits[i]);
    }
    free(hits);
    free(d_hits);

    return 0;
}
