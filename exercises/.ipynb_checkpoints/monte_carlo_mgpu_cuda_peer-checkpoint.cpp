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
    // Start on GPU 0
    cudaSetDevice(0);

    int device_count;
    cudaGetDeviceCount(&device_count);

    // Allocate host and device values
    int* hits;
    hits = (int*) malloc(sizeof(int));

    int* d_hits;
    cudaMalloc((void**) &d_hits, sizeof(int));

    // Initialize number of hits and copy to device
    *hits = 0;
    cudaMemcpy(d_hits, hits, sizeof(int), cudaMemcpyHostToDevice);

    // Check that every device can access its peers.
    // If it can, go ahead and enable that access.

    for (int dev = 0; dev < device_count; ++dev) {
        cudaSetDevice(dev);
        for (int peer = 0; peer < device_count; ++peer) {
            if (peer != dev) {
                int can_access_peer;
                cudaDeviceCanAccessPeer(&can_access_peer, dev, peer);  // Store whether or not `dev` and `peer` GPUs can access each other as peers in `can_access_peer`.

                if (can_access_peer) {
                     cudaDeviceEnablePeerAccess(peer, 0); // Enable peer access between this device and `peer`.
                } else {
                    std::cout << "Device " << dev << " could not access peer " << peer << std::endl;
                    return -1;
                }
            }
        }
    }

    // Launch kernel to do the calculation
    int threads_per_block = 256;
    int blocks = (N / device_count + threads_per_block - 1) / threads_per_block;

    // Allow for asynchronous execution by launching all kernels first
    // and then synchronizing on all devices after.
    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        calculate_pi<<<blocks, threads_per_block>>>(d_hits, i);
    }

    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    // Copy final result back to the host
    cudaMemcpy(hits, d_hits, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate final value of pi
    float pi_est = (float) *hits / (float) (N) * 4.0f;

    // Print out result
    std::cout << "Estimated value of pi = " << pi_est << std::endl;
    std::cout << "Error = " << std::abs((M_PI - pi_est) / pi_est) << std::endl;

    // Clean up
    free(hits);
    cudaFree(d_hits);

    return 0;
}

