#include <iostream>
#include <limits>
#include <cstdio>
#include <cmath>

// FIXME: Add header files for nvshmem.

inline void CUDA_CHECK (cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

// Number of points in the overall spatial domain
#define NUM_POINTS 1048576

// NOTE: `N` should now be passed in as `NUM_POINTS / num_pes`
__global__ void wave_update (float* u, const float* u_old, const float* u_older, float dtdxsq, int N)  // NOTE AGAIN: expect `N` to be `NUM_POINTS / num_pes`.
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // FIXME: Get "my PE" and "number of PEs".
    
    // FIXME: Create a "boundary" flag and set to `true` if this thread is on the boundary of the entire data set,
    // i.e., it's the first index in the first PE or the last index in the last PE.

    if (idx > 0 && idx < N - 1) {  // FIXME: This condition should now be for all indices < `N` and not on the boundary.
        
        // FIXME: If `idx` maps to the last element in (this PE's) `u_old`, `u_old_right` needs to be
        // gotten from the first element in the next PE's `u_old`.
        u_old_right = u_old[idx+1];
        
        // FIXME: If `idx` maps to the first element in (this PE's) `u_old`, `u_old_left` needs to be
        // gotten from the last element in the previous PE's `u_old`.
        u_old_left = u_old[idx-1];
        
        // NOTE: Assuming the FIXMEs above were completed correctly, you shouldn't need to change the following.
        u[idx] = 2.0f * u_old[idx] - u_older[idx] +
                 dtdxsq * (u_old_right - 2.0f * u_old[idx] + u_old_left);
    }
}

__global__ void initialize (float* u, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // FIXME: We need every point in `u` to be initialized based on a unique index within
    // the domain of `NUM_POINTS`. The below works on a single PE implementation since in the
    // calculation `2.0f * M_PI * idx`, `idx` is unique in the domain of `NUM_POINTS` for all threads.
    // However, when we refactor to use multiple PEs, each PE will have identical `idx` values and therefore,
    // each PE will initialize the same values for its `u`, even though what we actually want is
    // for each PE to initialize different values for its `u` such that if they were concatenated,
    // they would be equivalent to the single PE implementation's `u`.
    //
    // Therefore, calculate an offset value using the number of points per PE multiplied by this PE's index,
    // and multiply `idx` below by that offset.
    
    if (idx < N) {
        u[idx] = std::sin(2.0f * M_PI * idx / static_cast<float>(NUM_POINTS - 1));
    }
}

__global__ void check_solution (float* u, float* l2_norm, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < N) {
        // FIXME: `u_correct` should be the same value that `u[idx]` was initialized with in the `initialize`
        // kernel above. Therefore, as you did in `initialize` above, calculate an offset and multiply it
        // by `idx` below.
        float u_correct = std::sin(2.0f * M_PI * idx / static_cast<float>(NUM_POINTS - 1));
        
        // NOTE: Assuming you correctly calculated `u_correct` above, you should not need to
        // modify the rest of this kernel.
        float l2 = (u[idx] - u_correct) * (u[idx] - u_correct);
        atomicAdd(l2_norm, l2);
    }
}

int main() {
    // FIXME: Initialize NVSHMEM
    
    // FIXME: Obtain our NVSHMEM processing element ID and number of PEs/
    
    // FIXME: Each PE (arbitrarily) chooses the GPU corresponding to its ID.
    
    // FIXME: `N` should now equal the total number of points divided by the number of PEs.
    const int N = NUM_POINTS;

    // Allocate space for the grid data, and the temporary buffer
    // for the "old" and "older" data.
    
    // FIXME: `u_older`, `u_old`, and `u` should now be allocated as symmetric memory.
    float* u_older;
    float* u_old;
    float* u;

    CUDA_CHECK(cudaMalloc(&u_older, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&u_old, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&u, N * sizeof(float)));

    // Initialize the data
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    initialize<<<blocks, threads_per_block>>>(u_older, N);
    initialize<<<blocks, threads_per_block>>>(u_old, N);
    initialize<<<blocks, threads_per_block>>>(u, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Now iterate until we've completed a full period
    const float period = 1.0f;
    const float start_time = 0.0f;
    const float stop_time = period;

    // Maximum stable timestep is <= dx
    float stability_factor = 0.5f;
    float dx = 1.0f / (NUM_POINTS - 1);
    float dt = stability_factor * dx;

    float t = start_time;
    const float safety_factor = (1.0f - 1.0e-5f);

    int num_steps = 0;

    while (t < safety_factor * stop_time) {
        // Make sure the last step does not go over the target time
        if (t + dt >= stop_time) {
            dt = stop_time - t;
        }

        float dtdxsq = (dt / dx) * (dt / dx);

        // Launch kernel to do the calculation
        wave_update<<<blocks, threads_per_block>>>(u, u_old, u_older, dtdxsq, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // FIXME: Synchronize all PEs before peforming the swaps.

        // Swap u_old and u_older
        std::swap(u_old, u_older);

        // Swap u and u_old
        std::swap(u, u_old);

        // Print out diagnostics periodically
        // FIXME: Only do the periodic print if this is PE 0.
        if (num_steps % 100000 == 0) {
            std::cout << "Current integration time = " << t << "\n";
        }

        // Update t
        t += dt;
        ++num_steps;
    }

    // Check how close we are to the initial configuration
    float* l2_norm = (float*) malloc(sizeof(float));
    
    // FIXME: Allocate `d_l2_norm` as symmetric memory.
    float* d_l2_norm;
    CUDA_CHECK(cudaMalloc(&d_l2_norm, sizeof(float)));
    
    CUDA_CHECK(cudaMemset(d_l2_norm, 0, sizeof(float)));

    check_solution<<<blocks, threads_per_block>>>(u, d_l2_norm, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // FIXME: Reduce `d_l2_norm` in place over all PEs.

    CUDA_CHECK(cudaMemcpy(l2_norm, d_l2_norm, sizeof(float), cudaMemcpyDeviceToHost));

    // Normalize by number of grid points and take square root
    *l2_norm = std::sqrt(*l2_norm / NUM_POINTS);

    // FIXME: Only print if this is PE 0.
    std::cout << "Error = " << *l2_norm << "\n";

    // Clean up
    // FIXME: Use NVSHMEM to free these values.
    CUDA_CHECK(cudaFree(u_older));
    CUDA_CHECK(cudaFree(u_old));
    CUDA_CHECK(cudaFree(u));
    
    // FIXME: Finalize NVSHMEM.

    return 0;
}
