#include <iostream>
#include <limits>
#include <cstdio>

#include <nvshmem.h>
#include <nvshmemx.h>

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

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // Don't participate if we're the leftmost PE and on
    // the leftmost boundary point, or if we're the rightmost
    // PE and on the rightmost boundary point (as these are fixed).
    bool on_boundary = false;

    if (my_pe == 0 && idx == 0) {
        on_boundary = true;
    }
    else if (my_pe == n_pes - 1 && idx == N - 1) {
        on_boundary = true;
    }

    if (idx < N && !on_boundary) {
        // Retrieve the left and right points in the old data.
        // If we're fully in the interior of the local domain,
        // this is fully a local access. Otherwise, we need to
        // reach out to the remote PE to the left or right.

        float f_left;
        float f_right;

        if (idx == 0) {
            // Note we don't get here if my_pe == 0.
            f_left = nvshmem_float_g(&f_old[N - 1], my_pe-1);
        }
        else {
            f_left = f_old[idx-1];
        }

        if (idx == N - 1) {
            // Note we don't get here if my_pe == n_pes - 1.
            f_right = nvshmem_float_g(&f_old[0], my_pe+1);
        }
        else {
            f_right = f_old[idx+1];
        }

        f[idx] = 0.5f * (f_right + f_left);

        float l2 = (f[idx] - f_old[idx]) * (f[idx] - f_old[idx]);

        atomicAdd(l2_norm, l2);
    }
}

__global__ void initialize (float* f, float T_left, float T_right, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    if (idx == 0 && my_pe == 0) {
        f[idx] = T_left;
    }
    else if (idx == N - 1 && my_pe == n_pes - 1) {
        f[idx] = T_right;
    }
    else if (idx < N - 1) {
        f[idx] = 0.0f;
    }
}

int main() {
    // Initialize NVSHMEM
    nvshmem_init();

    // Obtain our NVSHMEM processing element ID and number of PEs
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // Each PE (arbitrarily) chooses the GPU corresponding to its ID
    int device = my_pe;
    CUDA_CHECK(cudaSetDevice(device));

    // Each device handles a fraction 1 / n_pes of the work.
    const int N = NUM_POINTS / n_pes;

    // Allocate space for the grid data, and the temporary buffer
    // for the "old" data.
    float* f_old = (float*) nvshmem_malloc(N * sizeof(float));
    float* f = (float*) nvshmem_malloc(N * sizeof(float));

    // Allocate memory for the L2 norm, on both the host and device.
    float* l2_norm = (float*) malloc(sizeof(float));
    float* d_l2_norm = (float*) nvshmem_malloc(sizeof(float));

    // Initialize the error to a large number.
    float error = std::numeric_limits<float>::max();

    // Initialize the data.
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    float T_left = 5.0f;
    float T_right = 10.0f;
    initialize<<<blocks, threads_per_block>>>(f_old, T_left, T_right, N);
    initialize<<<blocks, threads_per_block>>>(f, T_left, T_right, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Now iterate until the error is sufficiently small.
    // As a safety mechanism, cap the number of iterations.
    int num_iters = 0;

    while (error > TOLERANCE && num_iters < MAX_ITERS) {
        // Initialize the norm data to zero
        CUDA_CHECK(cudaMemset(d_l2_norm, 0, sizeof(float)));

        // Launch kernel to do the calculation
        jacobi<<<blocks, threads_per_block>>>(f_old, f, d_l2_norm, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap f_old and f
        std::swap(f_old, f);

        // Sum the L2 norm over all PEs
        // Note this is a blocking API, so no explicit barrier is needed afterward
        nvshmem_float_sum_reduce(NVSHMEM_TEAM_WORLD, d_l2_norm, d_l2_norm, 1);

        // Update norm on host; calculate error by normalizing by number of zones and take square root
        CUDA_CHECK(cudaMemcpy(l2_norm, d_l2_norm, sizeof(float), cudaMemcpyDeviceToHost));

        if (*l2_norm == 0.0f) {
            // Deal with first iteration
            error = 1.0f;
        }
        else {
            error = std::sqrt(*l2_norm / NUM_POINTS);
        }

        if (num_iters % 10 == 0 && my_pe == 0) {
            std::cout << "Iteration = " << num_iters << " error = " << error << std::endl;
        }

        ++num_iters;
    }

    if (my_pe == 0) {
        if (error <= TOLERANCE && num_iters < MAX_ITERS) {
            std::cout << "Success!\n";
        }
        else {
            std::cout << "Failure!\n";
        }
    }

    free(l2_norm);
    nvshmem_free(d_l2_norm);
    nvshmem_free(f_old);
    nvshmem_free(f);

    // Finalize nvshmem
    nvshmem_finalize();

    return 0;
}
