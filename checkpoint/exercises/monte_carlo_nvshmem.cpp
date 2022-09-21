#include <iostream>
#include <curand_kernel.h>

#include <mpi.h>

#include <nvshmem.h>
#include <nvshmemx.h>

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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Each PE (arbitrarily) chooses the GPU corresponding to its ID
    int dev = rank;
    cudaSetDevice(dev);

    // Ensure that we don't have more ranks than GPUs
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (num_ranks > device_count) {
        std::cout << "Error: more ranks than GPUs" << std::endl;
        return -1;
    }

    // Initialize NVSHMEM (with MPI support)
    nvshmemx_init_attr_t attr;
    MPI_Comm comm = MPI_COMM_WORLD;
    attr.mpi_comm = &comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // Obtain our NVSHMEM processing element ID and the number of PEs
    int my_pe = nvshmem_my_pe();
    int n_pes = FIXME;

    // Allocate device data
    int* d_hits = (int*) nvshmem_malloc(FIXME);

    // Initialize number of hits on device
    nvshmem_int_p(d_hits, FIXME, FIXME);

    // Launch kernel to do the calculation
    int threads_per_block = 256;
    int blocks = (N / device_count + threads_per_block - 1) / threads_per_block;

    calculate_pi<<<blocks, threads_per_block>>>(d_hits, dev);
    cudaDeviceSynchronize();

    // Insert an NVSHMEM barrier across all PEs
    nvshmem_barrier_all();

    // Accumulate the results across all PEs to the result on PE 0
    if (my_pe == 0) {
        int total_hits = 0;
        for (int i = 0; i < n_pes; ++i) {
            total_hits += nvshmem_int_g(d_hits, FIXME);
        }
 
        // Calculate final value of pi
        float pi_est = (float) total_hits / (float) (N) * 4.0f;

        // Print out result
        std::cout << "Estimated value of pi = " << pi_est << std::endl;
        std::cout << "Error = " << std::abs((M_PI - pi_est) / pi_est) << std::endl;
    }

    // Clean up
    nvshmem_free(d_hits);

    // Finalize nvshmem
    nvshmem_finalize();

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
