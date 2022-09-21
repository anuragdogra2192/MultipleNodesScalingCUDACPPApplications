#include <iostream>
#include <cstdlib>
#include <chrono>

#include <nvshmem.h>
#include <nvshmemx.h>

inline void CUDA_CHECK (cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

#define NUM_BUCKETS   16
#define MAX_VALUE     1048576
#define NUM_INPUTS    65536

__global__ void histogram_kernel(const int* input, int* histogram, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int n_pes = nvshmem_n_pes();
    int buckets_per_pe = NUM_BUCKETS / n_pes;

    if (idx < N) {
        int value = input[idx];

        // Calculate the "global" histogram index
        int global_histogram_index = ((size_t) value * NUM_BUCKETS) / MAX_VALUE;

        // Figure out which PE that histogram index belongs to.
        // We assume every PE has an equal number of buckets, and
        // we start with PE 0 containing the first bucket and going
        // up to the first 1 / n_pes worth of buckets, with a similar
        // approach for the other PEs. So we can do simple integer
        // division here.
        int target_pe = global_histogram_index / buckets_per_pe;

        // Now figure out the local histogram index on that PE.
        // We just need to offset by the starting bucket for that PE.
        int local_histogram_index = global_histogram_index - target_pe * buckets_per_pe;

        nvshmem_int_atomic_add(&histogram[local_histogram_index], 1, target_pe);
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

    // Each device handles a fraction 1 / n_pes of the work.
    const int N = NUM_INPUTS / n_pes;

    // Construct histogram input data on the host
    int* input = (int*) malloc(N * sizeof(int));

    // Initialize a different random number seed for each PE.
    srand(my_pe);

    // The input data ranges from 0 to MAX_VALUE - 1
    for (int i = 0; i < N; ++i) {
        input[i] = rand() % MAX_VALUE;
    }

    // Copy to device
    int* d_input;
    d_input = (int*) nvshmem_malloc(N * sizeof(int));
    CUDA_CHECK(cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate the histogram array -- sized to the full histogram on the
    // host, and only the relevant part for each GPU on the device.
    int* histogram = (int*) malloc(NUM_BUCKETS * sizeof(int));
    memset(histogram, 0, NUM_BUCKETS * sizeof(int));

    int buckets_per_pe = NUM_BUCKETS / n_pes;

    int* d_histogram;
    d_histogram = (int*) nvshmem_malloc(buckets_per_pe * sizeof(int));
    CUDA_CHECK(cudaMemset(d_histogram, 0, buckets_per_pe * sizeof(int)));

    // Also allocate a full-sized device histogram for concatenation
    int* d_concatenated_histogram = (int*) nvshmem_malloc(NUM_BUCKETS * sizeof(int));
    CUDA_CHECK(cudaMemset(d_concatenated_histogram, 0, NUM_BUCKETS * sizeof(int)));

    // Perform a barrier for reasonably accurate timing
    nvshmem_barrier_all();

    using namespace std::chrono;

    high_resolution_clock::time_point tabulation_start = high_resolution_clock::now();

    // Perform the histogramming
    int threads_per_block = 256;
    int blocks = (NUM_INPUTS / n_pes + threads_per_block - 1) / threads_per_block;

    histogram_kernel<<<blocks, threads_per_block>>>(d_input, d_histogram, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    nvshmem_barrier_all();

    high_resolution_clock::time_point tabulation_end = high_resolution_clock::now();

    // Concatenate over all PEs

    high_resolution_clock::time_point combination_start = high_resolution_clock::now();

    nvshmem_int_collect(NVSHMEM_TEAM_WORLD, d_concatenated_histogram, d_histogram, buckets_per_pe);

    high_resolution_clock::time_point combination_end = high_resolution_clock::now();

    // Print out results on PE 0
    if (my_pe == 0) {
        duration<double> tabulation_time = duration_cast<duration<double>>(tabulation_end - tabulation_start);
        std::cout << "Tabulation time = " << tabulation_time.count() * 1000 << " ms" << std::endl << std::endl;

        duration<double> combination_time = duration_cast<duration<double>>(combination_end - combination_start);
        std::cout << "Combination time = " << combination_time.count() * 1000 << " ms" << std::endl << std::endl;

        // Copy data back to the host
        CUDA_CHECK(cudaMemcpy(histogram, d_concatenated_histogram, NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "Histogram counters:" << std::endl << std::endl;
        int num_buckets_to_print = 4;
        for (int i = 0; i < NUM_BUCKETS; i += NUM_BUCKETS / num_buckets_to_print) {
            std::cout << "Bucket [" << i * (MAX_VALUE / NUM_BUCKETS) << ", " << (i + 1) * (MAX_VALUE / NUM_BUCKETS) - 1 << "]: " << histogram[i];
            std::cout << std::endl;
            if (i < NUM_BUCKETS - NUM_BUCKETS / num_buckets_to_print - 1) {
                std::cout << "..." << std::endl;
            }
        }
    }

    free(input);
    free(histogram);
    nvshmem_free(d_input);
    nvshmem_free(d_histogram);

    // Finalize nvshmem
    nvshmem_finalize();

    return 0;
}
