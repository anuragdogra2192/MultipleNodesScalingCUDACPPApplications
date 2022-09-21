#include <iostream>
#include <cstdlib>

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

    if (idx < N) {
        int value = input[idx];

        int histogram_index = (value * NUM_BUCKETS) / MAX_VALUE;

        atomicAdd(&histogram[histogram_index], 1);
    }
}

int main(int argc, char** argv) {
    const int N = NUM_INPUTS;

    // Construct histogram input data on the host
    int* input = (int*) malloc(N * sizeof(int));

    // The input data ranges from 0 to MAX_VALUE - 1
    for (int i = 0; i < N; ++i) {
        input[i] = rand() % MAX_VALUE;
    }

    // Copy to device
    int* d_input;
    CUDA_CHECK(cudaMalloc((void**) &d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate the histogram array
    int* histogram = (int*) malloc(NUM_BUCKETS * sizeof(int));
    memset(histogram, 0, NUM_BUCKETS * sizeof(int));

    int* d_histogram;
    CUDA_CHECK(cudaMalloc((void**) &d_histogram, NUM_BUCKETS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BUCKETS * sizeof(int)));

    // Perform the histogramming
    int threads_per_block = 256;
    int blocks = (NUM_INPUTS + threads_per_block - 1) / threads_per_block;

    histogram_kernel<<<blocks, threads_per_block>>>(d_input, d_histogram, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy data back to the host and sanity check a few values
    CUDA_CHECK(cudaMemcpy(histogram, d_histogram, NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Histogram counters:" << std::endl << std::endl;
    int num_buckets_to_print = 4;
    for (int i = 0; i < NUM_BUCKETS; i += NUM_BUCKETS / num_buckets_to_print) {
        std::cout << "Bucket [" << i * (MAX_VALUE / NUM_BUCKETS) << ", " << (i + 1) * (MAX_VALUE / NUM_BUCKETS) - 1 << "]: " << histogram[i];
        std::cout << std::endl;
        if (i < NUM_BUCKETS - NUM_BUCKETS / num_buckets_to_print - 1) {
            std::cout << "..." << std::endl;
        }
    }

    free(input);
    free(histogram);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_histogram));

    return 0;
}
