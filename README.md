# Multiple Nodes Scaling CUDA C++ Applications
Scaling CUDA C++ Applications to Multiple Nodes

* Skills-based coding assessment: Students must refactor a single-GPU 1D wave function solver to be GPU-cluster-ready with NVSHMEM.
* Tools, libraries, and frameworks: CUDA, MPI, NVSHMEM, CUB (https://nvlabs.github.io/cub/)

Present-day high-performance computing (HPC) and deep learning applications benefit from, and even require, cluster-scale GPU compute power. Writing CUDA® applications that can correctly and efficiently utilize GPUs across a cluster requires a distinct set of skills. In this workshop, you’ll learn the tools and techniques needed to write CUDA C++ applications that can scale efficiently to clusters of NVIDIA GPUs. 

You’ll do this by working on code from several CUDA C++ applications in an interactive cloud environment backed by several NVIDIA GPUs. You’ll gain exposure to a handful of multi-GPU programming methods, including CUDA-aware Message Passing Interface (MPI), before proceeding to the main focus of this course, NVSHMEM™.

NVSHMEM is a parallel programming interface based on OpenSHMEM that provides efficient and scalable communication for NVIDIA GPU clusters. NVSHMEM creates a global address space for data that spans the memory of multiple GPUs and can be accessed with fine-grained GPU-initiated operations, CPU-initiated operations, and operations on CUDA streams. NVSHMEM's asynchronous, GPU-initiated data transfers eliminate synchronization overheads between the CPU and the GPU. They also enable long-running kernels that include both communication and computation, reducing overheads that can limit an application’s performance when strong scaling. NVSHMEM has been used on systems such as the Summit supercomputer located at the Oak Ridge Leadership Computing Facility (OLCF), the Lawrence Livermore National Laboratory’s Sierra supercomputer, and the NVIDIA DGX™ A100.  NVSHMEM supports PCIe.

What is NVIDIA CUDA Peer-to-Peer (P2P)?
P2P is functionality in NVIDIA GPU's that allow CUDA programs to access and transfer data from one GPU's memory to another without having to go through a shared pool of system memory attached to a CPU. It has been a feature for NVIDIA GPU's for 8 or 9 years. P2P together with "Unified Virtual Addressing" (UVA) were big improvements in CUDA for GPU computing. They allowed efficient use of multi-GPU and multi-Node system environments and simplified programming for highly parallel code.

Reference: https://www.pugetsystems.com/labs/hpc/P2P-peer-to-peer-on-NVIDIA-RTX-2080Ti-vs-GTX-1080Ti-GPUs-1331/

### Learning outcomes
* Learn several methods for writing multi-GPU CUDA C++ applications
* Use a variety of multi-GPU communication patterns and understand their tradeoffs
* Write portable, scalable CUDA code with the single-program multiple-data (SPMD) paradigm using CUDA-aware MPI and NVSHMEM
* Improve multi-GPU SPMD code with NVSHMEM’s symmetric memory model and its ability to perform GPU-initiated data transfers
* Get practice with common multi-GPU coding paradigms like domain decomposition and halo exchanges

MPI + CUDA =  Single Program, Multiple Data (SPMD)
