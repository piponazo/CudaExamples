#include <gflags/gflags.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

DEFINE_int32(N, 1000, "Number of elements in vectors");

__global__ void vecAddKernel(const float *a, const float *b, const int n, float *c)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void inputInitialization(float *h_A, float *h_B, const int n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);

    for (int i = 0; i < n; i++) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }
}

void vecAdd(const float *h_A, const float *h_B, const int n, float *h_C)
{
    const size_t sizeInBytes = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeInBytes);
    cudaMalloc(&d_B, sizeInBytes);
    cudaMalloc(&d_C, sizeInBytes);

    cudaMemcpy(d_A, h_A, sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeInBytes, cudaMemcpyHostToDevice);

    vecAddKernel<<<std::ceil(n / 256.0), 256>>>(d_A, d_B, n, d_C);

    cudaMemcpy(h_C, d_C, sizeInBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void verificationHost(const float *h_A, const float *h_B, const float *h_C, const int n)
{
    for (int i = 0; i < n; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char **argv)
{
    gflags::SetUsageMessage("appName -N=numberOfElements");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    auto start = std::chrono::steady_clock::now();

    std::vector<float> h_A(FLAGS_N), h_B(FLAGS_N), h_C(FLAGS_N);
    inputInitialization(h_A.data(), h_B.data(), FLAGS_N);
    vecAdd(h_A.data(), h_B.data(), FLAGS_N, h_C.data());
    verificationHost(h_A.data(), h_B.data(), h_C.data(), FLAGS_N);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us\n";

    return EXIT_SUCCESS;
}
