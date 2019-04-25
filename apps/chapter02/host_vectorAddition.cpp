#include <gflags/gflags.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

DEFINE_int32(N, 1000, "Number of elements in vectors");

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
    for (int i = 0; i < n; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
}

void verification(const float *h_A, const float *h_B, const float *h_C, const int n)
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
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << "vector addition [host version]" << std::endl;

    std::vector<float> h_A(FLAGS_N), h_B(FLAGS_N), h_C(FLAGS_N);

    auto start = std::chrono::steady_clock::now();
    inputInitialization(h_A.data(), h_B.data(), FLAGS_N);
    vecAdd(h_A.data(), h_B.data(), FLAGS_N, h_C.data());
    verification(h_A.data(), h_B.data(), h_C.data(), FLAGS_N);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us\n";

    return EXIT_SUCCESS;
}
