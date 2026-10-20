#include <cstddef>
#include <vector>
#include <arm_neon.h>
#include <iostream>
#include <chrono>

float dot_scalar(const std::vector<float>& a, const std::vector<float>& b)
{
    float sum = 0.0f;

    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

float dot_neon(const float* a, const float* b, std::size_t n)
{
    float32x4_t acc = vdupq_n_f32(0.0f);

    std::size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);

        acc = vfmaq_f32(acc, va, vb);
    }

    float sum = vaddvq_f32(acc);

    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}



int main()
{
    const std::size_t N = 1'000'000;
    const int runs = 100;
    std::vector<float> a(N);
    std::vector<float> b(N);

    for (std::size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i % 100);
        b[i] = static_cast<float>((i + 1) % 100);
    }

    long long total = 0;
    float scalar_result = 0.0f;
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        scalar_result = dot_scalar(a, b);

        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration_cast<std::chrono::microseconds>((end - start)).count();
    }

        std::cout << "Scalar result: " << scalar_result << '\n';
        std::cout << "Time: " << total/runs << " average \n";

    total = 0;
    float neon_result = 0.0f;
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        neon_result = dot_neon(&a[0], &b[0], N);

        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration_cast<std::chrono::microseconds>((end - start)).count();
    }

        std::cout << "NEON result: " << neon_result << '\n';
        std::cout << "Time: " << total/runs << " average \n"; 
}