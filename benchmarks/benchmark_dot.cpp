#include <cstddef>
#include <vector>
#include <arm_neon.h>
#include <iostream>

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
    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {5, 6, 7, 8};

    std::cout << dot_scalar(a, b) << '\n';
    std::cout << dot_neon(&a[0], &b[0], a.size()) << '\n';
}