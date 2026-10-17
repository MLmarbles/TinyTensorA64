#pragma once

#include <cstddef>
#include <vector>

class Tensor {
    private:
        std::vector<float> data_;
        std::vector<std::size_t> shape_;

    public:
        Tensor(const std::vector<std::size_t>& shape);

        void fill(float value);
        void print() const;

        std::size_t size() const;
        const std::vector<std::size_t>& shape() const;

        float& operator()(std::size_t row, std::size_t col);
        const float& operator()(std::size_t row, std::size_t col) const;

        Tensor add(const Tensor& to_add) const;
        Tensor matmul(const Tensor& to_mul) const;
        Tensor matmul_blocked(const Tensor& to_mul, std::size_t block_size = 32) const;
        Tensor matmul_parallel(const Tensor& to_mul, std::size_t num_threads) const;
        Tensor matmul_parallel_blocked(const Tensor& to_mul, std::size_t num_threads, std::size_t block_size) const;
        void relu();

        void randomize(float min = -0.1f, float max = 0.1f);
};