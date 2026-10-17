#include "Tensor.hpp"
#include <iostream>
#include <stdexcept>
#include <random>
#include <thread>
#include <vector>

Tensor::Tensor(const std::vector<std::size_t>& shape)
    : shape_(shape)
{
    std::size_t total = 1;

    for (auto dim : shape_) {
        total *= dim;
    }

    data_.resize(total);
}

void Tensor::fill(float value)
{
    for (auto& x : data_) {
        x = value;
    }
}

void Tensor::print() const 
{
    for (std::size_t r = 0; r < shape_[0]; ++r) {
        for (std::size_t c = 0; c < shape_[1]; ++c) {
            std::cout << (*this)(r, c) << ' ';
        }
        std::cout << '\n';
    }
}

float& Tensor::operator()(std::size_t row, std::size_t col)
{
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("tensor index out of bounds");
    }
    std::size_t cols = shape_[1];
    return data_[row * cols + col];
}

const float& Tensor::operator()(std::size_t row, std::size_t col) const
{
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("tensor index out of bounds");
    }
    std::size_t cols = shape_[1];
    return data_[row * cols + col];
}

Tensor Tensor::add(const Tensor& to_add) const
{
    if (shape_ != to_add.shape_) {
        throw std::invalid_argument("tensor sizes must match");
    }

    Tensor result({shape_[0], shape_[1]});

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + to_add.data_[i];
    }

    return result;
}

Tensor Tensor::matmul(const Tensor& to_mul) const
{
    if (shape_[1] != to_mul.shape_[0]) {
        throw std::invalid_argument("invalid tensor sizes for matmul");
    }

    Tensor result({shape_[0], to_mul.shape_[1]});

    for (std::size_t i = 0; i < shape_[0]; ++i) {
        for (std::size_t k = 0; k < shape_[1]; ++k) {

            float a = (*this)(i, k);

            for (std::size_t j = 0; j < to_mul.shape_[1]; ++j) {
                result(i, j) += a * to_mul(k, j);
            }
        }
    }

    return result;
} 

Tensor Tensor::matmul_blocked(const Tensor& to_mul, std::size_t block_size) const
{
    if (shape_[1] != to_mul.shape_[0]) {
        throw std::invalid_argument("invalid tensor sizes for matmul");
    }

    Tensor result({shape_[0], to_mul.shape_[1]});

    for (std::size_t ii = 0; ii < shape_[0]; ii += block_size) {
        for (std::size_t kk = 0; kk < shape_[1]; kk += block_size) {
            for (std::size_t jj = 0; jj < to_mul.shape_[1]; jj += block_size) {

                const std::size_t i_end = std::min(ii + block_size, shape_[0]);
                const std::size_t k_end = std::min(kk + block_size, shape_[1]);
                const std::size_t j_end = std::min(jj + block_size, to_mul.shape_[1]);

                for (std::size_t i = ii; i < i_end; ++i) {
                    for (std::size_t k = kk; k < k_end; ++k) {

                        const float a = (*this)(i, k);

                        for (std::size_t j = jj; j < j_end; ++j) {
                            result(i, j) += a * to_mul(k, j);
                        }
                    }
                }
            }
        }
    }

    return result;
}

Tensor Tensor::matmul_parallel(const Tensor& to_mul, std::size_t num_threads) const {
    if (shape_[1] != to_mul.shape_[0]) {
        throw std::invalid_argument("invalid tensor sizes for matmul");
    }

    if (num_threads == 0) {
        throw std::invalid_argument("num_threads must be greater than 0");
    }

    if (num_threads > shape_[0]) {
        num_threads = shape_[0];
    }

    Tensor result({shape_[0], to_mul.shape_[1]});

    std::size_t rows_per_thread = shape_[0] / num_threads;
    const std::size_t extra_rows = shape_[0] % num_threads;
    std::vector<std::thread> threads;
    std::size_t row_start = 0;

    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t rows_for_this_thread = rows_per_thread;

        if (t < extra_rows) {
            ++rows_for_this_thread;
        }

        std::size_t row_end = row_start + rows_for_this_thread;

        threads.emplace_back([&, row_start, row_end]() {
            for (std::size_t i = row_start; i < row_end; ++i) {
                for (std::size_t k = 0; k < shape_[1]; ++k) {
                    const float a = (*this)(i, k);

                    for (std::size_t j = 0; j < to_mul.shape_[1]; ++j) {
                        result(i, j) += a * to_mul(k, j);
                    }
                }
            }
        });
        row_start = row_end;
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

Tensor Tensor::matmul_parallel_blocked(const Tensor& to_mul, std::size_t num_threads, std::size_t block_size) const
{
    if (shape_[1] != to_mul.shape_[0]) {
        throw std::invalid_argument("invalid tensor sizes for matmul");
    }

    if (num_threads == 0) {
        throw std::invalid_argument("num_threads must be greater than 0");
    }

    Tensor result({shape_[0], to_mul.shape_[1]});

    struct Tile {
        std::size_t ii;
        std::size_t jj;
    };

    std::vector<Tile> tiles;

    for (std::size_t ii = 0; ii < shape_[0]; ii += block_size) {
        for (std::size_t jj = 0; jj < to_mul.shape_[1]; jj += block_size) {
            tiles.push_back({ii, jj});
        }
    }

    if (num_threads > tiles.size()) {
        num_threads = tiles.size();
    }

    std::vector<std::thread> threads;

    auto worker = [&](std::size_t thread_id) {
        for (std::size_t tile_index = thread_id; tile_index < tiles.size(); tile_index += num_threads) {
            const auto [ii, jj] = tiles[tile_index];

            const std::size_t i_end = std::min(ii + block_size, shape_[0]);
            const std::size_t j_end = std::min(jj + block_size, to_mul.shape_[1]);

            for (std::size_t kk = 0; kk < shape_[1]; kk += block_size) {

                const std::size_t k_end = std::min(kk + block_size, shape_[1]);

                for (std::size_t i = ii; i < i_end; ++i) {
                    for (std::size_t k = kk; k < k_end; ++k) {

                        const float a = (*this)(i, k);

                        for (std::size_t j = jj; j < j_end; ++j) {
                            result(i, j) += a * to_mul(k, j);
                        }
                    }
                }
            }
        }
    };
    for (std::size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

void Tensor::relu()
{
    for (std::size_t i = 0; i < data_.size(); ++i) {
        if (data_[i] < 0) {
            data_[i] = 0;
        }
    }
}

void Tensor::randomize(float min, float max)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(min, max);

    for (auto& value : data_) {
        value = dist(gen);
    }
}