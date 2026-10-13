#include "Tensor.hpp"
#include <iostream>
#include <stdexcept>

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
    std::size_t cols = shape_[1];
    return data_[row * cols + col];
}

const float& Tensor::operator()(std::size_t row, std::size_t col) const
{
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
        for (std::size_t j = 0; j < to_mul.shape_[1]; ++j) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < shape_[1]; ++k) {
                sum +=  (*this)(i, k) * to_mul(k, j);
            }

            result(i, j) = sum;
        }
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