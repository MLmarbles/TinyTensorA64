#include "Tensor.hpp"
#include <iostream>

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