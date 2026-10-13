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

};