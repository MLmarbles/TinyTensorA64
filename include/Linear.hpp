#pragma once

#include "Tensor.hpp"

class Linear {
    private:
        Tensor weights_;
        Tensor bias_;

    public:
        Linear(std::size_t input_features, std::size_t output_features);

        Tensor forward(const Tensor& input) const;

        Tensor& weights();
        Tensor& bias(); 
        
};