#include "Linear.hpp"

Linear::Linear(std::size_t input, std::size_t output) 
    : weights_({input, output}),
      bias_({1, output})
{
}

Tensor Linear::forward(const Tensor& input) const
{
    Tensor output = input.matmul(weights_);

    return output.add(bias_);
}  

Tensor& Linear::weights() 
{
    return weights_;
}

Tensor& Linear::bias()
{
    return bias_;
}