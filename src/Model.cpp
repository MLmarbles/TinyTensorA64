#include "Model.hpp"
#include <iostream>

Model::Model()
    : layer1_({3, 4}),
      layer2_({4, 2})
{
}

Tensor Model::forward(const Tensor& input) const
{
    std::cout << "before: \n";
    input.print();

    Tensor x = layer1_.forward(input);
    std::cout << "after layer 1: \n";
    x.print();

    x.relu();

    std::cout << "after relu: \n";
    x.print();

    x = layer2_.forward(x);

    std::cout << "after layer 2: \n";
    x.print();

    return x;
}