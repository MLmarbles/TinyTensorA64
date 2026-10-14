#include "Tensor.hpp"
#include <Linear.hpp>

int main() 
{
    Linear layer(3, 2);

    layer.weights()(0, 0) = 1;
    layer.weights()(0, 1) = 2;
    layer.weights()(1, 0) = 3;
    layer.weights()(1, 1) = 4;
    layer.weights()(2, 0) = 5;
    layer.weights()(2, 1) = 6;

    layer.bias()(0, 0) = 3;
    layer.bias()(0, 1) = 3;

    Tensor input({1, 3});

    input(0, 0) = 1;
    input(0, 1) = 2;
    input(0, 2) = 3;

    Tensor output = layer.forward(input);

    output.print();
}