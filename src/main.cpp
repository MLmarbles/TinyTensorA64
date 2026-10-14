#include "Tensor.hpp"
#include "Linear.hpp"
#include "Model.hpp"

int main() 
{
    Model model;

    Tensor x({1, 3});

    x(0, 0) = 1;
    x(0, 1) = 2;
    x(0, 2) = 3;

    Tensor out = model.forward(x);

    out.print();
}