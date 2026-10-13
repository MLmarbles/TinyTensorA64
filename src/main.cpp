#include "tensor.hpp"

int main() 
{
    Tensor x({2, 3});

    x.fill(1.0f);
    x(0, 1) = 5.0f;

    x.print();
}