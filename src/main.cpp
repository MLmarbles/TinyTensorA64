#include "tensor.hpp"

int main() 
{
    Tensor x({2, 2});
    Tensor y({2, 2});
    
    x(0, 0) = 1;
    x(0, 1) = 2;
    x(1, 0) = 3;
    x(1, 1) = 4;

    y(0, 0) = 5;
    y(0, 1) = 6;
    y(1, 0) = 7;
    y(1, 1) = 8;

    Tensor z = x.matmul(y);

    z.print();
}