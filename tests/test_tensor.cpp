#include "Tensor.hpp"

#include <cassert>
#include <iostream>

void test_add()
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

    Tensor z = x.add(y);

    assert(z(0, 0) == 6);
    assert(z(0, 1) == 8);
    assert(z(1, 0) == 10);
    assert(z(1, 1) == 12);
}

void test_relu()
{
    Tensor x({2, 2});

    x(0, 0) = -5;
    x(0, 1) = 2;
    x(1, 0) = -1;
    x(1, 1) = 7;

    x.relu();

    assert(x(0, 0) == 0);
    assert(x(0, 1) == 2);
    assert(x(1, 0) == 0);
    assert(x(1, 1) == 7);
}

void test_matmul()
{
    Tensor x({2, 3});
    Tensor y({3, 2});

    x(0, 0) = 1;
    x(0, 1) = 2;
    x(0, 2) = 3;
    x(1, 0) = 4;
    x(1, 1) = 5;
    x(1, 2) = 6;

    y(0, 0) = 7;
    y(0, 1) = 8;
    y(1, 0) = 9;
    y(1, 1) = 10;
    y(2, 0) = 11;
    y(2, 1) = 12;

    Tensor z = x.matmul(y);

    assert(z(0, 0) == 58);
    assert(z(0, 1) == 64);
    assert(z(1, 0) == 139);
    assert(z(1, 1) == 154);
}

int main()
{
    test_add();
    test_relu();
    test_matmul();

    std::cout << "we good g\n";
}