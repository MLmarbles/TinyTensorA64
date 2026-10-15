#include "Tensor.hpp"

#include <cassert>
#include <iostream>
#include <Linear.hpp>
#include <Model.hpp>

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

void test_linear()
{
    Linear x({2, 2});
    x.weights()(0,0) = 1;
    x.weights()(0,1) = 1;
    x.weights()(1,0) = 1;
    x.weights()(1,1) = 1;

    Tensor y({2, 2});
    y(0,0) = 6;
    y(0,1) = 7;
    y(1,0) = 6;
    y(1,1) = 7;

    Tensor z = x.forward(y);
    assert(z(0, 0) == 6);
    assert(z(0, 1) == 7);
    assert(z(1, 0) == 6);
    assert(z(1, 1) == 7);
}

void test_model()
{
    Model x;

    x.layer1().weights().fill(0);
    x.layer1().bias().fill(0);
    x.layer2().weights().fill(0);
    x.layer2().bias().fill(0);

    x.layer1().weights()(0, 0) = 6;
    x.layer1().weights()(1, 1) = -2;
    x.layer1().weights()(2, 2) = 3;
    x.layer1().bias()(0, 0) = 1;
    x.layer1().bias()(0, 1) = 1;
    x.layer1().bias()(0, 3) = 2;

    x.layer2().weights()(0, 0) = 1;
    x.layer2().weights()(0, 1) = -1;
    x.layer2().weights()(1, 0) = 10;
    x.layer2().weights()(1, 1) = 10;
    x.layer2().weights()(2, 0) = 2;
    x.layer2().weights()(2, 1) = 1;
    x.layer2().weights()(3, 0) = -1;
    x.layer2().weights()(3, 1) = 2;
    x.layer2().bias()(0, 0) = 3;
    x.layer2().bias()(0, 1) = -2;

    Tensor input({1, 3});
    input(0, 0) = 1;
    input(0, 1) = 2;
    input(0, 2) = 3;

    Tensor output = x.forward(input);

    assert(output(0, 0) == 26);
    assert(output(0, 1) == 4);
}

void test_linear_invalid_shape()
{
    Linear layer(3, 4);
    Tensor input({1, 2});

    bool threw = false;

    try {
        layer.forward(input);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

int main()
{
    test_add();
    test_relu();
    test_matmul();
    test_model();

    std::cout << "we good g\n";
}
