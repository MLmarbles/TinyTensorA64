#pragma once

#include "Linear.hpp"

class Model {
    private:
        Linear layer1_;
        Linear layer2_;

    public:
        Model();

        Tensor forward(const Tensor& input) const;

        Linear& layer1();
        Linear& layer2();
};