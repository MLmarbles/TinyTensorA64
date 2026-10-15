#include "Tensor.hpp"

#include <chrono>
#include <iostream>

int main()
{
    const std::vector<std::size_t> sizes = {
        64,
        128,
        256,
        512
    };

    for (std::size_t size : sizes) {
        Tensor a({size, size});
        Tensor b({size, size});

        a.randomize();
        b.randomize();

        auto start = std::chrono::high_resolution_clock::now();

        Tensor result = a.matmul(b);

        auto end = std::chrono::high_resolution_clock::now();

        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Matrix size: " << size << "x" << size << '\n';
        std::cout << "Time: " << duration.count() << "\n";
    }
}