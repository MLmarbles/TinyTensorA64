#include "Tensor.hpp"

#include <chrono>
#include <iostream>

int main()
{
    const std::vector<std::size_t> sizes = {
        64,
        128,
        256,
        512,
        1024
    };

    int runs = 5;

    for (std::size_t size : sizes) {
        long long total = 0;
        for (int run = 0; run < runs; ++run) {
            Tensor a({size, size});
            Tensor b({size, size});

            a.randomize();
            b.randomize();

            auto start = std::chrono::high_resolution_clock::now();

            Tensor result = a.matmul(b);

            auto end = std::chrono::high_resolution_clock::now();

            total += std::chrono::duration_cast<std::chrono::microseconds>((end - start)).count();

        }
        std::cout << "Default---> Matrix size: " << size << "x" << size << '\n';
        std::cout << "Time: " << total/runs << " average \n";
    }

    for (std::size_t size : sizes) {
        long long total = 0;
        for (int run = 0; run < runs; ++run) {
            Tensor a({size, size});
            Tensor b({size, size});

            a.randomize();
            b.randomize();

            auto start = std::chrono::high_resolution_clock::now();

            Tensor result = a.matmul_neon(b);

            auto end = std::chrono::high_resolution_clock::now();

            total += std::chrono::duration_cast<std::chrono::microseconds>((end - start)).count();

        }
        std::cout << "Neon---> Matrix size: " << size << "x" << size << '\n';
        std::cout << "Time: " << total/runs << " average \n";

    }
}