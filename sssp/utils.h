#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>

using namespace sycl;

void load_from_file(const char *filename, std::vector<int> &vec)
{
    std::ifstream input;
    input.open(filename);
    int num;
    while ((input >> num) && input.ignore())
    {
        vec.push_back(num);
    }
    input.close();
}

void print_vector(std::vector<int> vec)
{
    for (auto x : vec)
        std::cout << x << " ";
    std::cout << std::endl;
}