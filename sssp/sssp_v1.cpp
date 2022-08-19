#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#define DEBUG 1
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

int main()
{
    std::vector<int> V, I, E, W;
    load_from_file("input/simple/V", V);
    load_from_file("input/simple/I", I);
    load_from_file("input/simple/E", E);
    load_from_file("input/simple/W", W);

    if (DEBUG)
    {
        std::cout << "Node: ";
        print_vector(V);
        std::cout << "Offset: ";
        print_vector(I);
        std::cout << "Edge: ";
        print_vector(E);
        std::cout << "Weight: ";
        print_vector(W);
    }

    int N = V.size();
    std::vector<int> dist(N, INT_MAX);
    std::vector<int> dist_i, par;

    queue Q;
    std::cout << "Selected device: " << Q.get_device().get_info<info::device::name>() << "\n";

    buffer V_buf(V);
    buffer I_buf(I);
    buffer E_buf(E);
    buffer W_buf(W);
    buffer dist_buf(dist);
    buffer dist_i_buf(dist_i);
    buffer par_buf(par);

    for (int round = 1; round < N; round++)
    {
        Q.submit([&](handler &h)
                 {
                     accessor acc_V(V_buf, h, read_only);
                     accessor acc_I(I_buf, h, read_only);
                     accessor acc_E(E_buf, h, read_only);
                     accessor acc_W(W_buf, h, read_only);
                     accessor acc_dist(dist_buf, h, read_only);
                     accessor acc_dist_i(dist_i_buf, h, read_write);

                     h.parallel_for(
                         N, [=](id<1> i){
                            for(int j = acc_I[i]; j < acc_I[i + 1]; j++){
                                int w = acc_W[j];
                                int du = acc_dist[j];
                                int dv = acc_dist[acc_E[j]];
                            }
                         }); })
            .wait();
    }
    return 0;
}
