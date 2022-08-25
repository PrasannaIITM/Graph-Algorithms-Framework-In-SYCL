#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#define DEBUG 1
#define NUM_THREADS 1024

using namespace sycl;

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
    int stride = NUM_THREADS;

    std::vector<int> dist(N, INT_MAX);
    std::vector<int> dist_i(N, INT_MAX), par(N);

    dist[0] = 0;
    dist_i[0] = 0;

    queue Q;
    std::cout << "Selected device: " << Q.get_device().get_info<info::device::name>() << "\n";

    {
        buffer V_buf{V};
        buffer I_buf{I};
        buffer E_buf{E};
        buffer W_buf{W};
        buffer dist_buf{dist};
        buffer dist_i_buf{dist_i};
        buffer par_buf{par};

        for (int round = 1; round < N; round++)
        {

            Q.submit([&](handler &h)
                     {
                 accessor acc_V{V_buf, h, read_only};
                 accessor acc_I{I_buf, h, read_only};
                 accessor acc_E{E_buf, h, read_only};
                 accessor acc_W{W_buf, h, read_only};
                 accessor acc_dist{dist_buf, h, read_only};
                 accessor acc_dist_i{dist_i_buf, h, read_write};

                 stream out(1024, 256, h);

                 h.parallel_for(
                     NUM_THREADS, [=](id<1> i)
                     {
                        for(; i < N; i+= stride){
                            for(int j = acc_I[i]; j < acc_I[i + 1]; j++){
                                int w = acc_W[j];
                                int du = acc_dist[i];
                                int dv = acc_dist[acc_E[j]];
                                int new_dist = du + w;

                                if(du == INT_MAX){
                                    continue;
                                }

                                atomic_ref<int, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(acc_dist_i[acc_E[j]]);
                                atomic_data.fetch_min(new_dist);
                                // out<< round << " " << i << " " << j << " " << new_dist << " " << acc_dist_i[acc_E[j]] << endl;
                            }
                        } }); })
                .wait();

            Q.submit([&](handler &h)
                     {
                accessor acc_dist{dist_buf, h, read_write};
                accessor acc_dist_i{dist_i_buf, h, read_write};

                h.parallel_for(
                    NUM_THREADS, [=](id<1> i)
                    {

                    for (; i < N; i += stride)
                    {
                        if (acc_dist[i] > acc_dist_i[i])
                        {
                            acc_dist[i] = acc_dist_i[i];
                        }
                        acc_dist_i[i] = acc_dist[i];
                    } }); })
                .wait();
        }
    }
    for (int i = 0; i < N; i++)
    {
        std::cout << i << " " << dist[i] << std::endl;
    }

    return 0;
}
