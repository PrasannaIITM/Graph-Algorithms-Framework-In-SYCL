#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#define DEBUG 0

using namespace sycl;

int main()
{
    std::chrono::steady_clock::time_point tic, toc;
    std::string name = "simple";
    // The default device selector will select the most performant device.
    default_selector d_selector;
    queue Q(d_selector);

    std::vector<int> V, I, E, W;
    load_from_file("input/" + name + "/V", V);
    load_from_file("input/" + name + "/I", I);
    load_from_file("input/" + name + "/E", E);
    load_from_file("input/" + name + "/W", W);
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
    std::vector<int> num_threads(11);
    int prev = 1024;
    num_threads[0] = prev;

    for (int i = 1; i < num_threads.size(); i++)
    {
        prev = prev * 2;
        num_threads[i] = prev;
    }

    for (auto NUM_THREADS : num_threads)
    {
        std::vector<int> dist(N, INT_MAX);
        std::vector<int> dist_i(N, INT_MAX), par(N);
        std::vector<int> flag(N, 0);

        flag[0] = 1;
        dist[0] = 0;
        dist_i[0] = 0;

        buffer<int> V_buf{V};
        buffer<int> I_buf{I};
        buffer<int> E_buf{E};
        buffer<int> W_buf{W};
        buffer<int> dist_buf{dist};
        buffer<int> dist_i_buf{dist_i};
        buffer<int> par_buf{par};
        buffer<int> flag_buf{flag};

        int stride = NUM_THREADS;
        tic = std::chrono::steady_clock::now();
        int *early_stop = malloc_shared<int>(1, Q);
        for (int round = 1; round < N; round++)
        {
            if (*early_stop == 1)
            {
                break;
            }
            *early_stop = 1;
            Q.submit([&](handler &h)
                     {
                 accessor acc_V{V_buf, h, read_only};
                 accessor acc_I{I_buf, h, read_only};
                 accessor acc_E{E_buf, h, read_only};
                 accessor acc_W{W_buf, h, read_only};
                 accessor acc_dist{dist_buf, h, read_only};
                 accessor acc_dist_i{dist_i_buf, h, read_write};
                 accessor acc_flag{flag_buf, h, read_write};

                 h.parallel_for(
                     NUM_THREADS, [=](id<1> i)
                     {
                        for(; i < N; i+= stride){
                            if (acc_flag[i])
                            {
                                acc_flag[i] = 0;
                                for (int j = acc_I[i]; j < acc_I[i + 1]; j++)
                                {
                                    int w = acc_W[j];
                                    int du = acc_dist[i];
                                    int dv = acc_dist[acc_E[j]];
                                    int new_dist = du + w;

                                    if (du == INT_MAX)
                                    {
                                        continue;
                                    }

                                    atomic_ref<int, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(acc_dist_i[acc_E[j]]);
                                    atomic_data.fetch_min(new_dist);
                                }
                            }
                        }
                        }); })
                .wait();

            Q.submit([&](handler &h)
                     {
                accessor acc_dist{dist_buf, h, read_write};
                accessor acc_dist_i{dist_i_buf, h, read_write};
                accessor acc_flag{flag_buf, h, read_write};

                h.parallel_for(
                    NUM_THREADS, [=](id<1> i)
                    {

                    for (; i < N; i += stride)
                    {
                        if (acc_dist[i] > acc_dist_i[i])
                        {
                            acc_dist[i] = acc_dist_i[i];
                            acc_flag[i] = 1;
                            *early_stop = 0;
                        }
                        acc_dist_i[i] = acc_dist[i];
                    } }); })
                .wait();
        }
        toc = std::chrono::steady_clock::now();
        std::cout << NUM_THREADS << "  " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;
    }

    return 0;
}
