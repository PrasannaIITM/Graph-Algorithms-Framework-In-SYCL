#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#define DEBUG 1
#define NUM_THREADS 65536
using namespace sycl;

int main()
{

    std::chrono::steady_clock::time_point tic_0 = std::chrono::steady_clock::now();
    std::string name = "simple";
    // The default device selector will select the most performant device.
    default_selector d_selector;
    queue Q(d_selector);
    std::cout << "Selected device: " << Q.get_device().get_info<info::device::name>() << "\n";

    std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
    std::vector<int> V, I, E, W;
    load_from_file("input/" + name + "/V", V);
    load_from_file("input/" + name + "/I", I);
    load_from_file("input/" + name + "/E", E);
    load_from_file("input/" + name + "/W", W);
    std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
    std::cout << "Time to load data from files: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

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

    int N = V.size(), stride = NUM_THREADS;
    std::vector<int> dist(N, INT_MAX);
    std::vector<int> dist_i(N, INT_MAX);

    dist[0] = 0;
    dist_i[0] = 0;

    {
        buffer<int> V_buf{V};
        buffer<int> I_buf{I};
        buffer<int> E_buf{E};
        buffer<int> W_buf{W};
        buffer<int> dist_buf{dist};
        buffer<int> dist_i_buf{dist_i};

        int *wlin = malloc_shared<int>(N, Q);
        int *wlout = malloc_shared<int>(N, Q);

        tic = std::chrono::steady_clock::now();
        int *early_stop = malloc_shared<int>(1, Q);
        for (int round = 1; round < N; round++)
        {
            std::cout << "Round num: " << round << std::endl;
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
                accessor acc_wlin{wlin_buf, h, read_write};

                h.parallel_for(
                     acc_wlin.size(), [=](id<1> i)
                     {
                        int node = acc_wlin[i];
                        for (int j = acc_I[node]; j < acc_I[node + 1]; j++)
                        {
                            int neigh = acc_E[j];
                            int w = acc_W[j];
                            int du = acc_dist[node];
                            int dv = acc_dist[neigh];
                            int new_dist = du + w;

                            if (du == INT_MAX)
                            {
                                continue;
                            }

                            atomic_ref<int, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(acc_dist_i[neigh]);
                            atomic_data.fetch_min(new_dist);
                            
                        }
                        }); })
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
                                    wlout[i];
                                    *early_stop = 0;
                                }
                                acc_dist_i[i] = acc_dist[i];
                    } }); })
                .wait();

            host_accessor acc_wlin(wlin_buf);
            acc_wlin = wlout;
        }

        toc = std::chrono::steady_clock::now();
        std::cout << "Time to run SSSP: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;
    }
    tic = std::chrono::steady_clock::now();
    std::ofstream myfile;

    std::string NUM_THREADS_STR = std::to_string(NUM_THREADS);
    myfile.open("output/" + name + "/sssp_v301_result_" + NUM_THREADS_STR + ".txt");

    for (int i = 0; i < N; i++)
    {
        myfile << i << " " << dist[i] << std::endl;
    }
    myfile.close();
    toc = std::chrono::steady_clock::now();
    std::cout << "Time to write data to file: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    std::chrono::steady_clock::time_point toc_0 = std::chrono::steady_clock::now();
    std::cout << "Total time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(toc_0 - tic_0).count() << "[µs]" << std::endl;

    return 0;
}
