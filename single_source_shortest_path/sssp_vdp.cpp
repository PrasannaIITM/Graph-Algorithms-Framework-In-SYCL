#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#define DEBUG 0

using namespace sycl;

int main(int argc, char **argv)
{

    std::chrono::steady_clock::time_point tic_0 = std::chrono::steady_clock::now();
    std::ofstream logfile;

    std::string name = argv[1];
    int source = atoi(argv[2]);
    int NUM_THREADS = atoi(argv[3]);
    std::string NUM_THREADS_STR = std::to_string(NUM_THREADS);

    logfile.open("single_source_shortest_path/output/" + name + "_sssp_dp_time_" + NUM_THREADS_STR + ".txt");

    logfile << "Processing " << name << std::endl;

    default_selector d_selector;
    queue Q(d_selector);
    logfile << "Selected device: " << Q.get_device().get_info<info::device::name>() << std::endl;
    logfile << "Number of parallel work items: " << NUM_THREADS << std::endl;

    std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
    std::vector<int> V, I, E, W;
    load_from_file("csr_graphs/" + name + "/V", V);
    load_from_file("csr_graphs/" + name + "/I", I);
    load_from_file("csr_graphs/" + name + "/E", E);
    load_from_file("csr_graphs/" + name + "/W", W);
    std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
    logfile << "Time to load data from files: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    int N = V.size();
    int stride = NUM_THREADS;

    std::vector<int> dist(N, INT_MAX);
    std::vector<int> dist_i(N, INT_MAX);

    dist[source] = 0;
    dist_i[source] = 0;

    {
        buffer<int> V_buf{V};
        buffer<int> I_buf{I};
        buffer<int> E_buf{E};
        buffer<int> W_buf{W};
        buffer<int> dist_buf{dist};
        buffer<int> dist_i_buf{dist_i};

        tic = std::chrono::steady_clock::now();
        logfile << "Starting SSSP..." << std::endl;
        int *active_count = malloc_shared<int>(1, Q);
        int *wl = malloc_shared<int>(N, Q);
        *active_count = 1;
        wl[0] = source;

        for (int round = 1; round < N; round++)
        {
            if (*active_count == 0)
            {
                logfile << "Number of rounds required for convergence: " << round << std::endl;
                break;
            }
            Q.submit([&](handler &h)
                     {
                accessor acc_V{V_buf, h, read_only};
                accessor acc_I{I_buf, h, read_only};
                accessor acc_E{E_buf, h, read_only};
                accessor acc_W{W_buf, h, read_only};
                accessor acc_dist{dist_buf, h, read_only};
                accessor acc_dist_i{dist_i_buf, h, read_write};

                h.parallel_for(
                     *active_count, [=](id<1> i)
                     {
                        int node = wl[i];
                        for (int j = acc_I[node]; j < acc_I[node + 1]; j++)
                        {
                            int w = acc_W[j];
                            int du = acc_dist[node];
                            int dv = acc_dist[acc_E[j]];
                            int new_dist = du + w;

                            if (du == INT_MAX)
                            {
                                continue;
                            }

                            atomic_ref<int, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(acc_dist_i[acc_E[j]]);
                            atomic_data.fetch_min(new_dist);
                        }
                            
                        }); })
                .wait();

            *active_count = 0;
            free(wl, Q);
            wl = malloc_shared<int>(N, Q);

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
                            atomic_ref<int, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(*active_count);
                            wl[atomic_data++] = i;
                        }
                        acc_dist_i[i] = acc_dist[i];
                    } }); })
                .wait();
        }
        toc = std::chrono::steady_clock::now();
        logfile << "Time to run SSSP: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;
    }
    tic = std::chrono::steady_clock::now();
    std::ofstream resultfile;
    int num_covered = 0;
    resultfile.open("single_source_shortest_path/output/" + name + "_sssp_dp_result_" + NUM_THREADS_STR + ".txt");

    for (int i = 0; i < N; i++)
    {
        if (dist[i] != INT_MAX)
        {
            num_covered += 1;
        }
        resultfile << i << " " << dist[i] << std::endl;
    }
    resultfile.close();
    toc = std::chrono::steady_clock::now();
    logfile << "Time to write data to file: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    std::chrono::steady_clock::time_point toc_0 = std::chrono::steady_clock::now();
    logfile << "Total time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(toc_0 - tic_0).count() << "[µs]" << std::endl;
    logfile << "Percentage coverage from given source: " << 100 * (1.0 * num_covered) / N << std::endl
            << "Number of nodes covered: " << num_covered << std::endl;

    return 0;
}
