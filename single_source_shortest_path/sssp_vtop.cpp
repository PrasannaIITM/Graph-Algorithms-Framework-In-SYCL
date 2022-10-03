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

    logfile.open("single_source_shortest_path/output/" + name + "_sssp_top_time_" + NUM_THREADS_STR + ".txt");

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

    int *dev_V = malloc_device<int>(V.size(), Q);
    int *dev_I = malloc_device<int>(I.size(), Q);
    int *dev_E = malloc_device<int>(E.size(), Q);
    int *dev_W = malloc_device<int>(W.size(), Q);
    Q.submit([&](handler &h)
             { h.memcpy(dev_V, &V[0], V.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_I, &I[0], I.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_E, &E[0], E.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_W, &W[0], W.size() * sizeof(int)); });

    Q.wait();

    int N = V.size();
    int stride = NUM_THREADS;
    std::vector<int> dist(N, INT_MAX);

    int *dev_flag = malloc_device<int>(N, Q);
    int *dev_dist = malloc_device<int>(N, Q);
    int *dev_dist_i = malloc_device<int>(N, Q);

    Q.submit([&](handler &h)
             { h.parallel_for(NUM_THREADS, [=](id<1> i)
                              {
                                  for (; i < N; i += stride)
                                  {
                                      dev_flag[i] = 0;
                                      dev_dist[i] = INT_MAX;
                                      dev_dist_i[i] = INT_MAX;

                                      if (i == source)
                                      {
                                          dev_flag[source] = 1;
                                          dev_dist[source] = 0;
                                          dev_dist_i[source] = 0;
                                      }
                                  } }); });
    Q.wait();

    tic = std::chrono::steady_clock::now();
    logfile << "Starting SSSP..." << std::endl;
    int *early_stop = malloc_shared<int>(1, Q);
    for (int round = 1; round < N; round++)
    {

        if (*early_stop == 1)
        {
            logfile << "Number of rounds required for convergence: " << round << std::endl;
            break;
        }
        *early_stop = 1;

        Q.submit([&](handler &h)
                 { h.parallel_for(
                       NUM_THREADS, [=](id<1> i)
                       {
                           
                            for(; i < N; i+= stride){
                                if (dev_flag[i])
                                {
                                    dev_flag[i] = 0;
                                    for (int j = dev_I[i]; j < dev_I[i + 1]; j++)
                                    {
                                        int w = dev_W[j];
                                        int du = dev_dist[i];
                                        int dv = dev_dist[dev_E[j]];
                                        int new_dist = du + w;

                                        if (du == INT_MAX)
                                        {
                                            continue;
                                        }
                                        atomic_ref<int, memory_order::relaxed, memory_scope::device, access::address_space::ext_intel_global_device_space> atomic_data(dev_dist_i[dev_E[j]]);
                                        atomic_data.fetch_min(new_dist);
                                    }
                                }
                        } }); })
            .wait();

        Q.submit([&](handler &h)
                 { h.parallel_for(
                       NUM_THREADS, [=](id<1> i)
                       {

                    for (; i < N; i += stride)
                    {
                        if (dev_dist[i] > dev_dist_i[i])
                        {
                            dev_dist[i] = dev_dist_i[i];
                            dev_flag[i] = 1;
                            *early_stop = 0;
                        }
                        dev_dist_i[i] = dev_dist[i];
                    } }); })
            .wait();
    }
    toc = std::chrono::steady_clock::now();
    logfile << "Time to run SSSP: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    tic = std::chrono::steady_clock::now();
    std::ofstream resultfile;
    int num_covered = 0;

    Q.submit([&](handler &h)
             { h.memcpy(&dist[0], dev_dist, N * sizeof(int)); })
        .wait();

    resultfile.open("single_source_shortest_path/output/" + name + "_sssp_top_result_" + NUM_THREADS_STR + ".txt");

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
