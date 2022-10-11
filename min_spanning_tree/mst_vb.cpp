#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#define DEBUG 1

using namespace sycl;

int main(int argc, char **argv)
{

    std::chrono::steady_clock::time_point tic_0 = std::chrono::steady_clock::now();
    std::ofstream logfile;

    std::string name = argv[1];
    int NUM_THREADS = atoi(argv[2]);
    std::string NUM_THREADS_STR = std::to_string(NUM_THREADS);

    logfile.open("min_spanning_tree/output/" + name + "_mst_vb_time_" + NUM_THREADS_STR + ".txt");

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

    int n_vertices = V.size();
    int n_edges = E.size();
    int stride = NUM_THREADS;

    int *d_V = malloc_device<int>(V.size(), Q);
    int *d_I = malloc_device<int>(I.size(), Q);
    int *d_E = malloc_device<int>(E.size(), Q);
    int *d_W = malloc_device<int>(W.size(), Q);

    int *d_parent = malloc_device<int>(n_vertices, Q);
    int *d_local_min_edge = malloc_device<int>(n_vertices, Q);
    int *d_comp_min_edge = malloc_device<int>(n_vertices, Q);
    int *d_edge_in_mst = malloc_device<int>(n_edges, Q);

    bool *single_comp = malloc_shared<bool>(1, Q);
    bool *found_min_edge = malloc_shared<bool>(1, Q);
    bool *parents_updated = malloc_shared<bool>(1, Q);

    Q.submit([&](handler &h)
             { h.memcpy(d_V, &V[0], V.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(d_I, &I[0], I.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(d_E, &E[0], E.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(d_W, &W[0], W.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.parallel_for(
                   NUM_THREADS, [=](id<1> i)
                   {
                               for (; i < n_edges; i += stride)
                               {
                                    d_edge_in_mst[i] = 0;
                               } }); })
        .wait();

    Q.submit([&](handler &h)
             { h.parallel_for(
                   NUM_THREADS, [=](id<1> i)
                   {
                               for (; i < n_vertices ;i += stride)
                               {
                                    d_parent[i] = i;
                               } }); })
        .wait();

    tic = std::chrono::steady_clock::now();
    logfile << "Starting minimum spanning tree calculation..." << std::endl;

    *single_comp = false;

    while (!*single_comp)
    {
        *single_comp = true;
        Q.submit([&](handler &h)
                 { h.parallel_for(
                       NUM_THREADS, [=](id<1> i)
                       {
                               for (; i < n_vertices ;i += stride)
                               {
                                    d_local_min_edge[i] = -1;
                                    d_comp_min_edge[i] = -1;
                               } }); })
            .wait();

        // find minimum edge to different component from each node
        Q.submit([&](handler &h)
                 { h.parallel_for(
                       NUM_THREADS, [=](id<1> u)
                       {
                               for (; u < n_vertices; u += stride)
                               {
                                   for(int edge = d_I[u]; edge < d_I[u + 1]; edge++){
                                        int v = d_E[edge];
                                        // if u and v in different components
                                        if(d_parent[u] != d_parent[v]){
                                            int curr_min_edge = d_local_min_edge[u];
                                            if (curr_min_edge == -1)
                                            { 
                                                d_local_min_edge[u] = edge;
                                            } 
                                            else
                                            { 
                                                int curr_neigh = d_E[curr_min_edge];
                                                if (d_W[edge] < d_W[curr_min_edge] || (d_W[edge] == d_W[curr_min_edge] && d_parent[v] < d_parent[curr_neigh]))
                                                { 
                                                    d_local_min_edge[u] = edge;
                                                } 
                                            } 
                                        }
                                   }
                               } }); })
            .wait();

        // find the minimum edge from the component
        *found_min_edge = false;
        while (!*found_min_edge)
        {
            *found_min_edge = true;
            Q.submit([&](handler &h)
                     { h.parallel_for(
                           NUM_THREADS, [=](id<1> u)
                           {
                               for (; u < n_vertices; u += stride)
                               {
                                    int my_comp = d_parent[u];

                                    int comp_min_edge = d_comp_min_edge[my_comp];

                                    int my_min_edge = d_local_min_edge[u];

                                    if (my_min_edge!= -1)
                                    {
                                        int v = d_E[my_min_edge];

                                        if (comp_min_edge == -1)
                                        {
                                            d_comp_min_edge[my_comp] = my_min_edge;
                                            *found_min_edge = false;
                                        }
                                        else
                                        {
                                            int curr_min_neigh = d_E[comp_min_edge];
                                            if (d_W[my_min_edge] < d_W[comp_min_edge] || (d_W[my_min_edge] == d_W[comp_min_edge] && d_parent[v] < d_parent[curr_min_neigh]))
                                            {
                                                d_comp_min_edge[my_comp] = my_min_edge;
                                                *found_min_edge = false;
                                            }

                                        }

                                    }
                               } }); })
                .wait();
        }

        // remove cycles of 2 nodes
        Q.submit([&](handler &h)
                 { h.parallel_for(
                       NUM_THREADS, [=](id<1> u)
                       {
                               for (; u < n_vertices; u += stride)
                               {
                                    // if u is the representative of its component
                                    if (d_parent[u] == d_V[u])
                                    { 
                                        int comp_min_edge = d_comp_min_edge[u];
                                        if (comp_min_edge != -1)
                                        {
                                            int v = d_E[comp_min_edge];
                                            int parent_v = d_parent[v];

                                            int v_comp_min_edge = d_comp_min_edge[parent_v];
                                            if (v_comp_min_edge != -1)
                                            {
                                                // v is comp(u)'s neighbour
                                                // w is comp(v)'s neighbout
                                                int w = d_E[v_comp_min_edge];
                                                if (d_parent[u] == d_parent[w] && d_parent[u] < d_parent[v])
                                                { 
                                                    d_comp_min_edge[parent_v] = -1;

                                                } 

                                            } 

                                        }
                                    }
                               } }); })
            .wait();

        // update the MST edges
        Q.submit([&](handler &h)
                 { h.parallel_for(
                       NUM_THREADS, [=](id<1> u)
                       {
                               for (; u < n_vertices; u += stride)
                               {
                                   // if u is the representative of its component
                                   if (d_parent[u] == d_V[u])
                                   {
                                       int curr_comp_min_edge = d_comp_min_edge[u];
                                       if (curr_comp_min_edge != -1)
                                       {
                                           d_edge_in_mst[curr_comp_min_edge] = 1;
                                       } 

                                    } 
                               } }); })
            .wait();

        // update parents
        Q.submit([&](handler &h)
                 { h.parallel_for(
                       NUM_THREADS, [=](id<1> u)
                       {
                               for (; u < n_vertices; u += stride)
                               {
                                   // if u is the representative of its component
                                   if (d_parent[u] == d_V[u])
                                   {
                                       int curr_comp_min_edge = d_comp_min_edge[u];
                                       if (curr_comp_min_edge != -1)
                                       {
                                           *single_comp = false;
                                           int v = d_E[curr_comp_min_edge];
                                           d_parent[u] = d_parent[v];
                                       } 

                                    } 
                               } }); })
            .wait();

        // flatten parents
        *parents_updated = false;
        while (!*parents_updated)
        {
            *parents_updated = true;
            Q.submit([&](handler &h)
                     { h.parallel_for(
                           NUM_THREADS, [=](id<1> u)
                           {
                               for (; u < n_vertices; u += stride)
                               {
                                    int parent_u = d_parent[u]; 
                                    int parent_parent_u = d_parent[parent_u];

                                    if (parent_u != parent_parent_u)
                                    { 
                                        *parents_updated = false;
                                        d_parent[u] = parent_parent_u;
                                    }
                               } }); })
                .wait();
        }
    }
    toc = std::chrono::steady_clock::now();
    logfile << "Time to run minimum spanning tree: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    tic = std::chrono::steady_clock::now();
    std::ofstream resultfile;

    resultfile.open("min_spanning_tree/output/" + name + "_mst_vb_result_" + NUM_THREADS_STR + ".txt");

    std::vector<int> op(n_edges);
    Q.submit([&](handler &h)
             { h.memcpy(&op[0], d_edge_in_mst, n_edges * sizeof(int)); })
        .wait();
    int count = 0;
    for (int i = 0; i < n_edges; i++)
    {
        if (op[i] == 1)
            count++;
        resultfile << i << " " << op[i] << std::endl;
    }
    resultfile << "Num edges included in MST: " << count << " Total nodes in graph: " << n_vertices << std::endl;
    resultfile.close();
    toc = std::chrono::steady_clock::now();
    logfile << "Time to write data to file: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    std::chrono::steady_clock::time_point toc_0 = std::chrono::steady_clock::now();
    logfile << "Total time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(toc_0 - tic_0).count() << "[µs]" << std::endl;
    return 0;
}
