#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <string.h>
#include <CL/sycl.hpp>
#include "utils.h"

using namespace sycl;

#define ATOMIC atomic_ref<int, memory_order::relaxed, memory_scope::device, access::address_space::global_space>
#define for_neighbours(u) for (int j = g->I[u], v = g->E[j], w = g->W[j]; j < g->I[u + 1]; j++)

#define forall(NUM_THREADS) Q.submit([&](handler &h){ h.parallel_for(NUM_THREADS, [=](id<1> u){for (; u < N; u += NUM_THREADS)
#define end \
    });     \
    }).wait();

class graph
{
private:
    int32_t num_nodes;
    int32_t num_edges;

public:
    int *V, *I, *E, *W, *RE, *RI;

    int get_num_nodes()
    {
        return num_nodes;
    }

    int get_num_edges()
    {
        return num_edges;
    }

    void set_num_nodes(int n)
    {
        num_nodes = n;
    }

    void set_num_edges(int n)
    {
        num_edges = n;
    }

    void load_graph(std::string name, queue Q)
    {
        std::vector<int> h_V, h_I, h_E, h_W, h_RE, h_RI;

        load_from_file("csr_graphs/" + name + "/V", h_V);
        load_from_file("csr_graphs/" + name + "/I", h_I);
        load_from_file("csr_graphs/" + name + "/E", h_E);
        load_from_file("csr_graphs/" + name + "/W", h_W);
        load_from_file("csr_graphs/" + name + "/RE", h_RE);
        load_from_file("csr_graphs/" + name + "/RI", h_RI);

        int num_nodes = h_V.size();
        int num_edges = h_E.size();

        set_num_edges(num_edges);
        set_num_nodes(num_nodes);
        V = malloc_device<int>(num_nodes, Q);
        I = malloc_device<int>((num_nodes + 1), Q);
        E = malloc_device<int>(num_edges, Q);
        W = malloc_device<int>(num_edges, Q);
        RE = malloc_device<int>(num_edges, Q);
        RI = malloc_device<int>((num_nodes + 1), Q);

        Q.submit([&](handler &h)
                 { h.memcpy(V, &h_V[0], h_V.size() * sizeof(int)); });

        Q.submit([&](handler &h)
                 { h.memcpy(I, &h_I[0], h_I.size() * sizeof(int)); });

        Q.submit([&](handler &h)
                 { h.memcpy(E, &h_E[0], h_E.size() * sizeof(int)); });

        Q.submit([&](handler &h)
                 { h.memcpy(W, &h_W[0], h_W.size() * sizeof(int)); });

        Q.submit([&](handler &h)
                 { h.memcpy(RE, &h_RE[0], h_RE.size() * sizeof(int)); });

        Q.submit([&](handler &h)
                 { h.memcpy(RI, &h_RI[0], h_RI.size() * sizeof(int)); });

        Q.wait();
    }

    void free_memory(queue Q)
    {
        free(V, Q);
        free(I, Q);
        free(E, Q);
        free(W, Q);
        free(RE, Q);
        free(RI, Q);
    }
};

// initialize device arr with val, if needed set arr[pos] = pos_val
template <typename T>
void initialize(T *arr, T val, int NUM_THREADS, int N, queue Q, int pos = -1, T pos_val = -1)
{
    int stride = NUM_THREADS;
    Q.submit([&](handler &h)
             { h.parallel_for(NUM_THREADS, [=](id<1> i)
                              {
                                  for (; i < N; i += stride)
                                  {
                                      arr[i] = val;
    
                                      if (i == pos)
                                      {
                                          arr[pos] = pos_val;
                                      }
                                  } }); });
    Q.wait();
}

// memcpy from src to dest
template <typename T>
void memcpy(T *dest, T *src, int N, queue Q)
{
    Q.submit([&](handler &h)
             { h.memcpy(dest, src, N * sizeof(T)); })
        .wait();
}

// TODO: returns True if u and v are neighbours
bool neighbours(int u, int v)
{
    return false;
}