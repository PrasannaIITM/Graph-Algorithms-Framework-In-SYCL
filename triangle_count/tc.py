import networkx as nx

G = nx.Graph()
one_way_edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2, 5), (1, 3), (3, 4)]
edges = []
for (u, v) in one_way_edges:
    edges.append((u, v))
    edges.append((v, u))
G.add_edges_from(edges)
tc = nx.triangles(G)
print(tc)