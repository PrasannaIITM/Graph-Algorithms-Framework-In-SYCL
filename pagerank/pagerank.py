import networkx as nx

G = nx.DiGraph()
edges = [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 3), (3, 1), (3, 4), (4, 3)]
pr = nx.pagerank(G)
print(pr)