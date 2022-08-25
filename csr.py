from collections import defaultdict
import random

DEBUG = 0
HERE = 1

name = "USAud"
if HERE:
    filename = f"raw_graphs/{name}.txt"
else:
    filename = f"/lfs/usrhome/btech/ee19b106/graph_algorithms/raw_graphs/{name}.txt"

with open(filename) as f:
    l = f.readlines()
    
for i in range(len(l)):
    l[i] = list(map(int, l[i].strip().split(" ")))

g = defaultdict(list)

for [u, v] in l:
    w = random.randint(1, 100)
    if DEBUG:
        print(f"{u} {v} {w}")
    g[u].append([v, w])

V = sorted(list(g.keys()))
E = []
I = []
W = []
pos = 0
for v in V:
    I.append(pos)
    for [nn, nw] in g[v]:
        E.append(nn)
        W.append(nw)
        pos += 1
I.append(pos)

if DEBUG:
    print(V)
    print(I)
    print(E)
    print(W)

with open(f"csr_graphs/{name}/V", "w") as f:
    f.write(" ".join(list(map(str, V))))

with open(f"csr_graphs/{name}/I", "w") as f:
    f.write(" ".join(list(map(str, I))))

with open(f"csr_graphs/{name}/E", "w") as f:
    f.write(" ".join(list(map(str, E))))

with open(f"csr_graphs/{name}/W", "w") as f:
    f.write(" ".join(list(map(str, W))))

