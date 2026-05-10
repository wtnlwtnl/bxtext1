from collections import Counter


def main():
    path = "Data.txt"
    edges = []
    nodes = set()
    src_nodes = set()
    dst_nodes = set()
    out_degree = Counter()
    self_loops = 0

    with open(path, encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            src, dst = map(int, stripped.split())
            edges.append((src, dst))
            nodes.add(src)
            nodes.add(dst)
            src_nodes.add(src)
            dst_nodes.add(dst)
            out_degree[src] += 1
            if src == dst:
                self_loops += 1

    unique_edges = set(edges)
    dangling_nodes = sum(1 for node in nodes if out_degree[node] == 0)

    print(f"原始边数 {len(edges)}")
    print(f"去重后边数 {len(unique_edges)}")
    print(f"重复边数 {len(edges) - len(unique_edges)}")
    print(f"节点编号范围 {min(nodes)}--{max(nodes)}")
    print(f"实际有效节点数 {len(nodes)}")
    print(f"出现为源节点的节点数 {len(src_nodes)}")
    print(f"出现为目标节点的节点数 {len(dst_nodes)}")
    print(f"出度为 0 的节点数 {dangling_nodes}")
    print(f"自环边数 {self_loops}")


if __name__ == "__main__":
    main()
