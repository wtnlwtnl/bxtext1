import argparse
import csv
from pathlib import Path

import networkx as nx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate full NetworkX PageRank results from Data.txt."
    )
    parser.add_argument("--input", default="Data.txt", help="Input edge list path.")
    parser.add_argument(
        "--output",
        default="NetworkX标准版全量PageRank结果_按分数降序.csv",
        help="Output CSV path for the full ranking sorted by score.",
    )
    parser.add_argument("--alpha", type=float, default=0.85, help="Teleport parameter.")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-14,
        help="NetworkX convergence tolerance.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="NetworkX maximum iterations.",
    )
    return parser.parse_args()


def load_graph(input_path):
    raw_edges = []
    nodes = set()
    with open(input_path, encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid edge format at line {line_no}: {line.rstrip()}")
            src, dst = map(int, parts)
            raw_edges.append((src, dst))
            nodes.add(src)
            nodes.add(dst)

    if not raw_edges:
        raise ValueError("Input graph is empty.")

    raw_nodes = sorted(nodes)
    return raw_nodes, raw_edges


def compute_pagerank(raw_nodes, raw_edges, alpha, tol, max_iter):
    graph = nx.DiGraph()
    graph.add_nodes_from(raw_nodes)
    graph.add_edges_from(raw_edges)
    uniform = {node: 1.0 / len(raw_nodes) for node in raw_nodes}
    return nx.pagerank(
        graph,
        alpha=alpha,
        personalization=uniform,
        dangling=uniform,
        tol=tol,
        max_iter=max_iter,
    )


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    raw_nodes, raw_edges = load_graph(args.input)
    rank = compute_pagerank(
        raw_nodes=raw_nodes,
        raw_edges=raw_edges,
        alpha=args.alpha,
        tol=args.tol,
        max_iter=args.max_iter,
    )

    sorted_nodes = sorted(raw_nodes, key=lambda node: (-rank[node], node))
    by_rank_rows = [
        {"Rank": index, "NodeID": node, "PageRank": f"{rank[node]:.15f}"}
        for index, node in enumerate(sorted_nodes, start=1)
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, ["Rank", "NodeID", "PageRank"], by_rank_rows)

    print(f"wrote {output_path}")
    print(f"nodes={len(raw_nodes)}")
    print(f"edges={len(raw_edges)}")
    print(f"alpha={args.alpha}")


if __name__ == "__main__":
    main()
