import argparse
import csv
import re
import subprocess
from pathlib import Path

import networkx as nx


ALPHA_PATTERN = re.compile(r"double alpha = [0-9.]+;")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run teleport-parameter experiments for PageRank_block.cpp."
    )
    parser.add_argument("--source", default="PageRank_block.cpp", help="C++ source file.")
    parser.add_argument("--input", default="Data.txt", help="Input edge list path.")
    parser.add_argument(
        "--output-dir",
        default="不同Teleport参数输出",
        help="Directory for generated result files.",
    )
    parser.add_argument(
        "--alphas",
        default="0.50,0.70,0.80,0.85,0.90,0.95",
        help="Comma-separated alpha values.",
    )
    parser.add_argument("--compiler", default="g++", help="C++ compiler.")
    parser.add_argument(
        "--cxxflags",
        default="-O2 -std=c++17",
        help="C++ compile flags.",
    )
    parser.add_argument(
        "--exe",
        default="PageRank_block_teleport.exe",
        help="Compiled executable path.",
    )
    return parser.parse_args()


def load_graph(input_path):
    raw_edges = []
    nodes = set()
    with open(input_path, encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            src, dst = map(int, stripped.split())
            raw_edges.append((src, dst))
            nodes.add(src)
            nodes.add(dst)
    raw_nodes = sorted(nodes)
    return raw_nodes, raw_edges


def compute_networkx_pagerank(raw_nodes, raw_edges, alpha):
    graph = nx.DiGraph()
    graph.add_nodes_from(raw_nodes)
    graph.add_edges_from(raw_edges)
    uniform = {node: 1.0 / len(raw_nodes) for node in raw_nodes}
    return nx.pagerank(
        graph,
        alpha=alpha,
        personalization=uniform,
        dangling=uniform,
        max_iter=1000,
        tol=1e-14,
    )


def read_full_scores(path):
    scores = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            node, score = stripped.split()
            scores[int(node)] = float(score)
    return scores


def read_top100(path):
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            node, score = stripped.split()
            rows.append((int(node), float(score)))
    return rows


def parse_program_stdout(stdout):
    data = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def update_source_alpha(source_path, alpha):
    content = source_path.read_text(encoding="utf-8")
    updated, count = ALPHA_PATTERN.subn(f"double alpha = {alpha:.2f};", content, count=1)
    if count != 1:
        raise RuntimeError("Failed to locate default alpha definition in source file.")
    source_path.write_text(updated, encoding="utf-8")


def compile_source(source_path, exe_path, compiler, cxxflags):
    cmd = [compiler] + cxxflags.split() + [str(source_path), "-o", str(exe_path)]
    subprocess.run(cmd, check=True)


def rank_nodes(score_map):
    return sorted(score_map, key=lambda node: (-score_map[node], node))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    source_path = Path(args.source)
    input_path = Path(args.input)
    exe_path = Path(args.exe).resolve()
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]
    raw_nodes, raw_edges = load_graph(input_path)

    original_source = source_path.read_text(encoding="utf-8")
    summary_rows = []
    alpha_to_top100 = {}

    try:
        for alpha in alphas:
            update_source_alpha(source_path, alpha)
            compile_source(source_path, exe_path, args.compiler, args.cxxflags)

            top100_path = output_dir / f"alpha_{alpha:.2f}_Res.txt"
            full_path = output_dir / f"alpha_{alpha:.2f}_full.txt"
            cmd = [
                str(exe_path),
                "--input",
                str(input_path),
                "--output",
                str(top100_path),
                "--full-output",
                str(full_path),
            ]
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            stats = parse_program_stdout(completed.stdout)

            our_scores = read_full_scores(full_path)
            nx_scores = compute_networkx_pagerank(raw_nodes, raw_edges, alpha)
            ordered_nodes = sorted(raw_nodes)

            l1_sum = sum(abs(our_scores[node] - nx_scores[node]) for node in ordered_nodes)
            l1_avg = l1_sum / len(ordered_nodes)
            max_abs_error = max(abs(our_scores[node] - nx_scores[node]) for node in ordered_nodes)

            our_ranked = rank_nodes(our_scores)
            nx_ranked = rank_nodes(nx_scores)
            our_top100 = our_ranked[:100]
            nx_top100 = nx_ranked[:100]
            alpha_to_top100[f"{alpha:.2f}"] = set(our_top100)

            summary_rows.append(
                {
                    "alpha": f"{alpha:.2f}",
                    "Top1节点": our_top100[0],
                    "与alpha0.85的Top100重合数": "",
                    "与NetworkX的Top100重合数": len(set(our_top100) & set(nx_top100)),
                    "pagerank值与标准值L1距离平均值": f"{l1_avg:.18e}",
                    "L1距离总和": f"{l1_sum:.18e}",
                    "最大绝对误差": f"{max_abs_error:.18e}",
                    "迭代轮数": stats.get("iterations", ""),
                    "程序内部时间秒": stats.get("elapsed_seconds", ""),
                    "最终L1误差": stats.get("final_l1_error", ""),
                }
            )

            print(
                f"alpha={alpha:.2f} "
                f"avg_l1={l1_avg:.18e} "
                f"top100_overlap_vs_nx={len(set(our_top100) & set(nx_top100))}"
            )
    finally:
        source_path.write_text(original_source, encoding="utf-8")

    if "0.85" not in alpha_to_top100:
        raise RuntimeError("Alpha list must include 0.85 to compute overlap-with-0.85 statistics.")

    alpha_085_top100 = alpha_to_top100["0.85"]
    for row in summary_rows:
        current_top100 = alpha_to_top100[row["alpha"]]
        row["与alpha0.85的Top100重合数"] = len(current_top100 & alpha_085_top100)

    summary_path = output_dir / "Teleport参数L1对比结果.csv"
    write_csv(summary_path, summary_rows)
    print(f"wrote {summary_path}")
    print(f"restored source alpha in {source_path}")


if __name__ == "__main__":
    main()
