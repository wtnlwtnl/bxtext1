import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Res.txt against the standard NetworkX Top-100 results."
    )
    parser.add_argument("--res", default="Res.txt", help="Our Top-100 result file.")
    parser.add_argument(
        "--standard",
        default="NetworkX标准版全量PageRank结果_按分数降序.csv",
        help="Standard NetworkX ranking CSV path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top entries to compare.",
    )
    return parser.parse_args()


def load_res(path):
    rows = []
    with open(path, encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid Res.txt format at line {line_no}: {line.rstrip()}")
            node, score = parts
            rows.append((int(node), float(score)))
    return rows


def load_standard(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append((int(row["NodeID"]), float(row["PageRank"])))
    return rows


def main():
    args = parse_args()
    res_rows = load_res(args.res)
    standard_rows = load_standard(args.standard)

    top_k = args.top_k
    if len(res_rows) < top_k:
        raise ValueError(f"{args.res} has only {len(res_rows)} rows, expected at least {top_k}.")
    if len(standard_rows) < top_k:
        raise ValueError(
            f"{args.standard} has only {len(standard_rows)} rows, expected at least {top_k}."
        )

    res_top = res_rows[:top_k]
    standard_top = standard_rows[:top_k]

    res_nodes = [node for node, _ in res_top]
    standard_nodes = [node for node, _ in standard_top]
    standard_score_map = dict(standard_top)

    abs_errors = [abs(score - standard_score_map[node]) for node, score in res_top]
    rel_errors = [
        abs(score - standard_score_map[node]) / max(abs(standard_score_map[node]), 1e-15)
        for node, score in res_top
    ]

    print(f"Top1 一致：{res_nodes[0] == standard_nodes[0]}")
    print(f"Top10 重合数：{len(set(res_nodes[:10]) & set(standard_nodes[:10]))}")
    print(f"Top100 重合数：{len(set(res_nodes) & set(standard_nodes))}")
    print(f"Top100 的顺序也完全一致：{res_nodes == standard_nodes}")
    print(f"最大绝对误差：{max(abs_errors):.12e}")
    print(f"平均绝对误差：{sum(abs_errors) / len(abs_errors):.12e}")
    print(f"最大相对误差：{max(rel_errors):.12e}")
    print(f"平均相对误差：{sum(rel_errors) / len(rel_errors):.12e}")
    print(f"比较范围：标准答案前 {top_k} 项 vs Res.txt 前 {top_k} 项")


if __name__ == "__main__":
    main()
