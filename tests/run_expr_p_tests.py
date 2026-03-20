#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def extract_last_meaningful_line(output: str, command: str) -> str:
    cleaned = strip_ansi(output).replace("\r", "")
    cleaned = cleaned.replace('(nemu) ', '')
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""

    filtered = []
    for line in lines:
        if line.startswith("Welcome to NEMU!"):
            continue
        if line.startswith("For help, type"):
            continue
        if line.startswith("[src/monitor/monitor.c"):
            continue
        if line.startswith("[src/monitor/debug/expr.c"):
            continue
        if line == command:
            continue
        if line == "q":
            continue
        filtered.append(line)

    if filtered:
        return filtered[-1]
    return lines[-1]


def maybe_build_nemu(nemu_dir: Path, binary: Path) -> None:
    if binary.exists():
        return
    print("NEMU binary not found, building it first...", file=sys.stderr)
    subprocess.run(["make"], cwd=nemu_dir, check=True)


def normalize_path(path_str: str, base: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def run_case(binary: Path, command: str) -> str:
    log_path = binary.parent / "nemu-test-log.txt"
    input_text = "{}\nq\n".format(command)
    proc = subprocess.run(
        [str(binary), "-l", str(log_path)],
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
        cwd=str(binary.parent.parent),
    )
    return extract_last_meaningful_line(proc.stdout + proc.stderr, command)


def judge(mode: str, expected: str, actual: str) -> bool:
    if mode == "exact":
        return actual == expected
    if mode == "prefix":
        return actual.startswith(expected)
    raise ValueError("Unknown mode: {}".format(mode))


def main() -> int:
    script_path = Path(__file__).resolve()
    nemu_dir = script_path.parent.parent
    default_cases = script_path.parent / "expr_p_cases.tsv"
    default_output = script_path.parent / "expr_p_results.txt"
    binary = nemu_dir / "build" / "nemu"

    parser = argparse.ArgumentParser(description="Run NEMU p-command expression tests.")
    parser.add_argument("--cases", default=str(default_cases), help="Path to the test case TSV file.")
    parser.add_argument("--output", default=str(default_output), help="Path to the result output file.")
    parser.add_argument("--binary", default=str(binary), help="Path to the NEMU binary.")
    args = parser.parse_args()

    cases_path = normalize_path(args.cases, Path.cwd())
    output_path = normalize_path(args.output, Path.cwd())
    binary_path = normalize_path(args.binary, Path.cwd())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    maybe_build_nemu(nemu_dir, binary_path)

    rows = []
    with cases_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            rows.append(row)

    passed = 0
    failed = 0
    output_lines = []
    output_lines.append("NEMU expression test results")
    output_lines.append("Cases file: {}".format(cases_path))
    output_lines.append("Binary: {}".format(binary_path))
    output_lines.append("")

    for row in rows:
        case_id, command, mode, expected, description = row
        actual = run_case(binary_path, command)
        ok = judge(mode, expected, actual)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        output_lines.append("[{}] {} {}".format(status, case_id, command))
        output_lines.append("Description: {}".format(description))
        output_lines.append("Expected ({}): {}".format(mode, expected))
        output_lines.append("Actual: {}".format(actual))
        output_lines.append("")

    output_lines.append("Total: {}".format(len(rows)))
    output_lines.append("Passed: {}".format(passed))
    output_lines.append("Failed: {}".format(failed))

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    print("Finished running {} cases.".format(len(rows)))
    print("Passed: {}, Failed: {}".format(passed, failed))
    print("Detailed report written to: {}".format(output_path))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
