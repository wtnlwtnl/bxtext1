#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// 直接修改这个参数即可切换线程数，修改后重新编译再运行。
constexpr int OPENMP_THREADS = 4;

struct Args {
    std::string input = "Data.txt";
    std::string output = "Res.txt";
    double alpha = 0.85;
    double tol = 1e-10;
    int max_iter = 200;
    uint32_t block_size = 1024;
};

struct GraphData {
    std::vector<int32_t> raw_node_ids;
    std::vector<uint16_t> src_idx;
    std::vector<uint16_t> dst_idx;
    std::vector<uint32_t> out_degree;
    uint32_t num_nodes = 0;
    uint32_t num_edges = 0;
};

struct BlockGraph {
    std::vector<uint16_t> src_sorted;
    std::vector<uint16_t> dst_local;
    std::vector<uint32_t> block_bases;
    std::vector<uint32_t> block_lengths;
    std::vector<uint32_t> block_starts;
    std::vector<uint32_t> block_ends;
};

struct PageRankResult {
    std::vector<double> rank;
    int iterations = 0;
    double final_error = 0.0;
    bool converged = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for argument: " + key);
        }
        std::string value = argv[++i];
        if (key == "--input") {
            args.input = value;
        } else if (key == "--output") {
            args.output = value;
        } else if (key == "--alpha") {
            args.alpha = std::stod(value);
        } else if (key == "--tol") {
            args.tol = std::stod(value);
        } else if (key == "--max-iter") {
            args.max_iter = std::stoi(value);
        } else if (key == "--block-size") {
            args.block_size = static_cast<uint32_t>(std::stoul(value));
        } else {
            throw std::invalid_argument("Unknown argument: " + key);
        }
    }
    return args;
}

GraphData load_graph(const std::string& input_path) {
    std::ifstream input(input_path);
    if (!input) {
        throw std::runtime_error("Failed to open input file: " + input_path);
    }

    std::vector<std::pair<int32_t, int32_t>> raw_edges;
    std::vector<int32_t> node_ids;
    int32_t src = 0;
    int32_t dst = 0;
    while (input >> src >> dst) {
        raw_edges.emplace_back(src, dst);
        node_ids.push_back(src);
        node_ids.push_back(dst);
    }
    if (raw_edges.empty()) {
        throw std::runtime_error("Input graph is empty.");
    }

    std::sort(node_ids.begin(), node_ids.end());
    node_ids.erase(std::unique(node_ids.begin(), node_ids.end()), node_ids.end());
    if (node_ids.size() > static_cast<size_t>(UINT16_MAX)) {
        throw std::runtime_error("uint16_t node index is not enough for this graph.");
    }

    std::unordered_map<int32_t, uint16_t> id_to_idx;
    id_to_idx.reserve(node_ids.size() * 2);
    for (uint32_t i = 0; i < node_ids.size(); ++i) {
        id_to_idx.emplace(node_ids[i], static_cast<uint16_t>(i));
    }

    GraphData graph;
    graph.raw_node_ids = std::move(node_ids);
    graph.num_nodes = static_cast<uint32_t>(graph.raw_node_ids.size());
    graph.num_edges = static_cast<uint32_t>(raw_edges.size());
    graph.src_idx.resize(graph.num_edges);
    graph.dst_idx.resize(graph.num_edges);
    graph.out_degree.assign(graph.num_nodes, 0);

    for (uint32_t i = 0; i < graph.num_edges; ++i) {
        const uint16_t s = id_to_idx.at(raw_edges[i].first);
        const uint16_t d = id_to_idx.at(raw_edges[i].second);
        graph.src_idx[i] = s;
        graph.dst_idx[i] = d;
        ++graph.out_degree[s];
    }

    return graph;
}

BlockGraph build_block_graph(const GraphData& graph, uint32_t block_size) {
    if (block_size == 0) {
        throw std::invalid_argument("block_size must be positive.");
    }

    std::vector<uint32_t> order(graph.num_edges);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return graph.dst_idx[a] < graph.dst_idx[b];
    });

    BlockGraph block_graph;
    block_graph.src_sorted.resize(graph.num_edges);
    block_graph.dst_local.resize(graph.num_edges);

    uint32_t previous_block = UINT32_MAX;
    for (uint32_t sorted_pos = 0; sorted_pos < graph.num_edges; ++sorted_pos) {
        const uint32_t edge_id = order[sorted_pos];
        const uint16_t dst = graph.dst_idx[edge_id];
        const uint32_t block_id = dst / block_size;
        const uint32_t block_base = block_id * block_size;

        if (block_id != previous_block) {
            if (!block_graph.block_starts.empty()) {
                block_graph.block_ends.push_back(sorted_pos);
            }
            block_graph.block_starts.push_back(sorted_pos);
            block_graph.block_bases.push_back(block_base);
            block_graph.block_lengths.push_back(
                std::min(block_size, graph.num_nodes - block_base)
            );
            previous_block = block_id;
        }

        block_graph.src_sorted[sorted_pos] = graph.src_idx[edge_id];
        block_graph.dst_local[sorted_pos] = static_cast<uint16_t>(dst - block_base);
    }
    block_graph.block_ends.push_back(graph.num_edges);

    return block_graph;
}

PageRankResult pagerank(
    const GraphData& graph,
    const BlockGraph& block_graph,
    double alpha,
    double tol,
    int max_iter
) {
    if (!(alpha > 0.0 && alpha < 1.0)) {
        throw std::invalid_argument("alpha must be in the open interval (0, 1).");
    }
    if (tol <= 0.0) {
        throw std::invalid_argument("tol must be positive.");
    }
    if (max_iter <= 0) {
        throw std::invalid_argument("max_iter must be positive.");
    }

    const uint32_t n = graph.num_nodes;
    std::vector<double> rank(n, 1.0 / n);
    std::vector<double> next_rank(n, 0.0);
    std::vector<double> contrib(n, 0.0);
    std::vector<uint16_t> dangling_nodes;
    dangling_nodes.reserve(n);

    for (uint32_t i = 0; i < n; ++i) {
        if (graph.out_degree[i] == 0) {
            dangling_nodes.push_back(static_cast<uint16_t>(i));
        }
    }

    PageRankResult result;
    for (int iter = 1; iter <= max_iter; ++iter) {
        double dangling_mass = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : dangling_mass) schedule(static)
#endif
        for (int i = 0; i < static_cast<int>(dangling_nodes.size()); ++i) {
            dangling_mass += rank[dangling_nodes[static_cast<size_t>(i)]];
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < static_cast<int>(n); ++i) {
            if (graph.out_degree[static_cast<size_t>(i)] == 0) {
                contrib[static_cast<size_t>(i)] = 0.0;
            } else {
                contrib[static_cast<size_t>(i)] =
                    alpha * rank[static_cast<size_t>(i)] /
                    static_cast<double>(graph.out_degree[static_cast<size_t>(i)]);
            }
        }

        const double base_rank = (1.0 - alpha) / n + alpha * dangling_mass / n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < static_cast<int>(n); ++i) {
            next_rank[static_cast<size_t>(i)] = base_rank;
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int block = 0; block < static_cast<int>(block_graph.block_starts.size()); ++block) {
            const uint32_t start = block_graph.block_starts[static_cast<size_t>(block)];
            const uint32_t end = block_graph.block_ends[static_cast<size_t>(block)];
            const uint32_t base = block_graph.block_bases[static_cast<size_t>(block)];
            for (uint32_t pos = start; pos < end; ++pos) {
                next_rank[base + block_graph.dst_local[pos]] +=
                    contrib[block_graph.src_sorted[pos]];
            }
        }

        double error = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : error) schedule(static)
#endif
        for (int i = 0; i < static_cast<int>(n); ++i) {
            error += std::abs(next_rank[static_cast<size_t>(i)] - rank[static_cast<size_t>(i)]);
        }
        rank.swap(next_rank);

        result.iterations = iter;
        result.final_error = error;
        if (error < tol) {
            result.converged = true;
            break;
        }
    }

    result.rank = std::move(rank);
    return result;
}

void write_top100(
    const std::string& output_path,
    const std::vector<int32_t>& raw_node_ids,
    const std::vector<double>& rank
) {
    std::vector<uint32_t> indices(rank.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
        if (rank[a] != rank[b]) {
            return rank[a] > rank[b];
        }
        return raw_node_ids[a] < raw_node_ids[b];
    });

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }
    output << std::fixed << std::setprecision(12);
    const uint32_t top_k = std::min<uint32_t>(100, indices.size());
    for (uint32_t i = 0; i < top_k; ++i) {
        const uint32_t idx = indices[i];
        output << raw_node_ids[idx] << ' ' << rank[idx] << '\n';
    }
}

void print_stats(
    const GraphData& graph,
    const Args& args,
    const PageRankResult& result,
    double elapsed_seconds
) {
    uint32_t dangling_nodes = 0;
    for (uint32_t degree : graph.out_degree) {
        if (degree == 0) {
            ++dangling_nodes;
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "nodes=" << graph.num_nodes << '\n';
    std::cout << "edges=" << graph.num_edges << '\n';
    std::cout << "dangling_nodes=" << dangling_nodes << '\n';
    std::cout << "params=alpha:" << args.alpha
              << ", tol:" << std::scientific << std::setprecision(12) << args.tol
              << std::fixed << std::setprecision(6)
              << ", max_iter:" << args.max_iter
              << ", block_size:" << args.block_size << '\n';
    std::cout << "openmp_threads=" << OPENMP_THREADS << '\n';
    std::cout << "iterations=" << result.iterations << '\n';
    std::cout << "converged=" << (result.converged ? "True" : "False") << '\n';
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "final_l1_error=" << result.final_error << '\n';
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "elapsed_seconds=" << elapsed_seconds << '\n';
}

int main(int argc, char** argv) {
    try {
#ifdef _OPENMP
        omp_set_num_threads(OPENMP_THREADS);
#endif
        const Args args = parse_args(argc, argv);
        const auto start_time = std::chrono::steady_clock::now();

        const GraphData graph = load_graph(args.input);
        const BlockGraph block_graph = build_block_graph(graph, args.block_size);
        const PageRankResult result = pagerank(
            graph,
            block_graph,
            args.alpha,
            args.tol,
            args.max_iter
        );
        write_top100(args.output, graph.raw_node_ids, result.rank);

        const auto end_time = std::chrono::steady_clock::now();
        const double elapsed_seconds =
            std::chrono::duration<double>(end_time - start_time).count();

        print_stats(graph, args, result, elapsed_seconds);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
