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

struct Args {
    std::string input = "Data.txt";
    std::string output = "Res.txt";
    double alpha = 0.85;
    double tol = 1e-10;
    int max_iter = 200;
};

struct GraphData {
    std::vector<int32_t> raw_node_ids;
    std::vector<uint32_t> in_starts;
    std::vector<uint16_t> in_sources;
    std::vector<uint32_t> out_degree;
    uint32_t num_nodes = 0;
    uint32_t num_edges = 0;
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
    graph.in_starts.assign(graph.num_nodes + 1, 0);
    graph.in_sources.resize(graph.num_edges);
    graph.out_degree.assign(graph.num_nodes, 0);

    std::vector<std::pair<uint16_t, uint16_t>> dense_edges;
    dense_edges.reserve(graph.num_edges);
    for (const auto& edge : raw_edges) {
        const uint16_t s = id_to_idx.at(edge.first);
        const uint16_t d = id_to_idx.at(edge.second);
        dense_edges.emplace_back(s, d);
        ++graph.in_starts[d + 1];
        ++graph.out_degree[s];
    }

    for (uint32_t i = 1; i <= graph.num_nodes; ++i) {
        graph.in_starts[i] += graph.in_starts[i - 1];
    }

    std::vector<uint32_t> cursor = graph.in_starts;
    for (const auto& edge : dense_edges) {
        graph.in_sources[cursor[edge.second]++] = edge.first;
    }

    return graph;
}

PageRankResult pagerank(const GraphData& graph, double alpha, double tol, int max_iter) {
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
        for (uint16_t node : dangling_nodes) {
            dangling_mass += rank[node];
        }

        for (uint32_t i = 0; i < n; ++i) {
            if (graph.out_degree[i] == 0) {
                contrib[i] = 0.0;
            } else {
                contrib[i] = alpha * rank[i] / static_cast<double>(graph.out_degree[i]);
            }
        }

        const double base_rank = (1.0 - alpha) / n + alpha * dangling_mass / n;
        for (uint32_t dst = 0; dst < n; ++dst) {
            double value = base_rank;
            for (uint32_t pos = graph.in_starts[dst]; pos < graph.in_starts[dst + 1]; ++pos) {
                value += contrib[graph.in_sources[pos]];
            }
            next_rank[dst] = value;
        }

        double error = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            error += std::abs(next_rank[i] - rank[i]);
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
              << ", max_iter:" << args.max_iter << '\n';
    std::cout << "iterations=" << result.iterations << '\n';
    std::cout << "converged=" << (result.converged ? "True" : "False") << '\n';
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "final_l1_error=" << result.final_error << '\n';
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "elapsed_seconds=" << elapsed_seconds << '\n';
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const auto start_time = std::chrono::steady_clock::now();

        const GraphData graph = load_graph(args.input);
        const PageRankResult result = pagerank(graph, args.alpha, args.tol, args.max_iter);
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
