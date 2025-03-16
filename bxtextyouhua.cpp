#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>


using namespace std;
using namespace std::chrono;


// 定义宏，用于展开两次两元素累加操作
#define UNROLL2(i, a, sum1, sum2) \
    do {                         \
        sum1 += a[i];            \
        sum2 += a[i+1];          \
    } while (0)


// 原始多路链式算法：每次处理两个元素
int algorithm2(const vector<int>& a, int n) {
    int sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; i += 2) {
        sum1 += a[i];
        if (i + 1 < n) {
            sum2 += a[i + 1];
        }
    }
    return sum1 + sum2;
}


// 利用宏展开的多路链式算法：展开因子为4，每次循环处理4个元素（即两次两元素累加）
int algorithm2_unroll(const vector<int>& a, int n) {
    int sum1 = 0, sum2 = 0;
    int i = 0;
    const int unrollFactor = 4; // 每次循环处理4个元素
    // 主循环：保证至少 unrollFactor 个元素存在
    for (; i <= n - unrollFactor; i += unrollFactor) {
        UNROLL2(i, a, sum1, sum2);      // 处理 a[i] 和 a[i+1]
        UNROLL2(i + 2, a, sum1, sum2);   // 处理 a[i+2] 和 a[i+3]
    }
    // 处理剩余不足 unrollFactor 的元素
    for (; i < n; i += 2) {
        sum1 += a[i];
        if (i + 1 < n) {
            sum2 += a[i + 1];
        }
    }
    return sum1 + sum2;
}


int main() {
    srand(static_cast<unsigned int>(time(0)));


    // 输出表头
    cout << setw(12) << "n"
         << setw(30) << "Alg2 Avg Time (ms)"
         << setw(30) << "Alg2_unroll Avg Time (ms)" << endl;


    // 测试 n 从 10 到 1000000，每一行 n 为前一行的 10 倍
    for (int n = 10; n <= 1000000; n *= 10) {
        // 构造测试数组：大小为 n 的随机整数数组，元素在 0~100 范围内
        vector<int> a(n);
        for (int i = 0; i < n; i++) {
            a[i] = rand() % 101;
        }

        // 为避免首次运行影响缓存等因素，先预热一次
        volatile int dummy = algorithm2(a, n);
        dummy = algorithm2_unroll(a, n);

        const int repetitions = 100;
        long long totalTimeAlg2 = 0;         // 原始算法累计时间（微秒）
        long long totalTimeAlg2_unroll = 0;    // 展开算法累计时间（微秒）
        int result1 = 0, result2 = 0;          // 防止编译器优化


        // 测试原始多路链式算法
        for (int rep = 0; rep < repetitions; rep++) {
            auto start = high_resolution_clock::now();
            int sum = algorithm2(a, n);
            auto end = high_resolution_clock::now();
            totalTimeAlg2 += duration_cast<microseconds>(end - start).count();
            result1 += sum;
        }

        // 测试展开后的多路链式算法
        for (int rep = 0; rep < repetitions; rep++) {
            auto start = high_resolution_clock::now();
            int sum = algorithm2_unroll(a, n);
            auto end = high_resolution_clock::now();
            totalTimeAlg2_unroll += duration_cast<microseconds>(end - start).count();
            result2 += sum;
        }

        // 计算平均时间（毫秒）
        double avgTimeAlg2 = totalTimeAlg2 / (double)repetitions / 1000.0;
        double avgTimeAlg2_unroll = totalTimeAlg2_unroll / (double)repetitions / 1000.0;

        // 输出测试结果（n, 平均时间）
        cout << setw(12) << n
             << setw(30) << fixed << setprecision(6) << avgTimeAlg2
             << setw(30) << fixed << setprecision(6) << avgTimeAlg2_unroll << endl;
    }

    return 0;
}