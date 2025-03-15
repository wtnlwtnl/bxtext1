#include <iostream>
#include <windows.h>
#include <vector>
#include <iomanip>
#include <cstdlib> // 用于随机数生成
using namespace std;

// 使用高精度计时器 QueryPerformanceCounter()
double get_time_ms(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER freq) {
    return (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
}

// 测试算法：执行矩阵-向量乘法
void matrix_vector_mult(int n) {
    vector<vector<double>> B(n, vector<double>(n, 1.0)); // 生成 n x n 矩阵
    vector<double> a(n, 1.0); // 生成 n 维向量
    vector<double> sum(n, 0.0); // 存储结果

    for (int i = 0; i < n; i++) {
        sum[i] = 0.0;
        for (int j = 0; j < n; j++) {
            sum[i] += B[j][i] * a[j];
        }
    }
}

int main() {
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq); // 获取时钟频率

    vector<int> n_values = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                            200, 300, 400, 500, 600, 700, 800, 900,
                            1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000};

    // 打印表头
    cout << left << setw(10) << "n" 
         << setw(15) << "重复次数"
         << setw(15) << "总时间(ms)"
         << setw(15) << "平均时间(ms)" << endl;
    cout << string(55, '-') << endl;

    // 遍历不同的 n 值
    for (int n : n_values) {
        int repeat_count = rand() % 100 + 1; // 生成 1 到 100 之间的随机重复次数
        double total_time = 0.0;

        QueryPerformanceCounter(&start); // 记录开始时间
        for (int r = 0; r < repeat_count; r++) {
            matrix_vector_mult(n); // 执行算法
        }
        QueryPerformanceCounter(&end); // 记录结束时间

        total_time = get_time_ms(start, end, freq); // 计算总时间(ms)
        double avg_time = total_time / repeat_count; // 计算平均时间(ms)

        // 打印结果
        cout << left << setw(10) << n
             << setw(15) << repeat_count
             << setw(15) << fixed << setprecision(3) << total_time
             << setw(15) << fixed << setprecision(7) << avg_time << endl;
    }

    return 0;
}