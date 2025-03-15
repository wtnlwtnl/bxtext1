#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <algorithm> // for max()




using namespace std;




// 算法1：平凡算法（逐元素求和）
void algorithm1(const vector<int>& a, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
}




// 算法2：多路链式算法（两路累加）
void algorithm2(const vector<int>& a, int n) {
    int sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; i += 2) {
        sum1 += a[i];
        if (i + 1 < n) {
            sum2 += a[i + 1];
        }
    }
    int sum = sum1 + sum2;
}




// 算法3：二重循环求和算法（递归归约）
int algorithm3(vector<int> a, int n) {
    int m = n;
    while (m > 1) {
        for (int i = 0; i < m / 2; i++) {
            a[i] = a[2 * i] + a[2 * i + 1];
        }
        m /= 2;
    }
    return a[0];
}




int main() {
    srand(static_cast<unsigned int>(time(0)));




    // 取 n 的值
    vector<int> n_values = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
        2048, 4096, 8192,16384,32768};




    // 输出表头
    cout << setw(7) << "n"
         << setw(24) << "重复次数"
         << setw(26) << "Alg1总时间(ms)"
         << setw(22) << "Alg1平均(ms)"
         << setw(25) << "Alg2总时间(ms)"
         << setw(22) << "Alg2平均(ms)"
         << setw(25) << "Alg3总时间(ms)"
         << setw(22) << "Alg3平均(ms)"
         << endl;


    // 遍历所有 n 值进行测试
    for (auto n : n_values) {
        // 生成一个大小为 n 的随机数组，元素为 0 到 100 的随机数
        vector<int> a(n);
        for (int i = 0; i < n; i++) {
            a[i] = rand() % 101; // 0~100
        }


        // 随机生成重复次数（范围 0-100），若为 0 则至少重复一次
        int repetitions = max(1, rand() % 101);


        double total_time1 = 0.0, total_time2 = 0.0, total_time3 = 0.0;


        // 测试算法1：平凡算法
        clock_t start_time = clock();
        for (int rep = 0; rep < repetitions; rep++) {
            algorithm1(a, n);
        }
        total_time1 = double(clock() - start_time) / CLOCKS_PER_SEC * 1000.0; // ms
        double avg_time1 = total_time1 / repetitions;


        // 测试算法2：多路链式算法
        start_time = clock();
        for (int rep = 0; rep < repetitions; rep++) {
            algorithm2(a, n);
        }
        total_time2 = double(clock() - start_time) / CLOCKS_PER_SEC * 1000.0;
        double avg_time2 = total_time2 / repetitions;


        // 测试算法3：二重循环求和算法
        start_time = clock();
        for (int rep = 0; rep < repetitions; rep++) {
            // 需要复制数组，因为 algorithm3 会修改数组 a
            vector<int> temp_a = a;
            algorithm3(temp_a, n);
        }
        total_time3 = double(clock() - start_time) / CLOCKS_PER_SEC * 1000.0;
        double avg_time3 = total_time3 / repetitions;


        // 输出结果（时间保留三位小数）
        cout << setw(8) << n
             << setw(15) << repetitions
             << setw(20) << fixed << setprecision(3) << total_time1
             << setw(20) << fixed << setprecision(3) << avg_time1
             << setw(20) << fixed << setprecision(3) << total_time2
             << setw(20) << fixed << setprecision(3) << avg_time2
             << setw(20) << fixed << setprecision(3) << total_time3
             << setw(20) << fixed << setprecision(3) << avg_time3
             << endl;
    }


    return 0;
}
