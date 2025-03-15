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
    vector<int> n_values = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                            200, 300, 400, 500, 600, 700, 800, 900,
                            1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000};


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