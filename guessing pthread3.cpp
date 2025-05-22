#include "PCFG.h"
#include <pthread.h>
#include <queue>
#include <functional>
#include <atomic>
#include <thread>
#include <algorithm>
using namespace std;

// 静态线程池全局变量
pthread_t g_threads[4];             // 全局工作线程
pthread_mutex_t g_mutex;            // 全局互斥锁
pthread_cond_t g_condition;         // 条件变量用于线程同步
pthread_mutex_t g_complete_mutex;   // 完成任务互斥锁
pthread_cond_t g_complete_condition; // 完成条件变量
bool g_initialized = false;         // 线程池初始化标志
bool g_shutdown = false;            // 线程池关闭标志
int g_active_tasks = 0;             // 活动任务计数
thread_local vector<string> tls_buffer; // 线程局部存储缓冲区，避免频繁内存分配

// 任务结构
struct Task {
    std::function<void()> func;
};

// 任务队列
std::queue<Task> g_tasks;

// 线程工作函数
void* thread_worker(void* arg) {
    while (true) {
        Task task;
        bool has_task = false;

        // 获取任务
        pthread_mutex_lock(&g_mutex);
        while (g_tasks.empty() && !g_shutdown) {
            pthread_cond_wait(&g_condition, &g_mutex);
        }

        if (g_shutdown && g_tasks.empty()) {
            pthread_mutex_unlock(&g_mutex);
            break;
        }

        if (!g_tasks.empty()) {
            task = g_tasks.front();
            g_tasks.pop();
            has_task = true;
        }
        pthread_mutex_unlock(&g_mutex);

        // 执行任务
        if (has_task) {
            task.func();

            // 通知完成
            pthread_mutex_lock(&g_complete_mutex);
            g_active_tasks--;
            if (g_active_tasks == 0) {
                pthread_cond_signal(&g_complete_condition);
            }
            pthread_mutex_unlock(&g_complete_mutex);
        }
    }
    return NULL;
}

// 初始化静态线程池
void init_thread_pool() {
    if (g_initialized) return;

    pthread_mutex_init(&g_mutex, NULL);
    pthread_cond_init(&g_condition, NULL);
    pthread_mutex_init(&g_complete_mutex, NULL);
    pthread_cond_init(&g_complete_condition, NULL);

    for (int i = 0; i < 4; i++) {
        pthread_create(&g_threads[i], NULL, thread_worker, NULL);
    }

    g_initialized = true;
    g_shutdown = false;
    atexit([]() {
        // 关闭线程池
        pthread_mutex_lock(&g_mutex);
        g_shutdown = true;
        pthread_cond_broadcast(&g_condition);
        pthread_mutex_unlock(&g_mutex);

        // 等待线程结束
        for (int i = 0; i < 4; i++) {
            pthread_join(g_threads[i], NULL);
        }

        pthread_mutex_destroy(&g_mutex);
        pthread_cond_destroy(&g_condition);
        pthread_mutex_destroy(&g_complete_mutex);
        pthread_cond_destroy(&g_complete_condition);
    });
}

// 提交任务到线程池
void submit_task(std::function<void()> func) {
    pthread_mutex_lock(&g_mutex);
    Task task = {func};
    g_tasks.push(task);
    g_active_tasks++;
    pthread_cond_signal(&g_condition);
    pthread_mutex_unlock(&g_mutex);
}

// 等待所有任务完成
void wait_for_tasks() {
    pthread_mutex_lock(&g_complete_mutex);
    while (g_active_tasks > 0) {
        pthread_cond_wait(&g_complete_condition, &g_complete_mutex);
    }
    pthread_mutex_unlock(&g_complete_mutex);
}

// 4.2 自适应任务粒度 - 估计字符串平均长度
int estimateStringLength(segment* a, int work_size) {
    int total = 0;
    int samples = std::min(50, work_size);
    
    // 采样估算字符串长度
    for (int i = 0; i < samples; i++) {
        total += a->ordered_values[i].length();
    }
    return samples > 0 ? total / samples : 10; // 默认长度为10
}

// 4.2 自适应任务粒度 - 确定最优块大小
int determineOptimalChunkSize(int work_size, int string_avg_length) {
    // 大任务时块大小要随字符串长度增加而减小，以平衡工作量
    if (work_size >= 100000) {
        return std::max(2000, 8000 / (1 + string_avg_length / 10)); 
    } else if (work_size >= 10000) {
        return std::max(1000, 4000 / (1 + string_avg_length / 10));
    } else {
        return 500; // 小任务使用较小的块大小
    }
}

// 4.3 混合并行策略 - 确定最佳线程数
int determineOptimalThreadCount(int work_size, int string_avg_length) {
    int max_threads = std::thread::hardware_concurrency();
    if (max_threads == 0) max_threads = 4; // 默认值
    
    if (work_size <= 5000) {
        return 1; // 小任务用单线程
    } else if (work_size <= 20000) {
        return std::min(2, max_threads); // 中小任务用较少线程
    } else if (work_size <= 100000) {
        return std::min(4, max_threads); // 中型任务
    } else {
        return std::min(8, max_threads); // 大型任务尽可能多线程
    }
}

// 优化的GenerateParallel函数实现
void PriorityQueue::GenerateParallel(PT pt) {
    // 确保线程池已初始化
    init_thread_pool();

    // 计算PT的概率
    CalProb(pt);

    // 处理第一种情况：只有一个segment时
    if (pt.content.size() == 1) {
        // 指向最后一个segment的指针
        segment *a;

        // 在模型中定位到这个segment
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        } else {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        // 4.2 自适应任务粒度 - 计算工作大小和字符串平均长度
        int work_size = pt.max_indices[0];
        int avg_string_length = estimateStringLength(a, work_size);
        
        // 4.3 混合并行策略 - 根据任务特性选择执行方法
        
        // 1. 极小任务 - 直接串行处理
        if (work_size <= 1000) {
            for (int i = 0; i < work_size; i++) {
                string guess = a->ordered_values[i];
                guesses.emplace_back(std::move(guess));
                total_guesses++;
            }
            return;
        }

        // 2. 中小任务 - 使用简单的静态任务分配
        if (work_size <= 20000) {
            int num_threads = determineOptimalThreadCount(work_size, avg_string_length);
            
            // 预分配结果空间，避免频繁扩容
            guesses.reserve(guesses.size() + work_size);

            // 使用互斥锁保护共享数据
            pthread_mutex_t result_mutex;
            pthread_mutex_init(&result_mutex, NULL);

            int items_per_thread = (work_size + num_threads - 1) / num_threads;

            // 创建和提交任务到线程池
            for (int t = 0; t < num_threads; t++) {
                int start = t * items_per_thread;
                int end = std::min(start + items_per_thread, work_size);

                if (start >= end) continue;

                // 创建任务闭包
                auto task = [this, a, start, end, &result_mutex]() {
                    // 4.4 内存优化 - 使用线程局部存储
                    tls_buffer.clear();
                    tls_buffer.reserve(end - start);

                    // 4.4 局部性提升 - 使用预取优化缓存命中率
                    for (int i = start; i < end; i++) {
                        if (i + 16 < end) {
                            __builtin_prefetch(&a->ordered_values[i + 16], 0, 1);
                        }
                        tls_buffer.emplace_back(a->ordered_values[i]);
                    }

                    // 4.5 批处理更新 - 一次性合并结果减少锁竞争
                    pthread_mutex_lock(&result_mutex);
                    this->guesses.insert(this->guesses.end(),
                                        std::make_move_iterator(tls_buffer.begin()),
                                        std::make_move_iterator(tls_buffer.end()));
                    this->total_guesses += tls_buffer.size();
                    pthread_mutex_unlock(&result_mutex);
                };

                submit_task(task);
            }

            // 等待所有任务完成
            wait_for_tasks();

            // 清理
            pthread_mutex_destroy(&result_mutex);
        } 
        // 3. 大型任务 - 使用动态工作窃取策略
        else {
            // 预分配结果空间
            guesses.reserve(guesses.size() + work_size);
            
            // 使用互斥锁保护共享数据
            pthread_mutex_t result_mutex;
            pthread_mutex_init(&result_mutex, NULL);

            // 4.1 动态工作窃取策略 - 使用原子变量管理任务分配
            std::atomic<int> next_chunk(0);
            const int CHUNK_SIZE = determineOptimalChunkSize(work_size, avg_string_length);
            const int MERGE_THRESHOLD = 5000; // 批处理更新阈值
            int num_chunks = (work_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
            int num_threads = determineOptimalThreadCount(work_size, avg_string_length);
            
            // 创建工作窃取任务
            auto task = [this, a, work_size, &next_chunk, CHUNK_SIZE, MERGE_THRESHOLD, num_chunks, &result_mutex]() {
                // 线程局部缓冲区
                tls_buffer.clear();
                tls_buffer.reserve(CHUNK_SIZE * 2); // 预留充足空间
                
                while (true) {
                    // 动态获取下一个工作块
                    int chunk_id = next_chunk.fetch_add(1);
                    if (chunk_id >= num_chunks) break;
                    
                    int start = chunk_id * CHUNK_SIZE;
                    int end = std::min(start + CHUNK_SIZE, work_size);
                    
                    // 4.4 内存优化与局部性提升
                    for (int i = start; i < end; i++) {
                        if (i + 16 < end) {
                            __builtin_prefetch(&a->ordered_values[i + 16], 0, 1);
                        }
                        tls_buffer.emplace_back(a->ordered_values[i]);
                    }
                    
                    // 4.5 批处理更新 - 当累积足够多结果或是最后一块时合并
                    if (tls_buffer.size() >= MERGE_THRESHOLD || chunk_id == num_chunks - 1) {
                        pthread_mutex_lock(&result_mutex);
                        this->guesses.insert(this->guesses.end(),
                                          std::make_move_iterator(tls_buffer.begin()),
                                          std::make_move_iterator(tls_buffer.end()));
                        this->total_guesses += tls_buffer.size();
                        pthread_mutex_unlock(&result_mutex);
                        
                        tls_buffer.clear();
                    }
                }
                
                // 最后清理剩余的结果
                if (!tls_buffer.empty()) {
                    pthread_mutex_lock(&result_mutex);
                    this->guesses.insert(this->guesses.end(),
                                      std::make_move_iterator(tls_buffer.begin()),
                                      std::make_move_iterator(tls_buffer.end()));
                    this->total_guesses += tls_buffer.size();
                    pthread_mutex_unlock(&result_mutex);
                }
            };
            
            // 提交任务到线程池
            for (int i = 0; i < num_threads; i++) {
                submit_task(task);
            }
            
            // 等待所有任务完成
            wait_for_tasks();
            
            // 清理
            pthread_mutex_destroy(&result_mutex);
        }
    }
    else {
        string guess;
        int seg_idx = 0;

        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 2) {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 3) {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1) {
                break;
            }
        }

        // 指向最后一个segment的指针
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        } else if (pt.content[pt.content.size() - 1].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        } else {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }

        // 4.2 自适应任务粒度 - 计算工作大小和字符串平均长度
        int work_size = pt.max_indices[pt.content.size() - 1];
        int avg_string_length = estimateStringLength(a, work_size);
        
        // 初始猜测字符串长度
        int guess_length = guess.length() + avg_string_length;
        
        // 4.3 混合并行策略 - 根据任务特性选择执行方法
        
        // 1. 极小任务 - 直接串行处理
        if (work_size <= 1000) {
            for (int i = 0; i < work_size; i++) {
                string temp = guess + a->ordered_values[i];
                guesses.emplace_back(std::move(temp));
                total_guesses++;
            }
            return;
        }

        // 2. 中小任务 - 使用简单的静态任务分配
        if (work_size <= 20000) {
            int num_threads = determineOptimalThreadCount(work_size, guess_length);
            
            // 预分配结果空间，避免频繁扩容
            guesses.reserve(guesses.size() + work_size);

            // 使用互斥锁保护共享数据
            pthread_mutex_t result_mutex;
            pthread_mutex_init(&result_mutex, NULL);

            int items_per_thread = (work_size + num_threads - 1) / num_threads;

            // 创建和提交任务到线程池
            for (int t = 0; t < num_threads; t++) {
                int start = t * items_per_thread;
                int end = std::min(start + items_per_thread, work_size);

                if (start >= end) continue;

                // 创建任务闭包
                auto task = [this, a, start, end, guess, &result_mutex]() {
                    // 4.4 内存优化 - 使用线程局部存储
                    tls_buffer.clear();
                    tls_buffer.reserve(end - start);

                    // 4.4 局部性提升 - 使用预取优化缓存命中率
                    for (int i = start; i < end; i++) {
                        if (i + 16 < end) {
                            __builtin_prefetch(&a->ordered_values[i + 16], 0, 1);
                        }
                        tls_buffer.emplace_back(guess + a->ordered_values[i]);
                    }

                    // 4.5 批处理更新 - 一次性合并结果减少锁竞争
                    pthread_mutex_lock(&result_mutex);
                    this->guesses.insert(this->guesses.end(),
                                        std::make_move_iterator(tls_buffer.begin()),
                                        std::make_move_iterator(tls_buffer.end()));
                    this->total_guesses += tls_buffer.size();
                    pthread_mutex_unlock(&result_mutex);
                };

                submit_task(task);
            }

            // 等待所有任务完成
            wait_for_tasks();

            // 清理
            pthread_mutex_destroy(&result_mutex);
        } 
        // 3. 大型任务 - 使用动态工作窃取策略
        else {
            // 预分配结果空间
            guesses.reserve(guesses.size() + work_size);
            
            // 使用互斥锁保护共享数据
            pthread_mutex_t result_mutex;
            pthread_mutex_init(&result_mutex, NULL);

            // 4.1 动态工作窃取策略 - 使用原子变量管理任务分配
            std::atomic<int> next_chunk(0);
            const int CHUNK_SIZE = determineOptimalChunkSize(work_size, guess_length);
            const int MERGE_THRESHOLD = 5000; // 批处理更新阈值
            int num_chunks = (work_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
            int num_threads = determineOptimalThreadCount(work_size, guess_length);
            
            // 创建工作窃取任务
            auto task = [this, a, work_size, &next_chunk, CHUNK_SIZE, MERGE_THRESHOLD, num_chunks, guess, &result_mutex]() {
                // 线程局部缓冲区
                tls_buffer.clear();
                tls_buffer.reserve(CHUNK_SIZE * 2); // 预留充足空间
                
                while (true) {
                    // 动态获取下一个工作块
                    int chunk_id = next_chunk.fetch_add(1);
                    if (chunk_id >= num_chunks) break;
                    
                    int start = chunk_id * CHUNK_SIZE;
                    int end = std::min(start + CHUNK_SIZE, work_size);
                    
                    // 4.4 内存优化与局部性提升
                    for (int i = start; i < end; i++) {
                        if (i + 16 < end) {
                            __builtin_prefetch(&a->ordered_values[i + 16], 0, 1);
                        }
                        tls_buffer.emplace_back(guess + a->ordered_values[i]);
                    }
                    
                    // 4.5 批处理更新 - 当累积足够多结果或是最后一块时合并
                    if (tls_buffer.size() >= MERGE_THRESHOLD || chunk_id == num_chunks - 1) {
                        pthread_mutex_lock(&result_mutex);
                        this->guesses.insert(this->guesses.end(),
                                          std::make_move_iterator(tls_buffer.begin()),
                                          std::make_move_iterator(tls_buffer.end()));
                        this->total_guesses += tls_buffer.size();
                        pthread_mutex_unlock(&result_mutex);
                        
                        tls_buffer.clear();
                    }
                }
                
                // 最后清理剩余的结果
                if (!tls_buffer.empty()) {
                    pthread_mutex_lock(&result_mutex);
                    this->guesses.insert(this->guesses.end(),
                                      std::make_move_iterator(tls_buffer.begin()),
                                      std::make_move_iterator(tls_buffer.end()));
                    this->total_guesses += tls_buffer.size();
                    pthread_mutex_unlock(&result_mutex);
                }
            };
            
            // 提交任务到线程池
            for (int i = 0; i < num_threads; i++) {
                submit_task(task);
            }
            
            // 等待所有任务完成
            wait_for_tasks();
            
            // 清理
            pthread_mutex_destroy(&result_mutex);
        }
    }
}

// 原始串行版本保持不变，用于比较和小任务处理
void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是"纯粹的"PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中"123456"为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{
    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    GenerateParallel(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// 原始串行版Generate函数保持不变
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }

        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}

// 保留原始多线程函数为了兼容性
struct ThreadArg1 {
    PriorityQueue* pq;
    PT pt;
    int start;
    int end;
    vector<string>* local_guesses;
    segment* a;
    int* local_total;
};

struct ThreadArg2 {
    PriorityQueue* pq;
    PT pt;
    int start;
    int end;
    vector<string>* local_guesses;
    segment* a;
    int* local_total;
    string guess;
};

void* generate_worker1(void* arg) {
    ThreadArg1* targ = (ThreadArg1*)arg;
    for (int i = targ->start; i < targ->end; i++) {
        string guess = targ->a->ordered_values[i];
        targ->local_guesses->push_back(guess);
        (*targ->local_total)++;
    }
    return NULL;
}

void* generate_worker2(void* arg) {
    ThreadArg2* targ = (ThreadArg2*)arg;
    for (int i = targ->start; i < targ->end; i++) {
        string temp = targ->guess + targ->a->ordered_values[i];
        targ->local_guesses->push_back(temp);
        (*targ->local_total)++;
    }
    return NULL;
}