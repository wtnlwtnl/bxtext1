// #include "PCFG.h"
// #include "cuda_generate.h"
// #include <omp.h>
// #include <queue>
// #include <mutex>
// #include <condition_variable>
// #include <atomic>
// #include <thread>
// using namespace std;

// // 任务类型
// enum TaskType {
//     CPU_TASK,
//     GPU_TASK,
//     SHUTDOWN
// };

// // 前向声明
// class TaskScheduler;

// // 异步任务结构体
// struct AsyncTask {
//     PT pt;
//     TaskType type;
//     PriorityQueue* queuePtr;
//     bool completed;
//     std::mutex completionMutex;
//     std::condition_variable completionCV;
//     int priority;  // 优先级，值越低优先级越高

//     AsyncTask() : completed(false), queuePtr(nullptr), priority(0) {}
    
//     void markCompleted() {
//         {
//             std::lock_guard<std::mutex> lock(completionMutex);
//             completed = true;
//         }
//         completionCV.notify_all();
//     }
    
//     void waitForCompletion() {
//         std::unique_lock<std::mutex> lock(completionMutex);
//         completionCV.wait(lock, [this] { return completed; });
//     }
// };

// // 自定义比较函数用于优先队列
// struct TaskComparator {
//     bool operator()(AsyncTask* a, AsyncTask* b) {
//         return a->priority > b->priority;  // 低值优先
//     }
// };

// // 任务调度器
// class TaskScheduler {
// private:
//     std::priority_queue<AsyncTask*, vector<AsyncTask*>, TaskComparator> cpuTasks;
//     std::queue<AsyncTask*> gpuTasks;
//     std::mutex cpuQueueMutex, gpuQueueMutex;
//     std::condition_variable cpuQueueCV, gpuQueueCV;
//     std::atomic<bool> shutdown;
//     std::vector<std::thread> cpuWorkers;
//     std::thread gpuWorker;
//     std::atomic<int> activeCpuTasks;
//     std::atomic<int> activeGpuTasks;
//     int numCpuWorkers;

// public:
//     TaskScheduler(int cpuWorkers = 2) : shutdown(false), activeCpuTasks(0), activeGpuTasks(0), numCpuWorkers(cpuWorkers) {
//         // 启动CPU工作线程
//         for (int i = 0; i < numCpuWorkers; i++) {
//             this->cpuWorkers.emplace_back(&TaskScheduler::cpuWorkerThread, this, i);
//         }
        
//         // 启动GPU工作线程
//         gpuWorker = std::thread(&TaskScheduler::gpuWorkerThread, this);
//     }
    
//     ~TaskScheduler() {
//         stop();
//     }
    
//     void enqueueCpuTask(AsyncTask* task) {
//         {
//             std::lock_guard<std::mutex> lock(cpuQueueMutex);
//             cpuTasks.push(task);
//         }
//         cpuQueueCV.notify_one();
//     }
    
//     void enqueueGpuTask(AsyncTask* task) {
//         {
//             std::lock_guard<std::mutex> lock(gpuQueueMutex);
//             gpuTasks.push(task);
//         }
//         gpuQueueCV.notify_one();
//     }
    
//     bool stealCpuTask(AsyncTask** task) {
//         std::lock_guard<std::mutex> lock(cpuQueueMutex);
//         if (cpuTasks.empty()) return false;
//         *task = cpuTasks.top();
//         cpuTasks.pop();
//         return true;
//     }
    
//     int cpuQueueSize() {
//         std::lock_guard<std::mutex> lock(cpuQueueMutex);
//         return cpuTasks.size();
//     }
    
//     int gpuQueueSize() {
//         std::lock_guard<std::mutex> lock(gpuQueueMutex);
//         return gpuTasks.size();
//     }
    
//     // 添加任务并选择最佳执行设备
//     void addTask(AsyncTask* task) {
//         if (task->type == CPU_TASK) {
//             enqueueCpuTask(task);
//         } else if (task->type == GPU_TASK) {
//             // 如果GPU队列太长或GPU忙，考虑重定向到CPU
//             if (activeGpuTasks > 3 || gpuQueueSize() > 10) {
//                 task->type = CPU_TASK;
//                 enqueueCpuTask(task);
//             } else {
//                 enqueueGpuTask(task);
//             }
//         }
//     }
    
//     void stop() {
//         if (shutdown) return;
        
//         shutdown = true;
//         cpuQueueCV.notify_all();
//         gpuQueueCV.notify_all();
        
//         for (auto& worker : cpuWorkers) {
//             if (worker.joinable()) {
//                 worker.join();
//             }
//         }
        
//         if (gpuWorker.joinable()) {
//             gpuWorker.join();
//         }
        
//         cpuWorkers.clear();
//     }

// private:
//     // CPU工作线程
//     void cpuWorkerThread(int id) {
//         while (!shutdown) {
//             AsyncTask* task = nullptr;
            
//             // 尝试从CPU队列获取任务
//             {
//                 std::unique_lock<std::mutex> lock(cpuQueueMutex);
//                 cpuQueueCV.wait_for(lock, std::chrono::milliseconds(50), 
//                     [this] { return !cpuTasks.empty() || shutdown; });
                
//                 if (shutdown) break;
                
//                 if (!cpuTasks.empty()) {
//                     task = cpuTasks.top();
//                     cpuTasks.pop();
//                 }
//             }
            
//             // 如果CPU队列为空，尝试从GPU队列窃取任务
//             if (!task && activeGpuTasks > 0) {
//                 std::lock_guard<std::mutex> lock(gpuQueueMutex);
//                 if (!gpuTasks.empty()) {
//                     task = gpuTasks.front();
//                     gpuTasks.pop();
//                     task->type = CPU_TASK; // 转换为CPU任务
//                 }
//             }
            
//             // 处理任务
//             if (task) {
//                 activeCpuTasks++;
                
//                 if (task->queuePtr) {
//                     task->queuePtr->GenerateCPU(task->pt);
//                 }
                
//                 task->markCompleted();
//                 activeCpuTasks--;
//             }
//         }
//     }
    
//     // GPU工作线程
//     void gpuWorkerThread() {
//         while (!shutdown) {
//             AsyncTask* task = nullptr;
            
//             // 从GPU队列获取任务
//             {
//                 std::unique_lock<std::mutex> lock(gpuQueueMutex);
//                 gpuQueueCV.wait_for(lock, std::chrono::milliseconds(50), 
//                     [this] { return !gpuTasks.empty() || shutdown; });
                
//                 if (shutdown) break;
                
//                 if (!gpuTasks.empty()) {
//                     task = gpuTasks.front();
//                     gpuTasks.pop();
//                 }
//             }
            
//             // 处理任务
//             if (task) {
//                 activeGpuTasks++;
                
//                 if (task->queuePtr) {
//                     task->queuePtr->Generate(task->pt);
//                 }
                
//                 task->markCompleted();
//                 activeGpuTasks--;
//             }
//         }
//     }
// };

// // 全局任务调度器
// TaskScheduler* globalScheduler = nullptr;

// // 工作线程数组和任务队列不再需要，由TaskScheduler管理
// std::atomic<bool> shutdownThreads(false);

// void PriorityQueue::CalProb(PT &pt)
// {
//     // 计算PriorityQueue里面一个PT的流程如下：
//     // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
//     // 2. 需要注意的是，Queue里面的PT不是"纯粹的"PT，而是除了最后一个segment以外，全部被value实例化的PT
//     // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中"123456"为L6的一个具体value。
//     // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

//     // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
//     pt.prob = pt.preterm_prob;

//     // index: 标注当前segment在PT中的位置
//     int index = 0;

//     for (int idx : pt.curr_indices)
//     {
//         // pt.content[index].PrintSeg();
//         if (pt.content[index].type == 1)
//         {
//             // 下面这行代码的意义：
//             // pt.content[index]：目前需要计算概率的segment
//             // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
//             // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
//             // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
//             pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
//             pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
//             // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
//             // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
//         }
//         if (pt.content[index].type == 2)
//         {
//             pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
//             pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
//             // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
//             // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
//         }
//         if (pt.content[index].type == 3)
//         {
//             pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
//             pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
//             // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
//             // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
//         }
//         index += 1;
//     }
//     // cout << pt.prob << endl;
// }

// // 决定PT是在GPU还是CPU上处理 - 使用更复杂的启发式规则
// bool shouldProcessOnGPU(const PT& pt) {
//     // 获取最后一个segment
//     const segment& lastSeg = pt.content.back();
    
//     // 单段PT或长segment PT适合GPU处理
//     if (pt.content.size() == 1 || lastSeg.length > 4) {
//         return true;
//     }
    
//     // 使用全局调度器的负载信息做动态决策
//     if (globalScheduler) {
//         // GPU负载较轻时优先使用GPU
//         if (globalScheduler->gpuQueueSize() < globalScheduler->cpuQueueSize() / 2) {
//             return true;
//         }
//     }
    
//     // 默认在CPU上处理
//     return false;
// }

// void PriorityQueue::init()
// {
//     // 初始化全局任务调度器
//     if (!globalScheduler) {
//         // 获取系统CPU核心数的一半作为CPU工作线程数量
//         int cpuThreads = std::max(2, (int)(std::thread::hardware_concurrency() / 2));
//         globalScheduler = new TaskScheduler(cpuThreads);
//     }
    
//     // 用所有可能的PT，按概率降序填满整个优先队列
//     for (PT pt : m.ordered_pts)
//     {
//         for (segment seg : pt.content)
//         {
//             if (seg.type == 1)
//             {
//                 // 下面这行代码的意义：
//                 // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
//                 // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
//                 // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
//                 // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
//                 // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
//                 pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
                
//                 // 预缓存字母segment值到GPU
//                 cacheSegmentValues(seg.type, seg.length, m.letters[m.FindLetter(seg)].ordered_values);
//             }
//             if (seg.type == 2)
//             {
//                 pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
                
//                 // 预缓存数字segment值到GPU
//                 cacheSegmentValues(seg.type, seg.length, m.digits[m.FindDigit(seg)].ordered_values);
//             }
//             if (seg.type == 3)
//             {
//                 pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
                
//                 // 预缓存符号segment值到GPU
//                 cacheSegmentValues(seg.type, seg.length, m.symbols[m.FindSymbol(seg)].ordered_values);
//             }
//         }
//         pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;

//         // 计算当前pt的概率
//         CalProb(pt);
//         // 将PT放入优先队列
//         priority.emplace_back(pt);
//     }
// }

// void PriorityQueue::PopNext()
// {
//     // 调用多PT版本，但只处理一个PT
//     PopNextMultiple(1);
// }

// void PriorityQueue::PopNextMultiple(int count)
// {
//     // 确保不超出队列中可用的PT数量
//     count = min(count, (int)priority.size());
//     if (count <= 0) return;
    
//     // 创建任务并加入调度器
//     vector<AsyncTask*> tasks;
//     for (int i = 0; i < count; i++) {
//         AsyncTask* task = new AsyncTask();
//         task->pt = priority[i];
//         task->queuePtr = this;
//         task->priority = i;  // 使用队列位置作为优先级
        
//         // 根据PT特性决定在CPU还是GPU上处理
//         task->type = shouldProcessOnGPU(priority[i]) ? GPU_TASK : CPU_TASK;
        
//         tasks.push_back(task);
//         globalScheduler->addTask(task);
//     }
    
//     // 非阻塞方式检查任务完成情况，允许提前处理已完成的任务
//     vector<PT> all_new_pts;
//     vector<bool> processed(count, false);
//     int completedCount = 0;
    
//     // 使用更高效的完成检查策略
//     while (completedCount < count) {
//         for (int i = 0; i < count; i++) {
//             if (!processed[i]) {
//                 // 使用非阻塞检查
//                 std::unique_lock<std::mutex> lock(tasks[i]->completionMutex, std::try_to_lock);
//                 if (lock && tasks[i]->completed) {
//                     // 收集新PT
//                     vector<PT> new_pts = priority[i].NewPTs();
//                     all_new_pts.insert(all_new_pts.end(), new_pts.begin(), new_pts.end());
                    
//                     processed[i] = true;
//                     completedCount++;
//                 }
//             }
//         }
        
//         // 短暂休眠以减少CPU使用率
//         if (completedCount < count) {
//             std::this_thread::sleep_for(std::chrono::microseconds(100));
//         }
//     }
    
//     // 清理任务
//     for (auto* task : tasks) {
//         delete task;
//     }
    
//     // 移除已处理的PT
//     priority.erase(priority.begin(), priority.begin() + count);
    
//     // 为所有新PT计算概率并插入回优先队列
//     for (PT& pt : all_new_pts) {
//         // 计算概率
//         CalProb(pt);
        
//         // 根据概率将PT插入到优先队列的适当位置
//         bool inserted = false;
//         for (auto iter = priority.begin(); iter != priority.end(); iter++) {
//             // 对于非队首和队尾的特殊情况
//             if (iter != priority.end() - 1 && iter != priority.begin()) {
//                 // 判定概率
//                 if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
//                     priority.emplace(iter + 1, pt);
//                     inserted = true;
//                     break;
//                 }
//             }
//             if (iter == priority.begin() && iter->prob < pt.prob) {
//                 priority.emplace(iter, pt);
//                 inserted = true;
//                 break;
//             }
//         }
        
//         // 如果没有找到合适位置，添加到队列末尾
//         if (!inserted) {
//             priority.emplace_back(pt);
//         }
//     }
// }

// // CPU版本的Generate函数 - 在CPU上处理PT生成密码
// void PriorityQueue::GenerateCPU(PT pt)
// {
//     // 计算PT的概率
//     CalProb(pt);

//     // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
//     if (pt.content.size() == 1)
//     {
//         int segmentType = pt.content[0].type;
//         int segmentLength = pt.content[0].length;
//         int valueCount = pt.max_indices[0];
        
//         // 获取适当的segment列表
//         vector<string>* segmentValues = nullptr;
//         if (segmentType == 1) { // 字母
//             segmentValues = &m.letters[m.FindLetter(pt.content[0])].ordered_values;
//         } else if (segmentType == 2) { // 数字
//             segmentValues = &m.digits[m.FindDigit(pt.content[0])].ordered_values;
//         } else if (segmentType == 3) { // 符号
//             segmentValues = &m.symbols[m.FindSymbol(pt.content[0])].ordered_values;
//         }
        
//         if (segmentValues) {
//             size_t originalSize = guesses.size();
//             guesses.resize(originalSize + valueCount);
            
//             // 在CPU上生成所有猜测
//             #pragma omp parallel for
//             for (int i = 0; i < valueCount; i++) {
//                 if (i < segmentValues->size()) {
//                     guesses[originalSize + i] = (*segmentValues)[i];
//                 }
//             }
            
//             total_guesses += valueCount;
//         }
//     }
//     else
//     {
//         string guess;
//         int seg_idx = 0;
//         // 构建除了最后一个segment之外的前缀
//         for (int idx : pt.curr_indices)
//         {
//             if (pt.content[seg_idx].type == 1)
//             {
//                 guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
//             }
//             if (pt.content[seg_idx].type == 2)
//             {
//                 guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
//             }
//             if (pt.content[seg_idx].type == 3)
//             {
//                 guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
//             }
//             seg_idx += 1;
//             if (seg_idx == pt.content.size() - 1)
//             {
//                 break;
//             }
//         }

//         // 获取最后一个segment的值
//         int lastSegmentType = pt.content[pt.content.size() - 1].type;
//         int lastSegmentLength = pt.content[pt.content.size() - 1].length;
//         int valueCount = pt.max_indices[pt.content.size() - 1];
        
//         // 获取最后一个segment的所有可能值
//         vector<string>* lastSegmentValues = nullptr;
//         if (lastSegmentType == 1) { // 字母
//             lastSegmentValues = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])].ordered_values;
//         } else if (lastSegmentType == 2) { // 数字
//             lastSegmentValues = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])].ordered_values;
//         } else if (lastSegmentType == 3) { // 符号
//             lastSegmentValues = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])].ordered_values;
//         }
        
//         if (lastSegmentValues) {
//             size_t originalSize = guesses.size();
//             guesses.resize(originalSize + valueCount);
            
//             // 在CPU上生成所有猜测
//             #pragma omp parallel for
//             for (int i = 0; i < valueCount; i++) {
//                 if (i < lastSegmentValues->size()) {
//                     guesses[originalSize + i] = guess + (*lastSegmentValues)[i];
//                 }
//             }
            
//             total_guesses += valueCount;
//         }
//     }
// }

// // 这个函数是PCFG并行化算法的主要载体
// // 尽量看懂，然后进行并行实现
// void PriorityQueue::Generate(PT pt)
// {
//     // 计算PT的概率，这里主要是给PT的概率进行初始化
//     CalProb(pt);

//     // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
//     if (pt.content.size() == 1)
//     {
//         // 获取segment类型和长度
//         int segmentType = pt.content[0].type;
//         int segmentLength = pt.content[0].length;
        
//         // 使用优化的GPU实现
//         generateSingleSegmentGPU(segmentType, segmentLength, pt.max_indices[0], guesses, total_guesses);
//     }
//     else
//     {
//         string guess;
//         int seg_idx = 0;
//         // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
//         // segment值根据curr_indices中对应的值加以确定
//         for (int idx : pt.curr_indices)
//         {
//             if (pt.content[seg_idx].type == 1)
//             {
//                 guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
//             }
//             if (pt.content[seg_idx].type == 2)
//             {
//                 guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
//             }
//             if (pt.content[seg_idx].type == 3)
//             {
//                 guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
//             }
//             seg_idx += 1;
//             if (seg_idx == pt.content.size() - 1)
//             {
//                 break;
//             }
//         }

//         // 获取最后一个segment的类型和长度
//         int lastSegmentType = pt.content[pt.content.size() - 1].type;
//         int lastSegmentLength = pt.content[pt.content.size() - 1].length;
        
//         // 使用优化的GPU实现
//         generateMultiSegmentGPU(guess, lastSegmentType, lastSegmentLength, 
//                                pt.max_indices[pt.content.size() - 1], guesses, total_guesses);
//     }
// }

// // PT::NewPTs的实现
// vector<PT> PT::NewPTs()
// {
//     // 存储生成的新PT
//     vector<PT> res;

//     // 假如这个PT只有一个segment
//     // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
//     // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
//     if (content.size() == 1)
//     {
//         return res;
//     }
//     else
//     {
//         // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
//         int init_pivot = pivot;

//         // 开始遍历所有位置值大于等于init_pivot值的segment
//         // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
//         for (int i = pivot; i < curr_indices.size() - 1; i += 1)
//         {
//             // curr_indices: 标记各segment目前的value在模型里对应的下标
//             curr_indices[i] += 1;

//             // max_indices：标记各segment在模型中一共有多少个value
//             if (curr_indices[i] < max_indices[i])
//             {
//                 // 更新pivot值
//                 pivot = i;
//                 res.emplace_back(*this);
//             }

//             // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
//             curr_indices[i] -= 1;
//         }
//         pivot = init_pivot;
//         return res;
//     }
// }

// // 关闭所有工作线程
// void shutdownWorkerThreads() {
//     if (globalScheduler) {
//         globalScheduler->stop();
//         delete globalScheduler;
//         globalScheduler = nullptr;
//     }
// }

#include "PCFG.h"
#include "cuda_generate.h"
#include <omp.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
using namespace std;

// 任务类型
enum TaskType {
    CPU_TASK,
    GPU_TASK,
    SHUTDOWN
};

// 异步任务结构体
struct AsyncTask {
    PT pt;
    TaskType type;
    PriorityQueue* queuePtr;  // 指向队列的指针
    bool completed;
    std::mutex completionMutex;
    std::condition_variable completionCV;

    AsyncTask() : completed(false), queuePtr(nullptr) {}
    
    void markCompleted() {
        {
            std::lock_guard<std::mutex> lock(completionMutex);
            completed = true;
        }
        completionCV.notify_all();
    }
    
    void waitForCompletion() {
        std::unique_lock<std::mutex> lock(completionMutex);
        completionCV.wait(lock, [this] { return completed; });
    }
};

// 任务队列
std::queue<AsyncTask*> taskQueue;
std::mutex queueMutex;
std::condition_variable queueCV;
std::atomic<bool> shutdownThreads(false);
vector<std::thread> workerThreads;

// 预定义的工作线程数
const int NUM_WORKER_THREADS = 2;

// 工作线程函数
void workerThread(int id) {
    while (!shutdownThreads) {
        AsyncTask* task = nullptr;
        
        // 从队列中获取任务
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCV.wait(lock, []{ return !taskQueue.empty() || shutdownThreads; });
            
            if (shutdownThreads && taskQueue.empty()) {
                break;
            }
            
            if (!taskQueue.empty()) {
                task = taskQueue.front();
                taskQueue.pop();
            }
        }
        
        // 处理任务
        if (task) {
            if (task->type == CPU_TASK && task->queuePtr != nullptr) {
                // CPU处理
                task->queuePtr->GenerateCPU(task->pt);
            } else if (task->type == GPU_TASK) {
                // GPU处理 - 在实际实现中，这部分已经由主线程完成
            } else if (task->type == SHUTDOWN) {
                break;
            }
            
            // 标记任务完成
            task->markCompleted();
        }
    }
}

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
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

// 决定PT是在GPU还是CPU上处理
bool shouldProcessOnGPU(const PT& pt) {
    // 启发式规则：
    // 1. 单段PT适合在GPU上处理（大量相似任务，高并行）
    // 2. 多段PT如果最后一个segment长度很长，也适合GPU
    // 3. 其他情况在CPU上处理
    
    if (pt.content.size() == 1) {
        return true; // 单段PT在GPU上处理
    }
    
    // 获取最后一个segment
    const segment& lastSeg = pt.content.back();
    
    // 如果最后segment长度超过4，在GPU上处理
    if (lastSeg.length > 4) {
        return true;
    }
    
    // 默认在CPU上处理
    return false;
}

void PriorityQueue::init()
{
    // 初始化工作线程
    shutdownThreads = false;
    
    for (int i = 0; i < NUM_WORKER_THREADS; i++) {
        workerThreads.emplace_back(workerThread, i);
    }
    
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // max_indices用来表示PT中各个segment的可能数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
                
                // 预缓存字母segment值到GPU
                cacheSegmentValues(seg.type, seg.length, m.letters[m.FindLetter(seg)].ordered_values);
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
                
                // 预缓存数字segment值到GPU
                cacheSegmentValues(seg.type, seg.length, m.digits[m.FindDigit(seg)].ordered_values);
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
                
                // 预缓存符号segment值到GPU
                cacheSegmentValues(seg.type, seg.length, m.symbols[m.FindSymbol(seg)].ordered_values);
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    // 调用多PT版本，但只处理一个PT
    PopNextMultiple(1);
}

void PriorityQueue::PopNextMultiple(int count)
{
    // 确保不超出队列中可用的PT数量
    count = min(count, (int)priority.size());
    if (count <= 0) return;
    
    // 存储所有要处理的PT
    vector<PT> gpuPTs;
    vector<PT> cpuPTs;
    
    // 将PT分为GPU和CPU处理组
    for (int i = 0; i < count; i++) {
        if (shouldProcessOnGPU(priority[i])) {
            gpuPTs.push_back(priority[i]);
        } else {
            cpuPTs.push_back(priority[i]);
        }
    }
    
    // 创建并添加CPU任务到队列
    vector<AsyncTask*> cpuTasks;
    for (auto& pt : cpuPTs) {
        AsyncTask* task = new AsyncTask();
        task->pt = pt;
        task->type = CPU_TASK;
        task->queuePtr = this;  // 指向当前PriorityQueue实例
        
        cpuTasks.push_back(task);
        
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            taskQueue.push(task);
        }
        queueCV.notify_one();
    }
    
    // 同步处理GPU上的PT (由于GPU资源有限，我们直接处理而不是放入队列)
    for (auto& pt : gpuPTs) {
        Generate(pt);
    }
    
    // 等待所有CPU任务完成
    for (auto* task : cpuTasks) {
        task->waitForCompletion();
        delete task;
    }
    
    // 收集并处理所有新生成的PT
    vector<PT> all_new_pts;
    for (int i = 0; i < count; i++) {
        // 获取当前PT生成的新PT
        vector<PT> new_pts = priority[i].NewPTs();
        // 收集所有新的PT
        all_new_pts.insert(all_new_pts.end(), new_pts.begin(), new_pts.end());
    }
    
    // 移除已处理的PT
    priority.erase(priority.begin(), priority.begin() + count);
    
    // 为所有新PT计算概率并插入回优先队列
    for (PT& pt : all_new_pts) {
        // 计算概率
        CalProb(pt);
        
        // 根据概率将PT插入到优先队列的适当位置
        bool inserted = false;
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.emplace(iter + 1, pt);
                    inserted = true;
                    break;
                }
            }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.emplace(iter, pt);
                inserted = true;
                break;
            }
        }
        
        // 如果没有找到合适位置，添加到队列末尾
        if (!inserted) {
            priority.emplace_back(pt);
        }
    }
}

// CPU版本的Generate函数 - 在CPU上处理PT生成密码
void PriorityQueue::GenerateCPU(PT pt)
{
    // 计算PT的概率
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        int segmentType = pt.content[0].type;
        int segmentLength = pt.content[0].length;
        int valueCount = pt.max_indices[0];
        
        // 获取适当的segment列表
        vector<string>* segmentValues = nullptr;
        if (segmentType == 1) { // 字母
            segmentValues = &m.letters[m.FindLetter(pt.content[0])].ordered_values;
        } else if (segmentType == 2) { // 数字
            segmentValues = &m.digits[m.FindDigit(pt.content[0])].ordered_values;
        } else if (segmentType == 3) { // 符号
            segmentValues = &m.symbols[m.FindSymbol(pt.content[0])].ordered_values;
        }
        
        if (segmentValues) {
            size_t originalSize = guesses.size();
            guesses.resize(originalSize + valueCount);
            
            // 在CPU上生成所有猜测
            #pragma omp parallel for
            for (int i = 0; i < valueCount; i++) {
                if (i < segmentValues->size()) {
                    guesses[originalSize + i] = (*segmentValues)[i];
                }
            }
            
            total_guesses += valueCount;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 构建除了最后一个segment之外的前缀
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

        // 获取最后一个segment的值
        int lastSegmentType = pt.content[pt.content.size() - 1].type;
        int lastSegmentLength = pt.content[pt.content.size() - 1].length;
        int valueCount = pt.max_indices[pt.content.size() - 1];
        
        // 获取最后一个segment的所有可能值
        vector<string>* lastSegmentValues = nullptr;
        if (lastSegmentType == 1) { // 字母
            lastSegmentValues = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])].ordered_values;
        } else if (lastSegmentType == 2) { // 数字
            lastSegmentValues = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])].ordered_values;
        } else if (lastSegmentType == 3) { // 符号
            lastSegmentValues = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])].ordered_values;
        }
        
        if (lastSegmentValues) {
            size_t originalSize = guesses.size();
            guesses.resize(originalSize + valueCount);
            
            // 在CPU上生成所有猜测
            #pragma omp parallel for
            for (int i = 0; i < valueCount; i++) {
                if (i < lastSegmentValues->size()) {
                    guesses[originalSize + i] = guess + (*lastSegmentValues)[i];
                }
            }
            
            total_guesses += valueCount;
        }
    }
}

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
        // 最初的pivot值
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
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

            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
}

void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 获取segment类型和长度
        int segmentType = pt.content[0].type;
        int segmentLength = pt.content[0].length;
        
        // 使用优化的GPU实现
        generateSingleSegmentGPU(segmentType, segmentLength, pt.max_indices[0], guesses, total_guesses);
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 构建除了最后一个segment之外的前缀
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

        // 获取最后一个segment的类型和长度
        int lastSegmentType = pt.content[pt.content.size() - 1].type;
        int lastSegmentLength = pt.content[pt.content.size() - 1].length;
        
        // 使用优化的GPU实现
        generateMultiSegmentGPU(guess, lastSegmentType, lastSegmentLength, 
                              pt.max_indices[pt.content.size() - 1], guesses, total_guesses);
    }
}

// 关闭所有工作线程
void shutdownWorkerThreads() {
    shutdownThreads = true;
    queueCV.notify_all();
    
    for (auto& thread : workerThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    workerThreads.clear();
}