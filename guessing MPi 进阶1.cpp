#include "PCFG.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PT的概率
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
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

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
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
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

// 序列化和反序列化函数
void serialize_strings(const vector<string>& strings, vector<char>& buffer, vector<int>& sizes) {
    sizes.resize(strings.size());
    int total_size = 0;
    
    for (size_t i = 0; i < strings.size(); i++) {
        sizes[i] = strings[i].size();
        total_size += sizes[i];
    }
    
    buffer.resize(total_size);
    
    size_t pos = 0;
    for (const auto& str : strings) {
        copy(str.begin(), str.end(), buffer.begin() + pos);
        pos += str.size();
    }
}

vector<string> deserialize_strings(const vector<char>& buffer, const vector<int>& sizes) {
    vector<string> result;
    size_t pos = 0;
    
    for (int size : sizes) {
        result.push_back(string(buffer.begin() + pos, buffer.begin() + pos + size));
        pos += size;
    }
    
    return result;
}

// PT序列化函数 - 修复版本
void serialize_pt(const PT& pt, vector<char>& buffer) {
    // 1. 计算需要的缓冲区大小
    int buffer_size = sizeof(int);  // pivot
    buffer_size += sizeof(float) * 2;   // prob和preterm_prob
    
    // content大小
    buffer_size += sizeof(int);  // content的size
    buffer_size += pt.content.size() * sizeof(int) * 2;  // 每个segment有type和length两个int字段
    
    // max_indices和curr_indices大小
    buffer_size += sizeof(int) * 2;  // 两个向量的size
    buffer_size += pt.max_indices.size() * sizeof(int);
    buffer_size += pt.curr_indices.size() * sizeof(int);
    
    // 2. 分配缓冲区
    buffer.resize(buffer_size);
    
    // 3. 写入数据
    int pos = 0;
    
    // 基本属性
    *(int*)(&buffer[pos]) = pt.pivot;
    pos += sizeof(int);
    *(float*)(&buffer[pos]) = pt.prob;
    pos += sizeof(float);
    *(float*)(&buffer[pos]) = pt.preterm_prob;
    pos += sizeof(float);
    
    // content
    *(int*)(&buffer[pos]) = pt.content.size();
    pos += sizeof(int);
    for (const auto& seg : pt.content) {
        *(int*)(&buffer[pos]) = seg.type;
        pos += sizeof(int);
        *(int*)(&buffer[pos]) = seg.length;
        pos += sizeof(int);
    }
    
    // max_indices
    *(int*)(&buffer[pos]) = pt.max_indices.size();
    pos += sizeof(int);
    for (int val : pt.max_indices) {
        *(int*)(&buffer[pos]) = val;
        pos += sizeof(int);
    }
    
    // curr_indices
    *(int*)(&buffer[pos]) = pt.curr_indices.size();
    pos += sizeof(int);
    for (int val : pt.curr_indices) {
        *(int*)(&buffer[pos]) = val;
        pos += sizeof(int);
    }
}

// PT反序列化函数 - 修复版本
PT deserialize_pt(const vector<char>& buffer) {
    PT result;
    int pos = 0;
    
    // 基本属性
    result.pivot = *(int*)(&buffer[pos]);
    pos += sizeof(int);
    result.prob = *(float*)(&buffer[pos]);
    pos += sizeof(float);
    result.preterm_prob = *(float*)(&buffer[pos]);
    pos += sizeof(float);
    
    // content
    int content_size = *(int*)(&buffer[pos]);
    pos += sizeof(int);
    for (int i = 0; i < content_size; i++) {
        int type = *(int*)(&buffer[pos]);
        pos += sizeof(int);
        int length = *(int*)(&buffer[pos]);
        pos += sizeof(int);
        
        segment seg(type, length);
        result.content.push_back(seg);
    }
    
    // max_indices
    int max_indices_size = *(int*)(&buffer[pos]);
    pos += sizeof(int);
    for (int i = 0; i < max_indices_size; i++) {
        result.max_indices.push_back(*(int*)(&buffer[pos]));
        pos += sizeof(int);
    }
    
    // curr_indices
    int curr_indices_size = *(int*)(&buffer[pos]);
    pos += sizeof(int);
    for (int i = 0; i < curr_indices_size; i++) {
        result.curr_indices.push_back(*(int*)(&buffer[pos]));
        pos += sizeof(int);
    }
    
    return result;
}

// 处理单个PT的密码生成函数 - 修复版本
vector<string> ProcessPT(PT& pt, PriorityQueue& q) {
    vector<string> local_guesses;
    
    if (pt.content.size() == 1) {
        // 处理只有一个segment的情况
        segment *a = nullptr;
        if (pt.content[0].type == 1)
            a = &q.m.letters[q.m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &q.m.digits[q.m.FindDigit(pt.content[0])];
        else
            a = &q.m.symbols[q.m.FindSymbol(pt.content[0])];
        
        // 生成所有可能的密码
        for (int i = 0; i < pt.max_indices[0]; i++) {
            local_guesses.push_back(a->ordered_values[i]);
        }
    } else {
        // 处理多个segment的情况
        string prefix;
        int seg_idx = 0;
        
        // 构建除最后一个segment外的前缀
        for (int idx : pt.curr_indices) {
            if (seg_idx == (int)pt.content.size() - 1) break;

            const segment &segInfo = pt.content[seg_idx];
            if (segInfo.type == 1)
                prefix += q.m.letters[q.m.FindLetter(segInfo)].ordered_values[idx];
            else if (segInfo.type == 2)
                prefix += q.m.digits[q.m.FindDigit(segInfo)].ordered_values[idx];
            else
                prefix += q.m.symbols[q.m.FindSymbol(segInfo)].ordered_values[idx];

            ++seg_idx;
        }

        // 处理最后一个segment
        segment *a = nullptr;
        const segment &lastSegInfo = pt.content.back();
        if (lastSegInfo.type == 1)
            a = &q.m.letters[q.m.FindLetter(lastSegInfo)];
        else if (lastSegInfo.type == 2)
            a = &q.m.digits[q.m.FindDigit(lastSegInfo)];
        else
            a = &q.m.symbols[q.m.FindSymbol(lastSegInfo)];
        
        // 生成所有可能的密码
        for (int i = 0; i < pt.max_indices.back(); i++) {
            local_guesses.push_back(prefix + a->ordered_values[i]);
        }
    }
    
    return local_guesses;
}

// PT生成新PT的函数
vector<PT> PT::NewPTs()
{
    vector<PT> res;

    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;

        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;

            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }

            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
}

// 实现PT层面的MPI并行处理 - 修复版本
void PriorityQueue::PopNext() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 确定要处理的PT数量
    int num_pts_to_process = min(size, static_cast<int>(priority.size()));
    if (num_pts_to_process == 0 && priority.size() > 0) {
        num_pts_to_process = 1;  // 至少处理一个PT
    }
    
    // 广播要处理的PT数量
    MPI_Bcast(&num_pts_to_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 如果没有PT可处理，直接返回
    if (num_pts_to_process == 0) {
        return;
    }
    
    // rank 0负责分配PT到各进程
    vector<PT> local_pts;
    vector<vector<char>> serialized_pts(num_pts_to_process);
    vector<int> buffer_sizes(num_pts_to_process);
    
    if (rank == 0) {
        // 从队列中取出PT并序列化
        for (int i = 0; i < num_pts_to_process && !priority.empty(); i++) {
            PT current_pt = priority.front();
            priority.erase(priority.begin());
            
            serialize_pt(current_pt, serialized_pts[i]);
            buffer_sizes[i] = serialized_pts[i].size();
        }
    }
    
    // 广播每个PT的缓冲区大小
    MPI_Bcast(buffer_sizes.data(), num_pts_to_process, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 确定每个进程要处理的PT
    int pts_per_proc = num_pts_to_process / size;
    int remainder = num_pts_to_process % size;
    
    int start_idx = rank * pts_per_proc + min(rank, remainder);
    int end_idx = start_idx + pts_per_proc + (rank < remainder ? 1 : 0);
    
    // 接收分配给当前进程的PT
    for (int i = 0; i < num_pts_to_process; i++) {
        // 所有进程都需要接收buffer_size
        if (rank != 0) {
            serialized_pts[i].resize(buffer_sizes[i]);
        }
        
        // 所有进程参与广播
        MPI_Bcast(serialized_pts[i].data(), buffer_sizes[i], MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // 但只有负责处理此PT的进程执行实际处理
        if (i >= start_idx && i < end_idx) {
            PT current_pt = deserialize_pt(serialized_pts[i]);
            local_pts.push_back(current_pt);
        }
    }
    
    // 处理本地分配的所有PT
    vector<string> all_local_guesses;
    vector<PT> all_new_pts;
    
    for (PT& pt : local_pts) {
        // 生成密码
        vector<string> pt_guesses = ProcessPT(pt, *this);
        all_local_guesses.insert(all_local_guesses.end(), pt_guesses.begin(), pt_guesses.end());
        
        // 生成新的PT
        vector<PT> new_pts = pt.NewPTs();
        for (PT& new_pt : new_pts) {
            CalProb(new_pt);
        }
        all_new_pts.insert(all_new_pts.end(), new_pts.begin(), new_pts.end());
    }
    
    // 统计本地生成的密码数量和新PT数量
    int local_guesses_count = all_local_guesses.size();
    int local_new_pts_count = all_new_pts.size();
    
    // 收集所有进程的密码数量和新PT数量
    vector<int> all_guesses_counts(size);
    vector<int> all_new_pts_counts(size);
    
    MPI_Gather(&local_guesses_count, 1, MPI_INT, all_guesses_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_new_pts_count, 1, MPI_INT, all_new_pts_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 序列化本地猜测和新PT
    vector<char> guesses_buffer;
    vector<int> guesses_sizes;
    serialize_strings(all_local_guesses, guesses_buffer, guesses_sizes);
    
    // 序列化新PT
    vector<char> new_pts_buffer;
    for (PT& pt : all_new_pts) {
        vector<char> pt_buffer;
        serialize_pt(pt, pt_buffer);
        
        int buffer_size = pt_buffer.size();
        new_pts_buffer.insert(new_pts_buffer.end(), (char*)&buffer_size, (char*)&buffer_size + sizeof(int));
        new_pts_buffer.insert(new_pts_buffer.end(), pt_buffer.begin(), pt_buffer.end());
    }
    
    // 收集各进程的序列化数据大小
    int local_guesses_buffer_size = guesses_buffer.size();
    int local_guesses_sizes_size = guesses_sizes.size();
    int local_new_pts_buffer_size = new_pts_buffer.size();
    
    vector<int> guesses_buffer_sizes(size);
    vector<int> guesses_sizes_sizes(size);
    vector<int> new_pts_buffer_sizes(size);
    
    MPI_Gather(&local_guesses_buffer_size, 1, MPI_INT, guesses_buffer_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_guesses_sizes_size, 1, MPI_INT, guesses_sizes_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_new_pts_buffer_size, 1, MPI_INT, new_pts_buffer_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 计算位移
    vector<int> guesses_buffer_displacements(size);
    vector<int> guesses_sizes_displacements(size);
    vector<int> new_pts_buffer_displacements(size);
    
    if (rank == 0) {
        guesses_buffer_displacements[0] = 0;
        guesses_sizes_displacements[0] = 0;
        new_pts_buffer_displacements[0] = 0;
        
        for (int i = 1; i < size; i++) {
            guesses_buffer_displacements[i] = guesses_buffer_displacements[i-1] + guesses_buffer_sizes[i-1];
            guesses_sizes_displacements[i] = guesses_sizes_displacements[i-1] + guesses_sizes_sizes[i-1];
            new_pts_buffer_displacements[i] = new_pts_buffer_displacements[i-1] + new_pts_buffer_sizes[i-1];
        }
    }
    
    // 准备接收缓冲区
    int total_guesses_buffer_size = 0;
    int total_guesses_sizes_size = 0;
    int total_new_pts_buffer_size = 0;
    
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            total_guesses_buffer_size += guesses_buffer_sizes[i];
            total_guesses_sizes_size += guesses_sizes_sizes[i];
            total_new_pts_buffer_size += new_pts_buffer_sizes[i];
        }
    }
    
    vector<char> recv_guesses_buffer(total_guesses_buffer_size);
    vector<int> recv_guesses_sizes(total_guesses_sizes_size);
    vector<char> recv_new_pts_buffer(total_new_pts_buffer_size);
    
    // 收集所有序列化的数据
    MPI_Gatherv(guesses_buffer.data(), guesses_buffer.size(), MPI_CHAR,
               recv_guesses_buffer.data(), guesses_buffer_sizes.data(), guesses_buffer_displacements.data(),
               MPI_CHAR, 0, MPI_COMM_WORLD);
    
    MPI_Gatherv(guesses_sizes.data(), guesses_sizes.size(), MPI_INT,
               recv_guesses_sizes.data(), guesses_sizes_sizes.data(), guesses_sizes_displacements.data(),
               MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Gatherv(new_pts_buffer.data(), new_pts_buffer.size(), MPI_CHAR,
               recv_new_pts_buffer.data(), new_pts_buffer_sizes.data(), new_pts_buffer_displacements.data(),
               MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // 只在主进程中进行反序列化和结果合并
    if (rank == 0) {
        // 反序列化猜测
        int total_guesses_count = 0;
        for (int count : all_guesses_counts) {
            total_guesses_count += count;
        }
        
        // 将所有猜测解析并添加到结果中
        int pos = 0;
        for (int i = 0; i < size; i++) {
            if (all_guesses_counts[i] > 0) {
                vector<string> proc_guesses = deserialize_strings(
                    vector<char>(recv_guesses_buffer.begin() + guesses_buffer_displacements[i],
                                recv_guesses_buffer.begin() + guesses_buffer_displacements[i] + guesses_buffer_sizes[i]),
                    vector<int>(recv_guesses_sizes.begin() + guesses_sizes_displacements[i],
                               recv_guesses_sizes.begin() + guesses_sizes_displacements[i] + all_guesses_counts[i])
                );
                
                guesses.insert(guesses.end(), proc_guesses.begin(), proc_guesses.end());
            }
        }
        
        // 更新总计数
        total_guesses += total_guesses_count;
        
        // 反序列化新PT并插入到优先队列
        pos = 0;
        while (pos < total_new_pts_buffer_size) {
            int buffer_size = *(int*)(&recv_new_pts_buffer[pos]);
            pos += sizeof(int);
            
            vector<char> pt_buffer(recv_new_pts_buffer.begin() + pos, 
                                 recv_new_pts_buffer.begin() + pos + buffer_size);
            pos += buffer_size;
            
            PT new_pt = deserialize_pt(pt_buffer);
            
            // 将新PT插入优先队列
            for (auto iter = priority.begin(); iter != priority.end(); iter++) {
                if (iter != priority.end() - 1 && iter != priority.begin()) {
                    if (new_pt.prob <= iter->prob && new_pt.prob > (iter + 1)->prob) {
                        priority.emplace(iter + 1, new_pt);
                        break;
                    }
                }
                if (iter == priority.end() - 1) {
                    priority.emplace_back(new_pt);
                    break;
                }
                if (iter == priority.begin() && iter->prob < new_pt.prob) {
                    priority.emplace(iter, new_pt);
                    break;
                }
            }
            
            if (priority.empty()) {
                priority.push_back(new_pt);
            }
        }
    }
    
    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);
}