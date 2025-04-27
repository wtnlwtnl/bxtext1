#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <iostream>
using namespace std;
using namespace chrono;

// Build commands:
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -msse4.1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O1 -msse4.1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O2 -msse4.1

/**
 * Process MD5 hash calculation in parallel using SSE SIMD instructions
 * @param guesses List of strings to hash
 * @param time_hash Cumulative hash calculation time
 */
void process_parallel_md5(const vector<string>& guesses, double& time_hash) {
    // Use batch processing to avoid memory issues
    const int BATCH_SIZE = 10000;  // Process 10000 passwords at a time
    
    for (size_t offset = 0; offset < guesses.size(); offset += BATCH_SIZE) {
        auto start_hash = system_clock::now();
        
        // Calculate current batch size
        int batchSize = min(BATCH_SIZE, (int)(guesses.size() - offset));
        
        // Allocate memory on the heap instead of stack
        string* passwords = new string[batchSize];
        bit32 (*states)[4] = new bit32[batchSize][4];
        
        // Copy passwords for current batch
        for (int i = 0; i < batchSize; i++) {
            passwords[i] = guesses[offset + i];
        }
        
        // Use SIMD version of MD5 function
        MD5Hash_SIMD(passwords, batchSize, states);
        
        // Free memory
        delete[] passwords;
        delete[] states;

        // Calculate hash time for this batch
        auto end_hash = system_clock::now();
        auto duration = duration_cast<microseconds>(end_hash - start_hash);
        time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
        
        // Print progress for large batches
        if (guesses.size() > BATCH_SIZE) {
            cout << "Processed " << (offset + batchSize) << " of " << guesses.size() << " passwords" << endl;
        }
    }
}

int main()
{
    // Time measurement variables
    double time_hash = 0;    // Time for MD5 hash calculation
    double time_guess = 0;   // Time for guess generation
    double time_train = 0;   // Time for model training
    double time_total = 0;   // Total runtime
    
    // Save all generated guesses for later processing
    vector<vector<string>> all_guesses;
    
    cout << "Starting model training..." << endl;
    
    // Model training phase
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "Model training complete, time: " << time_train << " seconds" << endl;
    cout << "Starting guess generation..." << endl;

    q.init();
    int curr_num = 0;
    auto start_guess = system_clock::now();
    int history = 0;  // Record total number of guesses generated
    
    // Guess generation phase
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        
        // Periodically output progress
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // Exit after generating enough guesses
            if (history + q.total_guesses > 1000000)  // Reduced threshold for testing
            {
                all_guesses.push_back(q.guesses);
                break;
            }
        }
        
        // Periodically process generated guesses to avoid excessive memory usage
        if (curr_num > 100000)  // Reduced threshold for testing
        {
            all_guesses.push_back(q.guesses);
            
            // Record total number of guesses generated
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
    
    // If there are remaining unprocessed guesses
    if (!q.guesses.empty()) {
        all_guesses.push_back(q.guesses);
        history += q.guesses.size();
    }
    
    // Calculate total guess generation time
    auto end_guess = system_clock::now();
    auto duration_guess = duration_cast<microseconds>(end_guess - start_guess);
    time_guess = double(duration_guess.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "Guess generation complete, starting MD5 hash processing..." << endl;
    
    // Process all generated guesses
    auto start_hash = system_clock::now();
    for (const auto& guesses : all_guesses) {
        process_parallel_md5(guesses, time_hash);
    }
    auto end_hash = system_clock::now();
    
    // Calculate total time (training + guessing + hashing)
    time_total = time_train + time_guess + time_hash;
    
    // Output final time statistics
    cout << "\n======= FINAL STATISTICS =======" << endl;
    cout << "Total guesses: " << history << endl;
    cout << "Guess generation time: " << time_guess << " seconds" << endl;
    cout << "Hash calculation time: " << time_hash << " seconds" << endl;
    cout << "Model training time: " << time_train << " seconds" << endl;
    cout << "Total runtime: " << time_total << " seconds" << endl;
    
    return 0;
}