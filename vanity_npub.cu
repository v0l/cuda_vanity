#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <chrono>
#include <thread>

#include "secp256k1_jacobian.cuh"

// Bech32 charset (in constant memory for cache efficiency)
__constant__ char BECH32_CHARSET[33] = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

// Partial bech32 encode - only encode enough characters to check pattern
// Skips the constant "npub1" prefix and encodes only the data portion
// Returns false if pattern doesn't match, true if it matches
__device__ bool bech32_encode_and_check_pattern(const uint8_t *pubkey, const char *pattern, int pattern_len) {
    // We skip encoding the HRP "npub1" since it's constant
    // Just encode the data portion and check against pattern
    
    int bits = 0;
    uint32_t value = 0;
    int chars_encoded = 0;
    
    // Process bytes until we have enough 5-bit groups for the pattern
    for (int i = 0; i < 32 && chars_encoded < pattern_len; i++) {
        value = (value << 8) | pubkey[i];
        bits += 8;
        
        while (bits >= 5 && chars_encoded < pattern_len) {
            bits -= 5;
            uint8_t data5 = (value >> bits) & 0x1F;
            
            // Check if this character matches the pattern
            if (BECH32_CHARSET[data5] != pattern[chars_encoded]) {
                return false;
            }
            chars_encoded++;
        }
    }
    
    return true;
}

// Kernel to initialize precomputed table
__global__ void init_precomp_table(PointJ *table_out) {
    int idx = threadIdx.x;
    if (idx >= PRECOMP_TABLE_SIZE) return;
    
    // Get generator point G
    Big256 Gx, Gy;
    get_curve_G(Gx, Gy);
    PointJ G;
    to_jacobian(Gx, Gy, G);
    
    // Compute (idx+1) * G
    if (idx == 0) {
        // 1G = G
        table_out[0] = G;
    } else {
        // Compute by repeated addition
        PointJ result = G;
        for (int i = 0; i < idx; i++) {
            jacobian_add(result, G, result);
        }
        table_out[idx] = result;
    }
}

// Optimized scalar multiplication using windowed method with precomputed table
__device__ void scalarMultG_fast(const unsigned int privkey[8], unsigned int pubX[8], unsigned int pubY[8]) {
    // Convert privkey to Big256 format
    Big256 scalar;
    convert_to_big256(privkey, scalar);
    
    // Perform windowed scalar multiplication using precomputed table
    PointJ result;
    scalar_mul_windowed(scalar, result);
    
    // Convert back to affine coordinates
    Big256 ax, ay;
    from_jacobian(result, ax, ay);
    
    // Convert to our output format (big-endian unsigned int[8])
    convert_from_big256(ax, pubX);
    convert_from_big256(ay, pubY);
}


__global__ void vanity_search_kernel(
    const char *pattern,
    int pattern_len,
    uint8_t *found_privkey,
    uint8_t *found_pubkey,
    int *found_flag,
    unsigned long long *key_counter,
    unsigned long long seed,
    int keys_per_thread
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if already found
    if (*found_flag) return;
    
    // Initialize random state
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    unsigned int privkey[8];
    unsigned int pubX[8];
    unsigned int pubY[8];
    uint8_t pubkey_bytes[32];
    
    int keys_processed = 0;
    
    // Process a fixed number of keys per kernel launch
    for (int iter = 0; iter < keys_per_thread && !(*found_flag); iter++) {
        keys_processed++;
        
        // Generate random 32-byte private key
        for (int i = 0; i < 8; i++) {
            privkey[i] = curand(&state);
        }
        
        // Calculate public key using optimized Jacobian coordinates
        scalarMultG_fast(privkey, pubX, pubY);
        
        // Convert public key X coordinate to bytes (big-endian)
        for (int i = 0; i < 8; i++) {
            pubkey_bytes[i * 4 + 0] = (pubX[i] >> 24) & 0xFF;
            pubkey_bytes[i * 4 + 1] = (pubX[i] >> 16) & 0xFF;
            pubkey_bytes[i * 4 + 2] = (pubX[i] >> 8) & 0xFF;
            pubkey_bytes[i * 4 + 3] = pubX[i] & 0xFF;
        }
        
        // Partial encoding: only encode enough to check the pattern
        if (bech32_encode_and_check_pattern(pubkey_bytes, pattern, pattern_len)) {
            // Found a match!
            int old = atomicCAS(found_flag, 0, 1);
            if (old == 0) {
                // Convert privkey to bytes (big-endian)
                for (int i = 0; i < 8; i++) {
                    found_privkey[i * 4 + 0] = (privkey[i] >> 24) & 0xFF;
                    found_privkey[i * 4 + 1] = (privkey[i] >> 16) & 0xFF;
                    found_privkey[i * 4 + 2] = (privkey[i] >> 8) & 0xFF;
                    found_privkey[i * 4 + 3] = privkey[i] & 0xFF;
                }
                // Save pubkey (already in pubkey_bytes)
                for (int i = 0; i < 32; i++) {
                    found_pubkey[i] = pubkey_bytes[i];
                }
            }
            return;
        }
    }
    
    // Update counter once at the end instead of every iteration
    atomicAdd(key_counter, (unsigned long long)keys_processed);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <pattern>\n", argv[0]);
        printf("Example: %s abc\n", argv[0]);
        return 1;
    }
    
    const char *pattern = argv[1];
    int pattern_len = strlen(pattern);
    
    // Validate pattern contains only valid bech32 characters
    const char *bech32_chars = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    for (int i = 0; i < pattern_len; i++) {
        bool valid = false;
        for (int j = 0; j < 32; j++) {
            if (pattern[i] == bech32_chars[j]) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            printf("Error: Invalid character '%c' in pattern.\n", pattern[i]);
            printf("Valid bech32 characters are: %s\n", bech32_chars);
            printf("Note: bech32 does NOT contain: b, i, o, 1\n");
            return 1;
        }
    }
    
    printf("Searching for npub starting with: npub1%s\n", pattern);
    
    // Compute and upload precomputed table for G multiples
    printf("Initializing precomputed point table...\n");
    
    PointJ *d_table_temp;
    cudaMalloc(&d_table_temp, sizeof(PointJ) * PRECOMP_TABLE_SIZE);
    
    // Compute table on GPU
    init_precomp_table<<<1, PRECOMP_TABLE_SIZE>>>(d_table_temp);
    cudaDeviceSynchronize();
    
    // Copy to constant memory
    PrecompTable h_table;
    cudaMemcpy(&h_table, d_table_temp, sizeof(PrecompTable), cudaMemcpyDeviceToHost);
    
    // Get symbol address and copy to constant memory
    void *table_symbol;
    cudaGetSymbolAddress(&table_symbol, G_precomp_table);
    cudaMemcpy(table_symbol, &h_table, sizeof(PrecompTable), cudaMemcpyHostToDevice);
    
    cudaFree(d_table_temp);
    printf("Precomputed table initialization complete.\n");
    
    // Allocate device memory
    char *d_pattern;
    uint8_t *d_found_privkey;
    uint8_t *d_found_pubkey;
    int *d_found_flag;
    unsigned long long *d_key_counter;
    
    cudaMalloc(&d_pattern, pattern_len + 1);
    cudaMalloc(&d_found_privkey, 32);
    cudaMalloc(&d_found_pubkey, 32);
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMalloc(&d_key_counter, sizeof(unsigned long long));
    
    cudaMemcpy(d_pattern, pattern, pattern_len + 1, cudaMemcpyHostToDevice);
    cudaMemset(d_found_flag, 0, sizeof(int));
    cudaMemset(d_key_counter, 0, sizeof(unsigned long long));
    
    // Kernel launch parameters - optimized per architecture
    int threads_per_block;
    int num_blocks;
    int keys_per_thread;
    
    // Get device properties to optimize configuration
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int compute_capability = prop.major * 10 + prop.minor;
    
    if (compute_capability >= 120) {
        // Blackwell (sm_120+): More SMs, more registers available
        // GB10 has 128 SMs, can handle more parallelism
        threads_per_block = 256;
        num_blocks = 2048;  // Double blocks for more SMs
        keys_per_thread = 512;  // Reduce per-thread work for faster iterations
    } else if (compute_capability >= 89) {
        // Ada Lovelace (sm_89): RTX 40-series
        threads_per_block = 256;
        num_blocks = 1536;
        keys_per_thread = 768;
    } else if (compute_capability >= 86) {
        // Ampere (sm_86): RTX 30-series
        threads_per_block = 256;
        num_blocks = 1280;
        keys_per_thread = 896;
    } else if (compute_capability >= 75) {
        // Turing (sm_75): RTX 20-series
        threads_per_block = 256;
        num_blocks = 1024;
        keys_per_thread = 1024;
    } else {
        // Pascal and older (sm_61 and below): Limited registers
        threads_per_block = 256;
        num_blocks = 1024;
        keys_per_thread = 1024;
    }
    
    printf("GPU: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Configuration: %d blocks × %d threads × %d keys/thread = %llu keys/batch\n",
           num_blocks, threads_per_block, keys_per_thread,
           (unsigned long long)num_blocks * threads_per_block * keys_per_thread);
    
    unsigned long long seed = time(NULL);
    
    // Launch kernel in iterations
    int found = 0;
    unsigned long long last_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_print = start_time;
    
    while (!found) {
        // Launch kernel for one batch
        vanity_search_kernel<<<num_blocks, threads_per_block>>>(
            d_pattern, pattern_len, d_found_privkey, d_found_pubkey, d_found_flag, d_key_counter, seed, keys_per_thread
        );
        
        // Wait for this batch to complete
        cudaDeviceSynchronize();
        
        // Check if found
        cudaMemcpy(&found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Update seed for next iteration
        seed++;
        
        // Print progress
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print).count();
        
        if (elapsed >= 1000) {  // Print every second
            unsigned long long current_count;
            cudaMemcpy(&current_count, d_key_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            
            unsigned long long keys_generated = current_count - last_count;
            double keys_per_sec = keys_generated / (elapsed / 1000.0);
            
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            // Clear line with spaces before printing to avoid overlap
            printf("\r%-80s\r[%lds] Generated: %llu keys | Speed: %.2f keys/s", 
                   "", (long)total_elapsed, current_count, keys_per_sec);
            fflush(stdout);
            
            last_count = current_count;
            last_print = now;
        }
    }
    
    // Calculate final statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    unsigned long long total_keys;
    cudaMemcpy(&total_keys, d_key_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    double total_seconds = total_duration / 1000.0;
    double avg_keys_per_sec = total_keys / total_seconds;
    
    printf("\nTotal: %llu keys in %.2fs (avg %.2f keys/s)\n", 
           total_keys, total_seconds, avg_keys_per_sec);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Display result
    if (found) {
        uint8_t privkey[32];
        uint8_t pubkey[32];
        cudaMemcpy(privkey, d_found_privkey, 32, cudaMemcpyDeviceToHost);
        cudaMemcpy(pubkey, d_found_pubkey, 32, cudaMemcpyDeviceToHost);
        
        printf("\nFound matching npub!\n\n");
        
        printf("Private key (hex): ");
        for (int i = 0; i < 32; i++) {
            printf("%02x", privkey[i]);
        }
        printf("\n");
        
        printf("Public key (hex):  ");
        for (int i = 0; i < 32; i++) {
            printf("%02x", pubkey[i]);
        }
        printf("\n");
    } else {
        printf("No match found\n");
    }
    
    // Cleanup
    cudaFree(d_pattern);
    cudaFree(d_found_privkey);
    cudaFree(d_found_pubkey);
    cudaFree(d_found_flag);
    cudaFree(d_key_counter);
    
    return 0;
}
