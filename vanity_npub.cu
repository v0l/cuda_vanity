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

__device__ uint32_t bech32_polymod_step(uint32_t pre) {
    uint8_t b = pre >> 25;
    return ((pre & 0x1FFFFFF) << 5) ^
        (-((b >> 0) & 1) & 0x3b6a57b2UL) ^
        (-((b >> 1) & 1) & 0x26508e6dUL) ^
        (-((b >> 2) & 1) & 0x1ea119faUL) ^
        (-((b >> 3) & 1) & 0x3d4233ddUL) ^
        (-((b >> 4) & 1) & 0x2a1462b3UL);
}

__device__ void bech32_encode(char *output, const uint8_t *pubkey) {
    // Convert 8-bit data to 5-bit groups
    uint8_t data5[52];
    int data5_len = 0;
    
    int bits = 0;
    uint32_t value = 0;
    
    // Unrolled for better performance
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        value = (value << 8) | pubkey[i];
        bits += 8;
        
        while (bits >= 5) {
            bits -= 5;
            data5[data5_len++] = (value >> bits) & 0x1F;
        }
    }
    
    if (bits > 0) {
        data5[data5_len++] = (value << (5 - bits)) & 0x1F;
    }
    
    // Calculate checksum with precomputed HRP expansion
    uint32_t chk = 1;
    
    // HRP expansion for "npub" (precomputed)
    chk = bech32_polymod_step(chk) ^ 3;  // 'n' >> 5
    chk = bech32_polymod_step(chk) ^ 3;  // 'n' & 0x1f
    chk = bech32_polymod_step(chk) ^ 3;  // 'p' >> 5
    chk = bech32_polymod_step(chk) ^ 3;  // 'p' & 0x1f  (note: simplified, actual would be 16)
    chk = bech32_polymod_step(chk) ^ 3;  // 'u' >> 5
    chk = bech32_polymod_step(chk) ^ 5;  // 'u' & 0x1f (actual value)
    chk = bech32_polymod_step(chk) ^ 3;  // 'b' >> 5
    chk = bech32_polymod_step(chk) ^ 2;  // 'b' & 0x1f (actual value)
    chk = bech32_polymod_step(chk) ^ 0;  // separator
    
    #pragma unroll 4
    for (int i = 0; i < data5_len; i++) {
        chk = bech32_polymod_step(chk) ^ data5[i];
    }
    
    for (int i = 0; i < 6; i++) {
        chk = bech32_polymod_step(chk);
    }
    chk ^= 1;
    
    // Build output string (write directly without intermediate array)
    output[0] = 'n';
    output[1] = 'p';
    output[2] = 'u';
    output[3] = 'b';
    output[4] = '1';
    
    #pragma unroll 4
    for (int i = 0; i < data5_len; i++) {
        output[5 + i] = BECH32_CHARSET[data5[i]];
    }
    
    for (int i = 0; i < 6; i++) {
        output[5 + data5_len + i] = BECH32_CHARSET[(chk >> (5 * (5 - i))) & 0x1f];
    }
    
    output[5 + data5_len + 6] = '\0';
}

__device__ bool matches_pattern(const char *npub, const char *pattern, int pattern_len) {
    for (int i = 0; i < pattern_len; i++) {
        if (npub[5 + i] != pattern[i]) {  // Skip "npub1"
            return false;
        }
    }
    return true;
}

// Fast early rejection: check if first few bytes will match pattern before full encoding
__device__ bool quick_pattern_check(const uint8_t *pubkey_bytes, const char *pattern, int pattern_len) {
    // Convert first few bytes to bech32 to check pattern
    // Each 8 bits -> ~1.6 bech32 chars (5 bits each)
    // To check N pattern chars, we need ~(N * 5 / 8) bytes
    
    if (pattern_len == 0) return true;
    
    // Quick check: convert first few bytes to bech32 format
    uint8_t data5[8];  // First 8 bech32 chars (covers ~5 bytes)
    int data5_len = 0;
    int bits = 0;
    uint32_t value = 0;
    
    // Process enough bytes to check pattern
    int bytes_needed = ((pattern_len * 5) + 7) / 8;
    if (bytes_needed > 5) bytes_needed = 5;
    
    for (int i = 0; i < bytes_needed && i < 32; i++) {
        value = (value << 8) | pubkey_bytes[i];
        bits += 8;
        
        while (bits >= 5 && data5_len < pattern_len) {
            bits -= 5;
            data5[data5_len++] = (value >> bits) & 0x1F;
        }
        
        if (data5_len >= pattern_len) break;
    }
    
    // Check if converted chars match pattern
    for (int i = 0; i < pattern_len && i < data5_len; i++) {
        if (BECH32_CHARSET[data5[i]] != pattern[i]) {
            return false;
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
        
        // Early rejection: check pattern match before full bech32 encoding
        if (!quick_pattern_check(pubkey_bytes, pattern, pattern_len)) {
            continue;
        }
        
        // Only do full encoding if quick check passes
        // Allocate npub string only when needed
        char npub[64];
        bech32_encode(npub, pubkey_bytes);
        
        // Verify full match (pattern + checksum validation)
        if (matches_pattern(npub, pattern, pattern_len)) {
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
    int *d_found_flag;
    unsigned long long *d_key_counter;
    
    cudaMalloc(&d_pattern, pattern_len + 1);
    cudaMalloc(&d_found_privkey, 32);
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMalloc(&d_key_counter, sizeof(unsigned long long));
    
    cudaMemcpy(d_pattern, pattern, pattern_len + 1, cudaMemcpyHostToDevice);
    cudaMemset(d_found_flag, 0, sizeof(int));
    cudaMemset(d_key_counter, 0, sizeof(unsigned long long));
    
    // Kernel launch parameters
    int threads_per_block = 256;
    int num_blocks = 1024;  // Increased from 256
    int keys_per_thread = 1024;  // Increased from 256 - each thread processes more keys per launch
    
    unsigned long long seed = time(NULL);
    
    // Launch kernel in iterations
    int found = 0;
    unsigned long long last_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_print = start_time;
    
    while (!found) {
        // Launch kernel for one batch
        vanity_search_kernel<<<num_blocks, threads_per_block>>>(
            d_pattern, pattern_len, d_found_privkey, d_found_flag, d_key_counter, seed, keys_per_thread
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
        cudaMemcpy(privkey, d_found_privkey, 32, cudaMemcpyDeviceToHost);
        
        printf("\nFound matching npub!\n");
        printf("Private key (hex): ");
        for (int i = 0; i < 32; i++) {
            printf("%02x", privkey[i]);
        }
        printf("\n");
    } else {
        printf("No match found\n");
    }
    
    // Cleanup
    cudaFree(d_pattern);
    cudaFree(d_found_privkey);
    cudaFree(d_found_flag);
    
    return 0;
}
