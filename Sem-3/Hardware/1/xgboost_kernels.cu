
/**
 * XGBoost Parallel CUDA Kernels
 * 
 * Compilation: nvcc -O3 -arch=sm_75 xgboost_kernels.cu -o xgboost_inference
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// CONFIGURATION
// ============================================================================
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_SHARED_TREES 32      // Trees loaded into shared memory

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Tree Node - Array of Structures (AoS) format
struct TreeNode {
    int feature_id;      // Feature index for split (-1 if leaf)
    float threshold;     // Split threshold
    int left_child;      // Left child index
    int right_child;     // Right child index  
    float leaf_value;    // Prediction value (leaf only)
    int is_leaf;         // 1 if leaf, 0 otherwise
};

// Structure of Arrays (SoA) format for better coalescing
struct TreeNodeSoA {
    int* feature_ids;
    float* thresholds;
    int* left_children;
    int* right_children;
    float* leaf_values;
    int* is_leaf;
};

// ============================================================================
// KERNEL 1: Basic Inference (Sample-Level Parallelism)
// Each thread processes one sample through ALL trees
// ============================================================================
__global__ void xgboost_inference_basic(
    const float* __restrict__ features,      // [num_samples x num_features]
    const TreeNode* __restrict__ nodes,       // All tree nodes
    const int* __restrict__ tree_offsets,     // Start index of each tree
    float* __restrict__ predictions,          // Output predictions
    int num_samples,
    int num_features,
    int num_trees,
    float base_score
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= num_samples) return;

    // Get pointer to this sample's features
    const float* sample_features = features + sample_idx * num_features;

    // Sum predictions from all trees
    float sum = base_score;

    for (int tree_id = 0; tree_id < num_trees; tree_id++) {
        int node_idx = tree_offsets[tree_id];  // Start at root

        // Traverse tree until leaf
        while (!nodes[node_idx].is_leaf) {
            int feat_id = nodes[node_idx].feature_id;
            float feat_val = sample_features[feat_id];
            float thresh = nodes[node_idx].threshold;

            // Branch left or right based on feature value
            if (feat_val < thresh) {
                node_idx = nodes[node_idx].left_child;
            } else {
                node_idx = nodes[node_idx].right_child;
            }
        }

        // Add leaf value to sum
        sum += nodes[node_idx].leaf_value;
    }

    // Apply sigmoid for binary classification
    predictions[sample_idx] = 1.0f / (1.0f + expf(-sum));
}

// ============================================================================
// KERNEL 2: Optimized Inference with Structure of Arrays (SoA)
// Better memory coalescing for tree node access
// ============================================================================
__global__ void xgboost_inference_soa(
    const float* __restrict__ features,
    const int* __restrict__ feature_ids,
    const float* __restrict__ thresholds,
    const int* __restrict__ left_children,
    const int* __restrict__ right_children,
    const float* __restrict__ leaf_values,
    const int* __restrict__ is_leaf,
    const int* __restrict__ tree_offsets,
    float* __restrict__ predictions,
    int num_samples,
    int num_features,
    int num_trees,
    float base_score
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= num_samples) return;

    const float* sample_features = features + sample_idx * num_features;
    float sum = base_score;

    for (int tree_id = 0; tree_id < num_trees; tree_id++) {
        int node_idx = tree_offsets[tree_id];

        while (!is_leaf[node_idx]) {
            float feat_val = sample_features[feature_ids[node_idx]];
            node_idx = (feat_val < thresholds[node_idx]) 
                       ? left_children[node_idx] 
                       : right_children[node_idx];
        }

        sum += leaf_values[node_idx];
    }

    predictions[sample_idx] = 1.0f / (1.0f + expf(-sum));
}

// ============================================================================
// KERNEL 3: Shared Memory Optimization
// Load tree structure into shared memory for faster access
// ============================================================================
__global__ void xgboost_inference_shared(
    const float* __restrict__ features,
    const TreeNode* __restrict__ nodes,
    const int* __restrict__ tree_offsets,
    const int* __restrict__ tree_sizes,
    float* __restrict__ predictions,
    int num_samples,
    int num_features,
    int num_trees,
    int max_tree_size,
    float base_score
) {
    // Shared memory for tree nodes
    extern __shared__ TreeNode shared_tree[];

    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float sum = base_score;

    // Process trees in batches that fit in shared memory
    for (int tree_batch = 0; tree_batch < num_trees; tree_batch++) {
        int tree_offset = tree_offsets[tree_batch];
        int tree_size = tree_sizes[tree_batch];

        // Cooperatively load tree into shared memory
        for (int i = tid; i < tree_size; i += blockDim.x) {
            shared_tree[i] = nodes[tree_offset + i];
        }
        __syncthreads();

        // Each thread traverses the tree for its sample
        if (sample_idx < num_samples) {
            const float* sample_features = features + sample_idx * num_features;
            int node_idx = 0;  // Start at root (relative to shared memory)

            while (!shared_tree[node_idx].is_leaf) {
                float feat_val = sample_features[shared_tree[node_idx].feature_id];
                node_idx = (feat_val < shared_tree[node_idx].threshold)
                           ? shared_tree[node_idx].left_child - tree_offset
                           : shared_tree[node_idx].right_child - tree_offset;
            }

            sum += shared_tree[node_idx].leaf_value;
        }
        __syncthreads();
    }

    if (sample_idx < num_samples) {
        predictions[sample_idx] = 1.0f / (1.0f + expf(-sum));
    }
}

// ============================================================================
// KERNEL 4: Tree-Level Parallelism
// Each thread block processes one sample, threads handle different trees
// Best for models with many trees
// ============================================================================
__global__ void xgboost_inference_tree_parallel(
    const float* __restrict__ features,
    const TreeNode* __restrict__ nodes,
    const int* __restrict__ tree_offsets,
    float* __restrict__ predictions,
    int num_samples,
    int num_features,
    int num_trees,
    float base_score
) {
    __shared__ float tree_results[BLOCK_SIZE];

    int sample_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (sample_idx >= num_samples) return;

    const float* sample_features = features + sample_idx * num_features;

    // Each thread handles subset of trees
    float local_sum = 0.0f;

    for (int tree_id = tid; tree_id < num_trees; tree_id += blockDim.x) {
        int node_idx = tree_offsets[tree_id];

        while (!nodes[node_idx].is_leaf) {
            float feat_val = sample_features[nodes[node_idx].feature_id];
            node_idx = (feat_val < nodes[node_idx].threshold)
                       ? nodes[node_idx].left_child
                       : nodes[node_idx].right_child;
        }

        local_sum += nodes[node_idx].leaf_value;
    }

    tree_results[tid] = local_sum;
    __syncthreads();

    // Parallel reduction to sum all tree outputs
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tree_results[tid] += tree_results[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (tid == 0) {
        float sum = base_score + tree_results[0];
        predictions[sample_idx] = 1.0f / (1.0f + expf(-sum));
    }
}

// ============================================================================
// KERNEL 5: Warp-Level Optimization with Shuffle Instructions
// Uses warp shuffle for efficient reduction
// ============================================================================
__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void xgboost_inference_warp_optimized(
    const float* __restrict__ features,
    const TreeNode* __restrict__ nodes,
    const int* __restrict__ tree_offsets,
    float* __restrict__ predictions,
    int num_samples,
    int num_features,
    int num_trees,
    float base_score
) {
    int sample_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];

    if (sample_idx >= num_samples) return;

    const float* sample_features = features + sample_idx * num_features;
    float local_sum = 0.0f;

    // Each thread processes multiple trees
    for (int tree_id = tid; tree_id < num_trees; tree_id += blockDim.x) {
        int node_idx = tree_offsets[tree_id];

        while (!nodes[node_idx].is_leaf) {
            float feat_val = sample_features[nodes[node_idx].feature_id];
            node_idx = (feat_val < nodes[node_idx].threshold)
                       ? nodes[node_idx].left_child
                       : nodes[node_idx].right_child;
        }

        local_sum += nodes[node_idx].leaf_value;
    }

    // Warp-level reduction
    local_sum = warpReduceSum(local_sum);

    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        local_sum = (tid < blockDim.x / WARP_SIZE) ? warp_sums[tid] : 0.0f;
        local_sum = warpReduceSum(local_sum);

        if (tid == 0) {
            float sum = base_score + local_sum;
            predictions[sample_idx] = 1.0f / (1.0f + expf(-sum));
        }
    }
}

// ============================================================================
// TRAINING KERNELS
// ============================================================================

// Kernel for building gradient histograms (training phase)
__global__ void build_gradient_histogram(
    const float* __restrict__ features,
    const float* __restrict__ gradients,
    const float* __restrict__ hessians,
    const int* __restrict__ sample_indices,  // Samples in current node
    float* __restrict__ hist_grad,           // Histogram gradients
    float* __restrict__ hist_hess,           // Histogram hessians
    int* __restrict__ hist_count,            // Histogram counts
    int num_samples,
    int feature_id,
    int num_bins,
    float min_val,
    float bin_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_samples) return;

    int sample_idx = sample_indices[idx];
    float feat_val = features[sample_idx * gridDim.y + feature_id];

    // Compute bin index
    int bin = min((int)((feat_val - min_val) / bin_width), num_bins - 1);
    bin = max(bin, 0);

    // Atomic accumulation into histogram
    atomicAdd(&hist_grad[bin], gradients[sample_idx]);
    atomicAdd(&hist_hess[bin], hessians[sample_idx]);
    atomicAdd(&hist_count[bin], 1);
}

// Kernel for finding best split across features (parallel over features)
__global__ void find_best_split(
    const float* __restrict__ hist_grad,     // [num_features x num_bins]
    const float* __restrict__ hist_hess,
    const int* __restrict__ hist_count,
    float* __restrict__ split_gains,         // Output: best gain per feature
    int* __restrict__ split_bins,            // Output: best bin per feature
    int num_features,
    int num_bins,
    float lambda,                            // L2 regularization
    float gamma,                             // Min split gain
    float total_grad,                        // Total gradient in node
    float total_hess                         // Total hessian in node
) {
    int feature_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature_id >= num_features) return;

    float best_gain = 0.0f;
    int best_bin = -1;

    float left_grad = 0.0f;
    float left_hess = 0.0f;

    // Scan through bins to find best split
    for (int bin = 0; bin < num_bins - 1; bin++) {
        int hist_idx = feature_id * num_bins + bin;

        left_grad += hist_grad[hist_idx];
        left_hess += hist_hess[hist_idx];

        float right_grad = total_grad - left_grad;
        float right_hess = total_hess - left_hess;

        // Skip if either child would be too small
        if (left_hess < 1.0f || right_hess < 1.0f) continue;

        // Compute split gain
        float gain = (left_grad * left_grad) / (left_hess + lambda)
                   + (right_grad * right_grad) / (right_hess + lambda)
                   - (total_grad * total_grad) / (total_hess + lambda)
                   - gamma;

        if (gain > best_gain) {
            best_gain = gain;
            best_bin = bin;
        }
    }

    split_gains[feature_id] = best_gain;
    split_bins[feature_id] = best_bin;
}

// ============================================================================
// HOST WRAPPER FUNCTIONS
// ============================================================================

// Launch inference kernel with automatic configuration
void launch_xgboost_inference(
    float* d_features,
    TreeNode* d_nodes,
    int* d_tree_offsets,
    float* d_predictions,
    int num_samples,
    int num_features,
    int num_trees,
    float base_score
) {
    // Choose kernel based on model characteristics
    if (num_trees > 256) {
        // Many trees: use tree-level parallelism
        dim3 grid(num_samples);
        dim3 block(256);
        xgboost_inference_tree_parallel<<<grid, block>>>(
            d_features, d_nodes, d_tree_offsets, d_predictions,
            num_samples, num_features, num_trees, base_score
        );
    } else {
        // Fewer trees: use sample-level parallelism
        dim3 grid((num_samples + 255) / 256);
        dim3 block(256);
        xgboost_inference_basic<<<grid, block>>>(
            d_features, d_nodes, d_tree_offsets, d_predictions,
            num_samples, num_features, num_trees, base_score
        );
    }
}

