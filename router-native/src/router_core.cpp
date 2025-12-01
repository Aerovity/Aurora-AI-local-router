/**
 * AuroraAI Router - Core C++ implementation
 *
 * NOTE: This is a simplified implementation stub.
 * For production use, integrate with a C++ ML library like:
 * - ONNX Runtime Mobile (for embeddings)
 * - Eigen (for linear algebra)
 * - nlohmann/json (for profile parsing)
 */

#include "../include/cactus_router.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>

// Simple placeholder structures
struct ModelInfo {
    std::string model_id;
    std::string model_path;
    float size_mb;
    float avg_tokens_per_sec;
};

struct CactusRouterHandle {
    std::vector<std::vector<float>> cluster_centers;  // n_clusters × feature_dim
    std::unordered_map<std::string, std::vector<float>> error_rates;  // model_id -> per-cluster rates
    std::unordered_map<std::string, ModelInfo> models;

    int n_clusters;
    int feature_dim;
    float lambda_min;
    float lambda_max;
    float default_cost_preference;
    float min_size;
    float max_size;
};

/**
 * Helper: Compute cosine similarity between two vectors
 */
static float cosine_similarity(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

/**
 * Helper: Find nearest cluster
 */
static int find_nearest_cluster(
    const float* embedding,
    size_t embedding_dim,
    const std::vector<std::vector<float>>& cluster_centers
) {
    int best_cluster = 0;
    float best_similarity = -1.0f;

    for (size_t i = 0; i < cluster_centers.size(); i++) {
        float sim = cosine_similarity(
            embedding,
            cluster_centers[i].data(),
            embedding_dim
        );

        if (sim > best_similarity) {
            best_similarity = sim;
            best_cluster = static_cast<int>(i);
        }
    }

    return best_cluster;
}

/**
 * Initialize router from profile
 */
CactusRouterHandle* cactus_router_init(const CactusRouterOptions* options) {
    if (!options || !options->profile_path) {
        return nullptr;
    }

    // TODO: Load profile JSON and parse
    // For now, return a placeholder handle
    // In production, use nlohmann/json to parse the profile

    CactusRouterHandle* router = new CactusRouterHandle();
    router->lambda_min = options->lambda_min;
    router->lambda_max = options->lambda_max;
    router->default_cost_preference = options->default_cost_preference;
    router->n_clusters = 0;
    router->feature_dim = 0;
    router->min_size = 0.0f;
    router->max_size = 1000.0f;

    // TODO: Load actual profile data from JSON
    // This would include:
    // - cluster_centers (from profile['cluster_centers']['cluster_centers'])
    // - error_rates (from profile['llm_profiles'])
    // - models (from profile['models'])

    return router;
}

/**
 * Route prompt to optimal model
 */
int cactus_router_select(
    CactusRouterHandle* router,
    const char* prompt,
    const float* embedding,
    size_t embedding_dim,
    const char** available_models,
    int n_available,
    float cost_preference,
    CactusModelRecommendation* result
) {
    if (!router || !result) {
        return -1;
    }

    // TODO: If embedding is NULL, compute it from prompt
    // This would require integrating a lightweight embedding model
    // Options:
    // 1. ONNX Runtime Mobile with quantized SentenceTransformer
    // 2. TF-IDF vectorization (fast but less accurate)
    // 3. Simple bag-of-words heuristics

    if (!embedding) {
        // For now, return error if embedding not provided
        return -2;  // Embedding required
    }

    // Find nearest cluster
    int cluster_id = find_nearest_cluster(
        embedding,
        embedding_dim,
        router->cluster_centers
    );

    // Calculate lambda
    float lambda = router->lambda_max - cost_preference * (
        router->lambda_max - router->lambda_min
    );

    // Score all models
    std::string best_model_id;
    float best_score = 1e9f;

    for (const auto& [model_id, model_info] : router->models) {
        // Skip if not in available_models list
        if (available_models && n_available > 0) {
            bool found = false;
            for (int i = 0; i < n_available; i++) {
                if (strcmp(available_models[i], model_id.c_str()) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) continue;
        }

        // Get error rate for this model in this cluster
        float error_rate = router->error_rates[model_id][cluster_id];

        // Normalize size
        float normalized_size = (model_info.size_mb - router->min_size) /
                                (router->max_size - router->min_size);

        // Compute score
        float score = error_rate + lambda * normalized_size;

        if (score < best_score) {
            best_score = score;
            best_model_id = model_id;
        }
    }

    if (best_model_id.empty()) {
        return -3;  // No valid model found
    }

    // Fill result
    const ModelInfo& selected = router->models[best_model_id];

    // Allocate strings (caller must manage lifetime)
    static std::string result_model_id;
    static std::string result_model_path;
    result_model_id = selected.model_id;
    result_model_path = selected.model_path;

    result->model_id = result_model_id.c_str();
    result->model_path = result_model_path.c_str();
    result->score = best_score;
    result->cluster_id = cluster_id;
    result->estimated_latency_ms = (100.0f / selected.avg_tokens_per_sec) * 1000.0f;

    return 0;
}

/**
 * Get supported model IDs
 */
int cactus_router_get_models(
    CactusRouterHandle* router,
    const char** model_ids,
    int max_models
) {
    if (!router || !model_ids) {
        return -1;
    }

    int count = 0;
    static std::vector<std::string> model_id_storage;
    model_id_storage.clear();

    for (const auto& [model_id, _] : router->models) {
        if (count >= max_models) break;
        model_id_storage.push_back(model_id);
        model_ids[count++] = model_id_storage.back().c_str();
    }

    return count;
}

/**
 * Get router statistics
 */
int cactus_router_get_stats(
    CactusRouterHandle* router,
    int* n_clusters,
    int* n_models
) {
    if (!router) {
        return -1;
    }

    if (n_clusters) *n_clusters = router->n_clusters;
    if (n_models) *n_models = static_cast<int>(router->models.size());

    return 0;
}

/**
 * Destroy router
 */
void cactus_router_destroy(CactusRouterHandle* router) {
    delete router;
}
