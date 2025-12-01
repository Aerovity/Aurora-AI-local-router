/**
 * AuroraAI Router - C/C++ API for Cactus Integration
 *
 * Provides intelligent model routing for Cactus Compute models
 * using cluster-based selection with per-cluster error rates.
 */

#ifndef CACTUS_ROUTER_H
#define CACTUS_ROUTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to router instance
 */
typedef struct CactusRouterHandle CactusRouterHandle;

/**
 * Model recommendation result
 */
typedef struct {
    const char* model_id;          // Model identifier (e.g., "gemma-270m")
    const char* model_path;        // Path to model weights
    float score;                   // Selection score (lower is better)
    int cluster_id;                // Assigned cluster
    float estimated_latency_ms;    // Estimated inference latency
} CactusModelRecommendation;

/**
 * Router initialization options
 */
typedef struct {
    const char* profile_path;      // Path to router profile JSON
    float lambda_min;              // Minimum lambda (default: 0.0)
    float lambda_max;              // Maximum lambda (default: 2.0)
    float default_cost_preference; // Default cost preference (default: 0.5)
} CactusRouterOptions;

/**
 * Initialize router from profile.
 *
 * @param options Router configuration options
 * @return Router handle on success, NULL on failure
 *
 * Example:
 *   CactusRouterOptions opts = {
 *       .profile_path = "cactus_profile.json",
 *       .lambda_min = 0.0,
 *       .lambda_max = 2.0,
 *       .default_cost_preference = 0.5
 *   };
 *   CactusRouterHandle* router = cactus_router_init(&opts);
 */
CactusRouterHandle* cactus_router_init(const CactusRouterOptions* options);

/**
 * Route a prompt to optimal model.
 *
 * @param router Router handle
 * @param prompt Input text prompt
 * @param embedding Pre-computed embedding vector (can be NULL to compute)
 * @param embedding_dim Dimension of embedding vector
 * @param available_models Array of model IDs to consider (NULL for all)
 * @param n_available Number of models in available_models array
 * @param cost_preference Cost preference: 0.0=fast/small, 1.0=quality/large
 * @param result Output recommendation result
 * @return 0 on success, negative on error
 *
 * Example:
 *   CactusModelRecommendation result;
 *   int ret = cactus_router_select(
 *       router,
 *       "Explain quantum physics",
 *       NULL,  // Auto-compute embedding
 *       0,
 *       NULL,  // Consider all models
 *       0,
 *       0.8,   // Prefer quality
 *       &result
 *   );
 *   if (ret == 0) {
 *       printf("Selected: %s\n", result.model_id);
 *   }
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
);

/**
 * Get supported model IDs.
 *
 * @param router Router handle
 * @param model_ids Output array of model ID strings (allocated by caller)
 * @param max_models Maximum number of model IDs to return
 * @return Number of models returned
 */
int cactus_router_get_models(
    CactusRouterHandle* router,
    const char** model_ids,
    int max_models
);

/**
 * Get router statistics.
 *
 * @param router Router handle
 * @param n_clusters Output: number of clusters
 * @param n_models Output: number of models
 * @return 0 on success, negative on error
 */
int cactus_router_get_stats(
    CactusRouterHandle* router,
    int* n_clusters,
    int* n_models
);

/**
 * Destroy router and free resources.
 *
 * @param router Router handle to destroy
 */
void cactus_router_destroy(CactusRouterHandle* router);

#ifdef __cplusplus
}
#endif

#endif // CACTUS_ROUTER_H
