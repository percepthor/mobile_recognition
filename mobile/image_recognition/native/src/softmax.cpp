#include <cmath>
#include <cfloat>

extern "C" {

// Numerically stable softmax
void compute_softmax(const float* logits, int num_classes, float* probs) {
    // Find max for numerical stability
    float max_logit = -FLT_MAX;
    for (int i = 0; i < num_classes; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    // Compute exp(logit - max) and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum_exp += probs[i];
    }

    // Normalize
    for (int i = 0; i < num_classes; i++) {
        probs[i] /= sum_exp;
    }
}

}  // extern "C"
