#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

extern "C" {

// Exact letterbox algorithm matching training pipeline
// Input: RGB (0..255), output: 240x240x3 letterbox with bilinear and black padding
bool letterbox_preprocess(const uint8_t* rgb, int w, int h, int target_size, uint8_t** out_letterbox) {
    if (!rgb || w <= 0 || h <= 0 || !out_letterbox) {
        return false;
    }

    // Step 1: Calculate scale
    int max_dim = std::max(w, h);
    float scale = (float)target_size / (float)max_dim;

    // Step 2: Calculate new dimensions (using round as per spec)
    int new_w = (int)roundf(w * scale);
    int new_h = (int)roundf(h * scale);

    // Ensure dimensions are at least 1
    if (new_w < 1) new_w = 1;
    if (new_h < 1) new_h = 1;

    // Step 3: Allocate buffer for resized image
    uint8_t* resized = new uint8_t[new_w * new_h * 3];

    // Step 4: Resize using bilinear interpolation (stbir_resize_uint8_srgb)
    int result = stbir_resize_uint8_srgb(
        rgb, w, h, 0,              // source
        resized, new_w, new_h, 0,  // destination
        STBIR_RGB                  // pixel layout (3 channels)
    );

    if (result == 0) {
        delete[] resized;
        return false;
    }

    // Step 5: Create letterbox buffer initialized to 0 (black)
    uint8_t* letterbox = new uint8_t[target_size * target_size * 3];
    memset(letterbox, 0, target_size * target_size * 3);

    // Step 6: Calculate padding (centered)
    int pad_left = (target_size - new_w) / 2;  // floor division
    int pad_top = (target_size - new_h) / 2;   // floor division

    // Step 7: Copy resized image to center of letterbox
    for (int y = 0; y < new_h; y++) {
        int dst_y = pad_top + y;
        if (dst_y < 0 || dst_y >= target_size) continue;

        for (int x = 0; x < new_w; x++) {
            int dst_x = pad_left + x;
            if (dst_x < 0 || dst_x >= target_size) continue;

            int src_idx = (y * new_w + x) * 3;
            int dst_idx = (dst_y * target_size + dst_x) * 3;

            letterbox[dst_idx + 0] = resized[src_idx + 0];  // R
            letterbox[dst_idx + 1] = resized[src_idx + 1];  // G
            letterbox[dst_idx + 2] = resized[src_idx + 2];  // B
        }
    }

    delete[] resized;
    *out_letterbox = letterbox;
    return true;
}

// Fast path for RGBA (camera) - convert to RGB and letterbox
bool letterbox_preprocess_rgba(const uint8_t* rgba, int w, int h, int target_size, uint8_t** out_letterbox) {
    if (!rgba || w <= 0 || h <= 0 || !out_letterbox) {
        return false;
    }

    // Convert RGBA to RGB
    uint8_t* rgb = new uint8_t[w * h * 3];
    for (int i = 0; i < w * h; i++) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];  // R
        rgb[i * 3 + 1] = rgba[i * 4 + 1];  // G
        rgb[i * 3 + 2] = rgba[i * 4 + 2];  // B
        // Skip A channel
    }

    bool result = letterbox_preprocess(rgb, w, h, target_size, out_letterbox);
    delete[] rgb;
    return result;
}

void free_letterbox(uint8_t* letterbox) {
    if (letterbox) {
        delete[] letterbox;
    }
}

}  // extern "C"
