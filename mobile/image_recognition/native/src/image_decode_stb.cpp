#include <cstdint>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

extern "C" {

// Decode compressed image bytes (JPG/PNG) to RGB
bool decode_image_rgb(const uint8_t* bytes, int32_t length, uint8_t** out_rgb, int* out_w, int* out_h) {
    if (!bytes || length <= 0 || !out_rgb || !out_w || !out_h) {
        return false;
    }

    int width, height, channels;
    // Force 3 channels (RGB)
    uint8_t* data = stbi_load_from_memory(bytes, length, &width, &height, &channels, 3);

    if (!data) {
        return false;
    }

    *out_rgb = data;
    *out_w = width;
    *out_h = height;
    return true;
}

void free_decoded_image(uint8_t* rgb) {
    if (rgb) {
        stbi_image_free(rgb);
    }
}

}  // extern "C"
