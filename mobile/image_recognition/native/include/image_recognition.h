#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IMAGE_REC_LABEL_MAX   128
#define IMAGE_REC_ERRMSG_MAX  256

typedef enum image_rec_error {
  IMAGE_REC_OK = 0,
  IMAGE_REC_ERR_NOT_INITIALIZED = 1,
  IMAGE_REC_ERR_INVALID_ARGUMENT = 2,
  IMAGE_REC_ERR_DECODE_FAILED = 3,
  IMAGE_REC_ERR_MODEL_LOAD_FAILED = 4,
  IMAGE_REC_ERR_TFLITE_ALLOC_FAILED = 5,
  IMAGE_REC_ERR_INVOKE_FAILED = 6,
  IMAGE_REC_ERR_IO = 7,
  IMAGE_REC_ERR_OUT_OF_MEMORY = 8,
  IMAGE_REC_ERR_INTERNAL = 100
} image_rec_error;

typedef struct imageRecResult {
  int32_t error_code;                  // IMAGE_REC_OK if success
  int32_t class_index;                 // >=0 winner, -1 if "unknown" or error
  int32_t confidence;                  // 0..100 (integer)
  float   confidence_f;                // 0..1
  float   applied_threshold;           // threshold used
  int32_t is_unknown;                  // 1 if confidence_f < threshold
  char    label[IMAGE_REC_LABEL_MAX];  // UTF-8, null-terminated
  char    error_message[IMAGE_REC_ERRMSG_MAX];

  // timing (ms)
  uint32_t time_decode_ms;
  uint32_t time_preprocess_ms;
  uint32_t time_infer_ms;
  uint32_t time_post_ms;
  uint32_t time_total_ms;
} imageRecResult;

// Initialize engine. model_path: .tflite on filesystem.
// config_path: JSON with labels + threshold (see spec).
int32_t image_rec_init(const char* model_path, const char* config_path);

// Analyze image from compressed bytes (JPG/PNG).
int32_t image_rec_analyze_image_bytes(const uint8_t* bytes, int32_t length, imageRecResult* out_result);

// Fast path: RGBA already decoded (camera).
int32_t image_rec_analyze_pixels_rgba(const uint8_t* rgba, int32_t width, int32_t height, imageRecResult* out_result);

// Adjust threads (before init or reinitializing internally).
int32_t image_rec_set_num_threads(int32_t num_threads);

// Complete cleanup.
void image_rec_shutdown(void);

#ifdef __cplusplus
}
#endif
