#include "image_recognition.h"
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <mutex>
#include <chrono>

// Forward declarations from other modules
extern "C" {
    bool decode_image_rgb(const uint8_t* bytes, int32_t length, uint8_t** out_rgb, int* out_w, int* out_h);
    void free_decoded_image(uint8_t* rgb);

    bool letterbox_preprocess(const uint8_t* rgb, int w, int h, int target_size, uint8_t** out_letterbox);
    void free_letterbox(uint8_t* letterbox);

    bool letterbox_preprocess_rgba(const uint8_t* rgba, int w, int h, int target_size, uint8_t** out_letterbox);

    void compute_softmax(const float* logits, int num_classes, float* probs);

    bool load_config(const char* config_path, char** out_labels_path, char** out_threshold_path,
                     char** out_unknown_label, int* out_num_threads);
    void free_config_strings(char* labels_path, char* threshold_path, char* unknown_label);

    bool load_labels(const char* labels_path, std::vector<std::string>& labels);
    bool load_threshold(const char* threshold_path, float* recommended_threshold);

    uint64_t get_time_ms();
}

// Engine state
struct EngineState {
    TfLiteModel* model = nullptr;
    TfLiteInterpreterOptions* options = nullptr;
    TfLiteInterpreter* interpreter = nullptr;
    TfLiteXNNPackDelegateOptions xnnpack_opts;
    TfLiteDelegate* xnnpack_delegate = nullptr;

    // Model metadata
    int input_tensor_index = 0;
    int output_tensor_index = 0;
    TfLiteType input_dtype;
    TfLiteType output_dtype;
    int input_width = 0;
    int input_height = 0;
    int input_channels = 0;
    float input_scale = 1.0f;
    int32_t input_zero_point = 0;
    float output_scale = 1.0f;
    int32_t output_zero_point = 0;
    int num_classes = 0;

    // Configuration
    std::vector<std::string> labels;
    float recommended_threshold = 0.5f;
    std::string unknown_label = "desconocido";
    int num_threads = 2;

    // State
    bool initialized = false;
    std::mutex mutex;
};

static EngineState g_engine;

static void set_error(imageRecResult* result, int32_t code, const char* message) {
    result->error_code = code;
    result->class_index = -1;
    result->confidence = 0;
    result->confidence_f = 0.0f;
    result->is_unknown = 1;
    strncpy(result->error_message, message, IMAGE_REC_ERRMSG_MAX - 1);
    result->error_message[IMAGE_REC_ERRMSG_MAX - 1] = '\0';
    strncpy(result->label, "", IMAGE_REC_LABEL_MAX - 1);
    result->label[0] = '\0';
}

int32_t image_rec_set_num_threads(int32_t num_threads) {
    if (num_threads < 1 || num_threads > 16) {
        return IMAGE_REC_ERR_INVALID_ARGUMENT;
    }
    g_engine.num_threads = num_threads;
    return IMAGE_REC_OK;
}

int32_t image_rec_init(const char* model_path, const char* config_path) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);

    if (g_engine.initialized) {
        image_rec_shutdown();
    }

    // Load configuration
    char* labels_path = nullptr;
    char* threshold_path = nullptr;
    char* unknown_label_str = nullptr;
    int config_num_threads = 0;

    if (!load_config(config_path, &labels_path, &threshold_path, &unknown_label_str, &config_num_threads)) {
        return IMAGE_REC_ERR_IO;
    }

    if (config_num_threads > 0) {
        g_engine.num_threads = config_num_threads;
    }

    if (unknown_label_str) {
        g_engine.unknown_label = unknown_label_str;
    }

    // Load labels
    if (!load_labels(labels_path, g_engine.labels)) {
        free_config_strings(labels_path, threshold_path, unknown_label_str);
        return IMAGE_REC_ERR_IO;
    }

    // Load threshold
    if (!load_threshold(threshold_path, &g_engine.recommended_threshold)) {
        free_config_strings(labels_path, threshold_path, unknown_label_str);
        return IMAGE_REC_ERR_IO;
    }

    free_config_strings(labels_path, threshold_path, unknown_label_str);

    // Load model
    g_engine.model = TfLiteModelCreateFromFile(model_path);
    if (!g_engine.model) {
        return IMAGE_REC_ERR_MODEL_LOAD_FAILED;
    }

    // Create interpreter options
    g_engine.options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(g_engine.options, g_engine.num_threads);

    // Create XNNPACK delegate
    g_engine.xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    g_engine.xnnpack_opts.num_threads = g_engine.num_threads;
    g_engine.xnnpack_delegate = TfLiteXNNPackDelegateCreate(&g_engine.xnnpack_opts);

    if (g_engine.xnnpack_delegate) {
        TfLiteInterpreterOptionsAddDelegate(g_engine.options, g_engine.xnnpack_delegate);
    }

    // Create interpreter
    g_engine.interpreter = TfLiteInterpreterCreate(g_engine.model, g_engine.options);
    if (!g_engine.interpreter) {
        return IMAGE_REC_ERR_TFLITE_ALLOC_FAILED;
    }

    // Allocate tensors
    if (TfLiteInterpreterAllocateTensors(g_engine.interpreter) != kTfLiteOk) {
        return IMAGE_REC_ERR_TFLITE_ALLOC_FAILED;
    }

    // Get input tensor metadata
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(g_engine.interpreter, 0);
    if (!input_tensor) {
        return IMAGE_REC_ERR_TFLITE_ALLOC_FAILED;
    }

    g_engine.input_dtype = TfLiteTensorType(input_tensor);
    int num_dims = TfLiteTensorNumDims(input_tensor);

    // Expecting [batch, height, width, channels] or [height, width, channels]
    if (num_dims == 4) {
        g_engine.input_height = TfLiteTensorDim(input_tensor, 1);
        g_engine.input_width = TfLiteTensorDim(input_tensor, 2);
        g_engine.input_channels = TfLiteTensorDim(input_tensor, 3);
    } else if (num_dims == 3) {
        g_engine.input_height = TfLiteTensorDim(input_tensor, 0);
        g_engine.input_width = TfLiteTensorDim(input_tensor, 1);
        g_engine.input_channels = TfLiteTensorDim(input_tensor, 2);
    }

    // Get quantization parameters
    TfLiteQuantizationParams input_quant = TfLiteTensorQuantizationParams(input_tensor);
    g_engine.input_scale = input_quant.scale;
    g_engine.input_zero_point = input_quant.zero_point;

    // Get output tensor metadata
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(g_engine.interpreter, 0);
    if (!output_tensor) {
        return IMAGE_REC_ERR_TFLITE_ALLOC_FAILED;
    }

    g_engine.output_dtype = TfLiteTensorType(output_tensor);
    g_engine.num_classes = TfLiteTensorDim(output_tensor, TfLiteTensorNumDims(output_tensor) - 1);

    TfLiteQuantizationParams output_quant = TfLiteTensorQuantizationParams(output_tensor);
    g_engine.output_scale = output_quant.scale;
    g_engine.output_zero_point = output_quant.zero_point;

    g_engine.initialized = true;
    return IMAGE_REC_OK;
}

static void quantize_input(const uint8_t* letterbox, int size, int8_t* quantized) {
    // Fast path: if scale=1.0 and zero_point=-128
    if (std::abs(g_engine.input_scale - 1.0f) < 1e-6 && g_engine.input_zero_point == -128) {
        for (int i = 0; i < size; i++) {
            quantized[i] = (int8_t)(letterbox[i] - 128);
        }
    } else {
        // General quantization
        for (int i = 0; i < size; i++) {
            float val = letterbox[i];
            int32_t q = (int32_t)roundf(val / g_engine.input_scale) + g_engine.input_zero_point;
            if (g_engine.input_dtype == kTfLiteInt8) {
                q = std::max(-128, std::min(127, q));
                quantized[i] = (int8_t)q;
            } else {
                q = std::max(0, std::min(255, q));
                quantized[i] = (int8_t)q;
            }
        }
    }
}

static void quantize_input_uint8(const uint8_t* letterbox, int size, uint8_t* quantized) {
    // For uint8 input
    if (std::abs(g_engine.input_scale - 1.0f) < 1e-6 && g_engine.input_zero_point == 0) {
        memcpy(quantized, letterbox, size);
    } else {
        for (int i = 0; i < size; i++) {
            float val = letterbox[i];
            int32_t q = (int32_t)roundf(val / g_engine.input_scale) + g_engine.input_zero_point;
            q = std::max(0, std::min(255, q));
            quantized[i] = (uint8_t)q;
        }
    }
}

static int32_t run_inference_internal(const uint8_t* letterbox, imageRecResult* out_result) {
    uint64_t t0 = get_time_ms();

    // Get input tensor
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(g_engine.interpreter, 0);
    if (!input_tensor) {
        return IMAGE_REC_ERR_TFLITE_ALLOC_FAILED;
    }

    int input_size = g_engine.input_width * g_engine.input_height * g_engine.input_channels;

    // Quantize input
    uint64_t t_pre_start = get_time_ms();

    if (g_engine.input_dtype == kTfLiteInt8) {
        int8_t* input_data = (int8_t*)TfLiteTensorData(input_tensor);
        quantize_input(letterbox, input_size, input_data);
    } else {
        uint8_t* input_data = (uint8_t*)TfLiteTensorData(input_tensor);
        quantize_input_uint8(letterbox, input_size, input_data);
    }

    uint64_t t_pre_end = get_time_ms();
    out_result->time_preprocess_ms += (t_pre_end - t_pre_start);

    // Run inference
    uint64_t t_infer_start = get_time_ms();
    if (TfLiteInterpreterInvoke(g_engine.interpreter) != kTfLiteOk) {
        return IMAGE_REC_ERR_INVOKE_FAILED;
    }
    uint64_t t_infer_end = get_time_ms();
    out_result->time_infer_ms = (t_infer_end - t_infer_start);

    // Post-processing
    uint64_t t_post_start = get_time_ms();

    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(g_engine.interpreter, 0);
    const int8_t* output_data = (const int8_t*)TfLiteTensorData(output_tensor);

    // Dequantize logits
    std::vector<float> logits(g_engine.num_classes);
    for (int i = 0; i < g_engine.num_classes; i++) {
        logits[i] = (output_data[i] - g_engine.output_zero_point) * g_engine.output_scale;
    }

    // Compute softmax
    std::vector<float> probs(g_engine.num_classes);
    compute_softmax(logits.data(), g_engine.num_classes, probs.data());

    // Find argmax
    int best_index = 0;
    float best_prob = probs[0];
    for (int i = 1; i < g_engine.num_classes; i++) {
        if (probs[i] > best_prob) {
            best_prob = probs[i];
            best_index = i;
        }
    }

    // Apply threshold
    out_result->applied_threshold = g_engine.recommended_threshold;
    out_result->confidence_f = best_prob;
    out_result->confidence = (int32_t)roundf(best_prob * 100.0f);

    if (best_prob < g_engine.recommended_threshold) {
        out_result->is_unknown = 1;
        out_result->class_index = -1;
        strncpy(out_result->label, g_engine.unknown_label.c_str(), IMAGE_REC_LABEL_MAX - 1);
        out_result->label[IMAGE_REC_LABEL_MAX - 1] = '\0';
    } else {
        out_result->is_unknown = 0;
        out_result->class_index = best_index;
        if (best_index < (int)g_engine.labels.size()) {
            strncpy(out_result->label, g_engine.labels[best_index].c_str(), IMAGE_REC_LABEL_MAX - 1);
            out_result->label[IMAGE_REC_LABEL_MAX - 1] = '\0';
        } else {
            snprintf(out_result->label, IMAGE_REC_LABEL_MAX, "class_%d", best_index);
        }
    }

    uint64_t t_post_end = get_time_ms();
    out_result->time_post_ms = (t_post_end - t_post_start);

    return IMAGE_REC_OK;
}

int32_t image_rec_analyze_image_bytes(const uint8_t* bytes, int32_t length, imageRecResult* out_result) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);

    if (!g_engine.initialized) {
        set_error(out_result, IMAGE_REC_ERR_NOT_INITIALIZED, "Engine not initialized");
        return IMAGE_REC_ERR_NOT_INITIALIZED;
    }

    if (!bytes || length <= 0 || !out_result) {
        set_error(out_result, IMAGE_REC_ERR_INVALID_ARGUMENT, "Invalid arguments");
        return IMAGE_REC_ERR_INVALID_ARGUMENT;
    }

    memset(out_result, 0, sizeof(imageRecResult));
    uint64_t t_total_start = get_time_ms();

    // Decode image
    uint64_t t_decode_start = get_time_ms();
    uint8_t* rgb = nullptr;
    int w = 0, h = 0;
    if (!decode_image_rgb(bytes, length, &rgb, &w, &h)) {
        set_error(out_result, IMAGE_REC_ERR_DECODE_FAILED, "Failed to decode image");
        return IMAGE_REC_ERR_DECODE_FAILED;
    }
    uint64_t t_decode_end = get_time_ms();
    out_result->time_decode_ms = (t_decode_end - t_decode_start);

    // Letterbox preprocessing
    uint64_t t_pre_start = get_time_ms();
    uint8_t* letterbox = nullptr;
    if (!letterbox_preprocess(rgb, w, h, 240, &letterbox)) {
        free_decoded_image(rgb);
        set_error(out_result, IMAGE_REC_ERR_INTERNAL, "Letterbox preprocessing failed");
        return IMAGE_REC_ERR_INTERNAL;
    }
    uint64_t t_pre_end = get_time_ms();
    out_result->time_preprocess_ms = (t_pre_end - t_pre_start);

    free_decoded_image(rgb);

    // Run inference
    int32_t result = run_inference_internal(letterbox, out_result);
    free_letterbox(letterbox);

    if (result != IMAGE_REC_OK) {
        set_error(out_result, result, "Inference failed");
        return result;
    }

    uint64_t t_total_end = get_time_ms();
    out_result->time_total_ms = (t_total_end - t_total_start);
    out_result->error_code = IMAGE_REC_OK;

    return IMAGE_REC_OK;
}

int32_t image_rec_analyze_pixels_rgba(const uint8_t* rgba, int32_t width, int32_t height, imageRecResult* out_result) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);

    if (!g_engine.initialized) {
        set_error(out_result, IMAGE_REC_ERR_NOT_INITIALIZED, "Engine not initialized");
        return IMAGE_REC_ERR_NOT_INITIALIZED;
    }

    if (!rgba || width <= 0 || height <= 0 || !out_result) {
        set_error(out_result, IMAGE_REC_ERR_INVALID_ARGUMENT, "Invalid arguments");
        return IMAGE_REC_ERR_INVALID_ARGUMENT;
    }

    memset(out_result, 0, sizeof(imageRecResult));
    uint64_t t_total_start = get_time_ms();

    // Letterbox preprocessing from RGBA
    uint64_t t_pre_start = get_time_ms();
    uint8_t* letterbox = nullptr;
    if (!letterbox_preprocess_rgba(rgba, width, height, 240, &letterbox)) {
        set_error(out_result, IMAGE_REC_ERR_INTERNAL, "Letterbox preprocessing failed");
        return IMAGE_REC_ERR_INTERNAL;
    }
    uint64_t t_pre_end = get_time_ms();
    out_result->time_preprocess_ms = (t_pre_end - t_pre_start);

    // Run inference
    int32_t result = run_inference_internal(letterbox, out_result);
    free_letterbox(letterbox);

    if (result != IMAGE_REC_OK) {
        set_error(out_result, result, "Inference failed");
        return result;
    }

    uint64_t t_total_end = get_time_ms();
    out_result->time_total_ms = (t_total_end - t_total_start);
    out_result->error_code = IMAGE_REC_OK;

    return IMAGE_REC_OK;
}

void image_rec_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);

    if (g_engine.interpreter) {
        TfLiteInterpreterDelete(g_engine.interpreter);
        g_engine.interpreter = nullptr;
    }

    if (g_engine.xnnpack_delegate) {
        TfLiteXNNPackDelegateDelete(g_engine.xnnpack_delegate);
        g_engine.xnnpack_delegate = nullptr;
    }

    if (g_engine.options) {
        TfLiteInterpreterOptionsDelete(g_engine.options);
        g_engine.options = nullptr;
    }

    if (g_engine.model) {
        TfLiteModelDelete(g_engine.model);
        g_engine.model = nullptr;
    }

    g_engine.labels.clear();
    g_engine.initialized = false;
}
