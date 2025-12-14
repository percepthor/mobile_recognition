#include <cstdio>
<parameter name="cstdlib">
#include <cstring>
#include <string>
#include <vector>

#include "cJSON.h"

extern "C" {

// Helper to resolve relative paths relative to a base directory
static std::string resolve_path(const char* base_path, const char* relative_path) {
    if (relative_path[0] == '/') {
        // Already absolute
        return std::string(relative_path);
    }

    // Find directory of base_path
    std::string base(base_path);
    size_t last_slash = base.find_last_of('/');
    if (last_slash == std::string::npos) {
        // No directory component
        return std::string(relative_path);
    }

    std::string dir = base.substr(0, last_slash + 1);
    return dir + relative_path;
}

bool load_config(const char* config_path, char** out_labels_path, char** out_threshold_path,
                 char** out_unknown_label, int* out_num_threads) {
    FILE* f = fopen(config_path, "rb");
    if (!f) {
        return false;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buffer = (char*)malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);

    cJSON* json = cJSON_Parse(buffer);
    free(buffer);

    if (!json) {
        return false;
    }

    // Parse fields
    cJSON* labels_path_item = cJSON_GetObjectItem(json, "labels_path");
    cJSON* threshold_path_item = cJSON_GetObjectItem(json, "threshold_path");
    cJSON* unknown_label_item = cJSON_GetObjectItem(json, "unknown_label");
    cJSON* num_threads_item = cJSON_GetObjectItem(json, "num_threads");

    if (labels_path_item && cJSON_IsString(labels_path_item)) {
        std::string resolved = resolve_path(config_path, labels_path_item->valuestring);
        *out_labels_path = strdup(resolved.c_str());
    } else {
        cJSON_Delete(json);
        return false;
    }

    if (threshold_path_item && cJSON_IsString(threshold_path_item)) {
        std::string resolved = resolve_path(config_path, threshold_path_item->valuestring);
        *out_threshold_path = strdup(resolved.c_str());
    } else {
        free(*out_labels_path);
        cJSON_Delete(json);
        return false;
    }

    if (unknown_label_item && cJSON_IsString(unknown_label_item)) {
        *out_unknown_label = strdup(unknown_label_item->valuestring);
    } else {
        *out_unknown_label = nullptr;
    }

    if (num_threads_item && cJSON_IsNumber(num_threads_item)) {
        *out_num_threads = num_threads_item->valueint;
    } else {
        *out_num_threads = 0;
    }

    cJSON_Delete(json);
    return true;
}

void free_config_strings(char* labels_path, char* threshold_path, char* unknown_label) {
    if (labels_path) free(labels_path);
    if (threshold_path) free(threshold_path);
    if (unknown_label) free(unknown_label);
}

bool load_labels(const char* labels_path, std::vector<std::string>& labels) {
    FILE* f = fopen(labels_path, "r");
    if (!f) {
        return false;
    }

    labels.clear();
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        // Remove trailing newline
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (len > 0) {
            labels.push_back(std::string(line));
        }
    }

    fclose(f);
    return !labels.empty();
}

bool load_threshold(const char* threshold_path, float* recommended_threshold) {
    FILE* f = fopen(threshold_path, "rb");
    if (!f) {
        return false;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buffer = (char*)malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);

    cJSON* json = cJSON_Parse(buffer);
    free(buffer);

    if (!json) {
        return false;
    }

    cJSON* threshold_item = cJSON_GetObjectItem(json, "recommended_threshold");
    if (threshold_item && cJSON_IsNumber(threshold_item)) {
        *recommended_threshold = (float)threshold_item->valuedouble;
        cJSON_Delete(json);
        return true;
    }

    cJSON_Delete(json);
    return false;
}

}  // extern "C"
