# Third-Party Dependencies

This directory contains required third-party libraries for the native image recognition engine.

## Required Files

Download the following files and place them in this directory:

### 1. STB Image (stb_image.h)
- **Source**: https://github.com/nothings/stb/blob/master/stb_image.h
- **License**: MIT / Public Domain
- **Purpose**: Image decoding (JPG, PNG)
- **Usage**: Single-header library for image loading

### 2. STB Image Resize 2 (stb_image_resize2.h)
- **Source**: https://github.com/nothings/stb/blob/master/stb_image_resize2.h
- **License**: MIT / Public Domain
- **Purpose**: Bilinear image resizing
- **Usage**: Single-header library for high-quality image resizing

### 3. cJSON (cJSON.c, cJSON.h)
- **Source**: https://github.com/DaveGamble/cJSON
- **License**: MIT
- **Purpose**: JSON parsing for configuration files
- **Files needed**:
  - cJSON.c
  - cJSON.h

## Installation Instructions

```bash
cd native/third_party

# Download STB libraries
curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h

# Download cJSON
curl -O https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.c
curl -O https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.h
```

## Directory Structure

After downloading, you should have:
```
native/third_party/
├── README.md (this file)
├── stb_image.h
├── stb_image_resize2.h
├── cJSON.c
└── cJSON.h
```
