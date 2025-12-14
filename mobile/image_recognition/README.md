# Image Recognition Flutter Plugin

Offline image recognition for Flutter using TensorFlow Lite with full integer quantization (int8). Designed for mobile deployment with exact preprocessing matching the training pipeline.

## Features

- **Offline Inference**: No internet connection required
- **Full Integer Quantization**: int8 TFLite models for fast inference
- **Non-blocking**: Runs in worker isolate to avoid blocking UI
- **Exact Preprocessing**: Letterbox resize with bilinear interpolation (matches training)
- **XNNPACK Acceleration**: Optimized for mobile CPUs
- **Camera & Gallery**: Support for both compressed images and RGBA pixels
- **Unknown Detection**: Automatic threshold-based rejection of low-confidence predictions

## Quick Start

### 1. Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  image_recognition:
    path: path/to/image_recognition
```

### 2. Add Assets

Place your trained model files in `assets/image_recognition/`:

```
assets/
  image_recognition/
    model_qat_int8.tflite
    labels.txt
    threshold_recommendation.json
    runtime_config.json
```

Update `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/image_recognition/
```

### 3. Usage

```dart
import 'package:image_recognition/image_recognition.dart';

// Initialize
final engine = ImageRecognition();
await engine.initialize();

// Analyze from image bytes (JPG/PNG)
final result = await engine.analyze(imageBytes);
print('Class: ${result.imageClass}');
print('Confidence: ${result.classConfidence}%');
print('Time: ${result.timeTotalMs}ms');

// Analyze from RGBA pixels (camera)
final result = await engine.analyzeRgba(rgbaBytes, width, height);

// Dispose
await engine.dispose();
```

## Build Instructions

### Prerequisites

- Flutter SDK (>=3.0.0)
- Android NDK (for Android builds)
- Xcode (for iOS builds)
- Bazel (for building TensorFlow Lite)

### Step 1: Download Third-Party Dependencies

```bash
cd native/third_party
curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h
curl -O https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.c
curl -O https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.h
```

### Step 2: Build TensorFlow Lite C Libraries

#### Android

```bash
cd native/build
./build_android_all.sh
```

This builds `libtensorflowlite_c.so` for all ABIs (arm64-v8a, armeabi-v7a, x86, x86_64).

#### iOS

```bash
cd native/build
./build_ios_all.sh
```

This builds `TensorFlowLiteC.framework` as a universal framework.

### Step 3: Build libimage_recognition

#### Android

For each ABI:

```bash
cd native
mkdir -p build/android-<abi>
cd build/android-<abi>

cmake ../.. \
  -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=<abi> \
  -DANDROID_PLATFORM=android-21

make
cp libimage_recognition.so ../../../android/src/main/jniLibs/<abi>/
```

Replace `<abi>` with: `arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`

#### iOS

```bash
cd native
mkdir -p build/ios
cd build/ios

# Build for device (arm64)
cmake ../.. \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
  -DCMAKE_BUILD_TYPE=Release

make
cp libimage_recognition.a ../../../ios/libs/libimage_recognition_arm64.a

# Build for simulator (x86_64)
cmake ../.. \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DCMAKE_OSX_SYSROOT=iphonesimulator \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
  -DCMAKE_BUILD_TYPE=Release

make
cp libimage_recognition.a ../../../ios/libs/libimage_recognition_x86_64.a

# Create universal binary
lipo -create \
  ../../../ios/libs/libimage_recognition_arm64.a \
  ../../../ios/libs/libimage_recognition_x86_64.a \
  -output ../../../ios/libs/libimage_recognition.a
```

### Step 4: Generate FFI Bindings

```bash
flutter pub get
flutter pub run ffigen
```

This generates `lib/src/bindings.g.dart` from `native/include/image_recognition.h`.

### Step 5: Run Example

```bash
cd example
flutter run
```

## Runtime Configuration

Create `runtime_config.json`:

```json
{
  "labels_path": "labels.txt",
  "threshold_path": "threshold_recommendation.json",
  "unknown_label": "desconocido",
  "num_threads": 2
}
```

## API Reference

### ImageRecognition

Main API class for image recognition.

#### Methods

- `Future<void> initialize()`: Initialize engine and extract assets
- `Future<ImageRecognitionResult> analyze(Uint8List imageBytes)`: Analyze compressed image
- `Future<ImageRecognitionResult> analyzeRgba(Uint8List rgba, int width, int height)`: Analyze RGBA pixels
- `Future<void> dispose()`: Cleanup and shutdown
- `bool get isInitialized`: Check initialization status

### ImageRecognitionResult

Analysis result with detailed information.

#### Properties

- `String imageClass`: Predicted class name or "desconocido"
- `int classConfidence`: Confidence as percentage (0-100)
- `int? classIndex`: Class index (-1 for unknown)
- `bool isUnknown`: True if confidence below threshold
- `double confidence01`: Confidence as decimal (0-1)
- `double appliedThreshold`: Threshold used
- `int timeDecodeMs`: Image decode time
- `int timePreprocessMs`: Preprocessing time
- `int timeInferMs`: Inference time
- `int timePostMs`: Post-processing time
- `int timeTotalMs`: Total time

## Architecture

### Native Layer (C/C++)

- **image_recognition.cpp**: Core inference engine with TFLite C API
- **image_preprocess.cpp**: Exact letterbox preprocessing
- **image_decode_stb.cpp**: Image decoding with stb_image
- **softmax.cpp**: Numerically stable softmax
- **config_parse.cpp**: JSON configuration parsing

### Flutter Layer (Dart)

- **image_recognition.dart**: Public API
- **engine_isolate.dart**: Worker isolate management
- **asset_extractor.dart**: Asset extraction from bundle
- **native_loader.dart**: FFI library loading
- **result.dart**: Result model
- **errors.dart**: Error definitions

## Preprocessing

The plugin uses **letterbox preprocessing** to maintain aspect ratio:

1. Calculate scale: `scale = 240 / max(width, height)`
2. Resize to `(width × scale, height × scale)` using bilinear interpolation
3. Pad with black pixels (0,0,0) to center the image in 240×240 canvas

This **exactly matches** the training pipeline preprocessing.

## Performance

- Typical inference time: 20-50ms (varies by device)
- Non-blocking: Runs in worker isolate
- XNNPACK delegate: Accelerated int8 operations
- Memory efficient: No copies for RGBA fast path

## Troubleshooting

### FFI Loading Issues

**Android**: Ensure `libimage_recognition.so` and `libtensorflowlite_c.so` are in `android/src/main/jniLibs/<abi>/`

**iOS**: Verify podspec includes `-force_load` flag for static library

### Build Errors

- Check NDK/Xcode versions
- Verify Bazel installation
- Ensure all third-party dependencies are downloaded

### Runtime Errors

- Verify assets are included in pubspec.yaml
- Check model file paths in runtime_config.json
- Ensure model is int8 quantized TFLite

## License

Desarrollado por Felipe Lara felipe@lara.ac

## Support

See [example/](example/) for a complete working app with camera and gallery support.
