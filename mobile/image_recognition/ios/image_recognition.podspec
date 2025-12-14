Pod::Spec.new do |s|
  s.name             = 'image_recognition'
  s.version          = '1.0.0'
  s.summary          = 'Image recognition plugin for Flutter using TensorFlow Lite'
  s.description      = <<-DESC
A Flutter plugin for image recognition using TensorFlow Lite with full integer quantization.
Supports offline inference with exact letterbox preprocessing matching the training pipeline.
                       DESC
  s.homepage         = 'https://github.com/percepthor/image_recognition'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Percepthor' => 'felipe@lara.ac' }
  s.source           = { :path => '.' }

  # Platform
  s.platform = :ios, '12.0'

  # Source files (Flutter will handle this automatically)
  s.source_files = 'Classes/**/*'

  # Vendored libraries
  s.vendored_libraries = 'libs/libimage_recognition.a'
  s.vendored_frameworks = 'Frameworks/TensorFlowLiteC.framework'

  # Force load static library to prevent symbol stripping
  # This is critical for FFI to work with static libraries on iOS
  s.pod_target_xcconfig = {
    'OTHER_LDFLAGS' => '-Wl,-force_load,${PODS_TARGET_SRCROOT}/libs/libimage_recognition.a',
    'DEFINES_MODULE' => 'YES'
  }

  # Dependencies
  s.dependency 'Flutter'

  # Deployment target
  s.ios.deployment_target = '12.0'

  # Static framework (required for vendored static library)
  s.static_framework = true
end

# IMPORTANT: iOS Native Library Setup
#
# This plugin requires two native components:
#
# 1. libimage_recognition.a (this project's native code)
#    - Build with: native/build/build_ios_all.sh
#    - Creates universal (fat) static library with arm64 + x86_64
#    - Copy to: ios/libs/libimage_recognition.a
#
# 2. TensorFlowLiteC.framework (TensorFlow Lite C API)
#    - Build with Bazel: bazel build --config=ios_fat //tensorflow/lite/ios:TensorFlowLiteC_framework
#    - Extract from bazel-bin/tensorflow/lite/ios/TensorFlowLiteC_framework.zip
#    - Copy to: ios/Frameworks/TensorFlowLiteC.framework
#
# FFI on iOS:
#  - Uses DynamicLibrary.process() to load symbols from static library
#  - Requires -force_load flag to prevent linker from stripping unused symbols
#  - See: https://docs.flutter.dev/platform-integration/ios/c-interop
