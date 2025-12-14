"""TFLite model export with full integer quantization."""
import logging
import os
from typing import Optional, Callable
import tensorflow as tf
import numpy as np


logger = logging.getLogger(__name__)


def export_tflite_full_integer(
    model: tf.keras.Model,
    representative_dataset_gen: Callable,
    output_path: str,
    inference_input_type: str = 'int8',
    inference_output_type: str = 'int8'
) -> dict:
    """
    Export model to TFLite with full integer quantization.

    Args:
        model: Keras model (preferably QAT model)
        representative_dataset_gen: Generator function for calibration
        output_path: Path to save .tflite file
        inference_input_type: 'int8' or 'uint8'
        inference_output_type: 'int8' (output type)

    Returns:
        Dictionary with quantization info and validation results
    """
    logger.info(f"Exporting model to TFLite: {output_path}")
    logger.info(f"Input type: {inference_input_type}, Output type: {inference_output_type}")

    # Map string types to TF types
    type_map = {
        'int8': tf.int8,
        'uint8': tf.uint8
    }

    input_type = type_map.get(inference_input_type, tf.int8)
    output_type = type_map.get(inference_output_type, tf.int8)

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set representative dataset
    converter.representative_dataset = representative_dataset_gen

    # Force full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Set input/output types
    converter.inference_input_type = input_type
    converter.inference_output_type = output_type

    # Convert
    try:
        tflite_model = converter.convert()
        logger.info("Model conversion successful")
    except Exception as e:
        logger.error(f"Model conversion failed: {e}")
        raise

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    file_size_mb = len(tflite_model) / (1024 * 1024)
    logger.info(f"TFLite model saved: {output_path} ({file_size_mb:.2f} MB)")

    # Validate the model
    validation_info = validate_tflite_model(output_path)

    # Extract quantization parameters
    quant_params = extract_quantization_params(output_path)

    result = {
        'model_path': output_path,
        'file_size_mb': file_size_mb,
        'validation': validation_info,
        'quantization_params': quant_params
    }

    return result


def validate_tflite_model(tflite_path: str) -> dict:
    """
    Validate TFLite model to ensure it's full integer.

    Args:
        tflite_path: Path to .tflite file

    Returns:
        Dictionary with validation results

    Raises:
        ValueError: If model is not full integer
    """
    logger.info(f"Validating TFLite model: {tflite_path}")

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details['dtype']
    output_dtype = output_details['dtype']

    logger.info(f"Input dtype: {input_dtype}")
    logger.info(f"Output dtype: {output_dtype}")

    # Check if quantized
    input_is_quantized = input_dtype in [np.int8, np.uint8]
    output_is_quantized = output_dtype in [np.int8]

    if not input_is_quantized:
        raise ValueError(f"Input is not quantized (dtype: {input_dtype}). Expected int8 or uint8.")

    if not output_is_quantized:
        raise ValueError(f"Output is not quantized (dtype: {output_dtype}). Expected int8.")

    logger.info("Model validation passed: Full integer quantization confirmed")

    return {
        'input_dtype': str(input_dtype),
        'output_dtype': str(output_dtype),
        'input_shape': input_details['shape'].tolist(),
        'output_shape': output_details['shape'].tolist(),
        'is_full_integer': True
    }


def extract_quantization_params(tflite_path: str) -> dict:
    """
    Extract quantization parameters from TFLite model.

    Args:
        tflite_path: Path to .tflite file

    Returns:
        Dictionary with quantization parameters for input and output
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Extract input quantization params
    input_quant = input_details.get('quantization_parameters', {})
    input_scale = input_quant.get('scales', [1.0])[0]
    input_zero_point = input_quant.get('zero_points', [0])[0]

    # Extract output quantization params
    output_quant = output_details.get('quantization_parameters', {})
    output_scale = output_quant.get('scales', [1.0])[0]
    output_zero_point = output_quant.get('zero_points', [0])[0]

    params = {
        'input': {
            'dtype': str(input_details['dtype']),
            'scale': float(input_scale),
            'zero_point': int(input_zero_point),
            'shape': input_details['shape'].tolist()
        },
        'output': {
            'dtype': str(output_details['dtype']),
            'scale': float(output_scale),
            'zero_point': int(output_zero_point),
            'shape': output_details['shape'].tolist()
        }
    }

    logger.info(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
    logger.info(f"Output quantization: scale={output_scale}, zero_point={output_zero_point}")

    return params


def create_model_metadata(
    quant_params: dict,
    target_size: int = 240
) -> dict:
    """
    Create model_metadata.json for mobile deployment.

    Args:
        quant_params: Quantization parameters from extract_quantization_params
        target_size: Input image size

    Returns:
        Dictionary with complete metadata
    """
    metadata = {
        "model_file": "model_qat_int8.tflite",
        "labels_file": "labels.txt",
        "input": {
            "width": target_size,
            "height": target_size,
            "channels": 3,
            "color_space": "RGB",
            "dtype": quant_params['input']['dtype'],
            "quantization": {
                "scale": quant_params['input']['scale'],
                "zero_point": quant_params['input']['zero_point']
            }
        },
        "preprocess": {
            "letterbox": True,
            "keep_aspect_ratio": True,
            "pad_color_rgb": [0, 0, 0],
            "interpolation": "bilinear",
            "resize_rule": "scale = target/max(h,w); new=round(h*scale,w*scale); pad centered"
        },
        "output": {
            "dtype": quant_params['output']['dtype'],
            "quantization": {
                "scale": quant_params['output']['scale'],
                "zero_point": quant_params['output']['zero_point']
            },
            "is_logits": True,
            "softmax_in_model": False
        }
    }

    return metadata
