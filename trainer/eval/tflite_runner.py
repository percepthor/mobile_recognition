"""TFLite model inference runner."""
import logging
from typing import Tuple, List
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm


logger = logging.getLogger(__name__)


class TFLiteRunner:
    """Wrapper for TFLite model inference."""

    def __init__(self, tflite_path: str):
        """
        Initialize TFLite runner.

        Args:
            tflite_path: Path to .tflite model file
        """
        self.tflite_path = tflite_path
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        # Extract quantization params
        input_quant = self.input_details.get('quantization_parameters', {})
        self.input_scale = input_quant.get('scales', [1.0])[0]
        self.input_zero_point = input_quant.get('zero_points', [0])[0]
        self.input_dtype = self.input_details['dtype']

        output_quant = self.output_details.get('quantization_parameters', {})
        self.output_scale = output_quant.get('scales', [1.0])[0]
        self.output_zero_point = output_quant.get('zero_points', [0])[0]
        self.output_dtype = self.output_details['dtype']

        logger.info(f"TFLite model loaded: {tflite_path}")
        logger.info(f"Input: dtype={self.input_dtype}, scale={self.input_scale}, zp={self.input_zero_point}")
        logger.info(f"Output: dtype={self.output_dtype}, scale={self.output_scale}, zp={self.output_zero_point}")

    def quantize_input(self, float_input: np.ndarray) -> np.ndarray:
        """
        Quantize float input to model's input dtype.

        Args:
            float_input: Float array (typically [0, 255] range)

        Returns:
            Quantized array in input dtype
        """
        # Quantize: q = round(x / scale + zero_point)
        quantized = np.round(float_input / self.input_scale + self.input_zero_point)

        # Clamp to dtype range
        if self.input_dtype == np.int8:
            quantized = np.clip(quantized, -128, 127).astype(np.int8)
        elif self.input_dtype == np.uint8:
            quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        else:
            quantized = quantized.astype(self.input_dtype)

        return quantized

    def dequantize_output(self, quantized_output: np.ndarray) -> np.ndarray:
        """
        Dequantize model output to float logits.

        Args:
            quantized_output: Quantized output from model

        Returns:
            Float logits
        """
        # Dequantize: x = (q - zero_point) * scale
        return (quantized_output.astype(np.float32) - self.output_zero_point) * self.output_scale

    def predict(self, preprocessed_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a single preprocessed image.

        Args:
            preprocessed_image: Preprocessed image array [H, W, 3] in float32 [0, 255]

        Returns:
            Tuple of (logits, probabilities)
        """
        # Ensure correct shape
        if len(preprocessed_image.shape) == 3:
            preprocessed_image = preprocessed_image[np.newaxis, :]

        # Quantize input
        quantized_input = self.quantize_input(preprocessed_image)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details['index'], quantized_input)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor (quantized)
        quantized_output = self.interpreter.get_tensor(self.output_details['index'])[0]

        # Dequantize to get logits
        logits = self.dequantize_output(quantized_output)

        # Apply softmax to get probabilities
        probabilities = self._softmax(logits)

        return logits, probabilities

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """
        Compute softmax.

        Args:
            logits: Logit array

        Returns:
            Probability array
        """
        # Subtract max for numerical stability
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)


def evaluate_tflite_on_dataset(
    tflite_path: str,
    data_dir: str,
    split_df: pd.DataFrame,
    target_size: int = 240
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate TFLite model on a dataset split.

    Args:
        tflite_path: Path to .tflite model
        data_dir: Root data directory
        split_df: DataFrame with 'filepath', 'label_index' columns
        target_size: Image size

    Returns:
        Tuple of (y_true, y_pred, probabilities_max)
        - y_true: True labels
        - y_pred: Predicted labels (argmax)
        - probabilities_max: Max probabilities for each sample
    """
    from trainer.data.preprocess import preprocess_from_path

    runner = TFLiteRunner(tflite_path)

    y_true = []
    y_pred = []
    prob_max = []
    all_probs = []

    logger.info(f"Evaluating TFLite model on {len(split_df)} samples")

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Evaluating"):
        # Load and preprocess image
        image_path = f"{data_dir}/{row['filepath']}"
        preprocessed = preprocess_from_path(
            image_path,
            target_size=target_size,
            return_range_0_255=True
        ).numpy()

        # Run inference
        logits, probs = runner.predict(preprocessed)

        # Store results
        y_true.append(row['label_index'])
        y_pred.append(np.argmax(probs))
        prob_max.append(np.max(probs))
        all_probs.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    prob_max = np.array(prob_max)
    all_probs = np.array(all_probs)

    accuracy = np.mean(y_true == y_pred)
    logger.info(f"Accuracy: {accuracy:.4f}")

    return y_true, y_pred, prob_max, all_probs
