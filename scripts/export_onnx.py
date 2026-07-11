"""Export a simple transformer model to ONNX format.

Exports a lightweight GPT-style model with:
    - Dynamic batch size
    - Dynamic sequence length
    - Optional FP16 precision

The exported model is used by ONNXExecutor for inference comparison
against the PyTorch (simulated) Executor.

Usage:
    python3 scripts/export_onnx.py --output models/simple_model.onnx
    python3 scripts/export_onnx.py --output models/simple_model.onnx --fp16
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)


def create_simple_model():
    """Create a simple PyTorch model for ONNX export.

    The model simulates a minimal transformer layer:
        input_ids -> embedding -> linear -> output_logits

    This is intentionally simple - the purpose is to have a real ONNX
    model that ONNXExecutor can load and run, not to replicate a full LLM.
    """
    import torch
    import torch.nn as nn

    class SimpleTransformerModel(nn.Module):
        """Minimal transformer-style model for ONNX export.

        Attributes:
            embedding: Token embedding layer.
            linear: Linear projection layer.
            layer_norm: Layer normalization.
        """

        def __init__(self, vocab_size: int = 1000, hidden_size: int = 128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.linear = nn.Linear(hidden_size, hidden_size)
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                input_ids: Token IDs of shape (batch_size, seq_len).

            Returns:
                Output logits of shape (batch_size, seq_len, vocab_size).
            """
            x = self.embedding(input_ids)
            x = self.linear(x)
            x = self.layer_norm(x)
            logits = self.output(x)
            return logits

    # vocab_size=4096 covers the synthetic token ID range used across
    # all experiments (token IDs up to ~3000). This avoids out-of-bounds
    # errors in the ONNX Gather node at inference time.
    model = SimpleTransformerModel(vocab_size=4096, hidden_size=128)
    model.eval()
    return model


def export_to_onnx(
    output_path: str,
    fp16: bool = False,
) -> str:
    """Export the simple model to ONNX format.

    Args:
        output_path: Path to save the ONNX model.
        fp16: Whether to use FP16 precision (if supported).

    Returns:
        Path to the saved ONNX model.
    """
    import torch

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model = create_simple_model()

    # Create dummy input with dynamic shapes
    dummy_input = torch.randint(0, 1000, (1, 8), dtype=torch.long)

    # Dynamic axes: batch_size and sequence_length are dynamic
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    logger.info(f"Exporting model to ONNX: {output_path}")
    start_time = time.time()

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    export_time = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"ONNX export complete: {export_time:.2f}s, {file_size:.2f} MB")

    # Optionally convert to FP16
    if fp16:
        try:
            import onnx
            from onnxconverter_common import float16

            fp16_path = output_path.replace(".onnx", "_fp16.onnx")
            onnx_model = onnx.load(output_path)
            fp16_model = float16.convert_float_to_float16(onnx_model)
            onnx.save(fp16_model, fp16_path)
            logger.info(f"FP16 model saved: {fp16_path}")
            return fp16_path
        except ImportError:
            logger.warning(
                "FP16 conversion requires onnxconverter-common. "
                "Skipping FP16 export."
            )

    return output_path


def verify_onnx_model(model_path: str) -> bool:
    """Verify the ONNX model can be loaded and run.

    Args:
        model_path: Path to the ONNX model.

    Returns:
        True if verification passed.
    """
    try:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Test with different shapes (dynamic axes)
        for batch_size, seq_len in [(1, 8), (2, 16), (4, 32)]:
            input_ids = np.random.randint(
                0, 4096, (batch_size, seq_len), dtype=np.int64
            )
            outputs = session.run(None, {"input_ids": input_ids})
            logits = outputs[0]
            expected_shape = (batch_size, seq_len, 4096)
            assert logits.shape == expected_shape, (
                f"Shape mismatch: got {logits.shape}, expected {expected_shape}"
            )

        logger.info("ONNX model verification PASSED")
        return True

    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False


def main():
    """Main entry point for ONNX export."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "--output",
        type=str,
        default="models/simple_model.onnx",
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export in FP16 precision (if supported)",
    )
    args = parser.parse_args()

    output_path = export_to_onnx(args.output, fp16=args.fp16)
    verify_onnx_model(output_path)
    print(f"\nExport complete: {output_path}")


if __name__ == "__main__":
    main()
