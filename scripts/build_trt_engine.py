"""Build a TensorRT engine from an ONNX model.

Converts an exported ONNX model to an optimized TensorRT engine
with support for:
    - Dynamic batch size
    - Dynamic sequence length
    - FP16 precision (if GPU supports it)
    - Optimization profiles for dynamic shapes

Usage:
    python3 scripts/build_trt_engine.py \
        --onnx models/simple_model.onnx \
        --output models/simple_model.engine
    python3 scripts/build_trt_engine.py \
        --onnx models/simple_model.onnx \
        --output models/simple_model_fp16.engine --fp16
"""
from __future__ import annotations

import argparse
import logging
import os
import time

logger = logging.getLogger(__name__)


def build_trt_engine(
    onnx_path: str,
    output_path: str,
    fp16: bool = False,
    workspace_size: int = 1 << 30,
    min_batch: int = 1,
    opt_batch: int = 4,
    max_batch: int = 16,
    min_seq_len: int = 1,
    opt_seq_len: int = 32,
    max_seq_len: int = 128,
) -> str:
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to the input ONNX model.
        output_path: Path to save the TensorRT engine.
        fp16: Whether to enable FP16 precision.
        workspace_size: GPU workspace size in bytes (default 1 GB).
        min_batch: Minimum batch size for optimization profile.
        opt_batch: Optimal batch size for optimization profile.
        max_batch: Maximum batch size for optimization profile.
        min_seq_len: Minimum sequence length for optimization profile.
        opt_seq_len: Optimal sequence length for optimization profile.
        max_seq_len: Maximum sequence length for optimization profile.

    Returns:
        Path to the built TensorRT engine.

    Raises:
        ImportError: If tensorrt is not installed.
        FileNotFoundError: If the ONNX model file is not found.
        RuntimeError: If engine building fails.
    """
    import tensorrt as trt

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    logger.info(f"Building TensorRT engine from: {onnx_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"FP16: {fp16}")
    logger.info(f"Batch range: [{min_batch}, {opt_batch}, {max_batch}]")
    logger.info(f"Seq len range: [{min_seq_len}, {opt_seq_len}, {max_seq_len}]")

    # TensorRT logger
    trt_logger = trt.Logger(trt.Logger.WARNING)

    # Builder + network (explicit batch for dynamic shapes)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    # Parse ONNX model
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = []
            for i in range(parser.num_errors):
                errors.append(str(parser.get_error(i)))
            raise RuntimeError(
                f"Failed to parse ONNX model: {'; '.join(errors)}"
            )

    logger.info(
        f"Network has {network.num_layers} layers, "
        f"{network.num_inputs} inputs, {network.num_outputs} outputs"
    )

    # Build config
    config = builder.create_builder_config()

    # Workspace size (compatible with different TensorRT versions)
    if hasattr(config, "set_memory_pool_limit"):
        # TensorRT 8.4+
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_size
        )
    elif hasattr(config, "max_workspace_size"):
        # Older TensorRT versions
        config.max_workspace_size = workspace_size
    else:
        logger.warning("Could not set workspace size (unknown API)")

    # Precision
    if fp16:
        if not builder.platform_has_fast_fp16:
            logger.warning(
                "Platform does not have fast FP16 support. "
                "Building with FP32."
            )
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 precision enabled")

    # Optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name

    min_shape = (min_batch, min_seq_len)
    opt_shape = (opt_batch, opt_seq_len)
    max_shape = (max_batch, max_seq_len)

    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    logger.info(
        f"Optimization profile: {input_name} "
        f"min={min_shape}, opt={opt_shape}, max={max_shape}"
    )

    # Build engine
    start_time = time.time()
    logger.info("Building TensorRT engine (this may take a while)...")

    # TensorRT 8.4+ uses build_serialized_network
    if hasattr(builder, "build_serialized_network"):
        serialized_engine = builder.build_serialized_network(
            network, config
        )
    else:
        # Older versions use build_engine + serialize
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        serialized_engine = engine.serialize()

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine to file
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    build_time = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)

    logger.info(f"TensorRT engine built successfully: {build_time:.1f}s")
    logger.info(f"Engine size: {file_size:.2f} MB")
    logger.info(f"Saved to: {output_path}")

    return output_path


def verify_trt_engine(engine_path: str) -> bool:
    """Verify a TensorRT engine can be loaded and run.

    Uses PyTorch CUDA tensors for device memory (no pycuda needed).

    Args:
        engine_path: Path to the TensorRT engine file.

    Returns:
        True if verification passed.
    """
    try:
        import tensorrt as trt
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping TRT engine verification")
            return False

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            logger.error("Failed to deserialize TensorRT engine")
            return False

        context = engine.create_execution_context()

        # Find input/output bindings
        input_idx = -1
        output_idx = -1
        import tensorrt as trt
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_idx = i
                input_name = name
            else:
                output_idx = i
                output_name = name

        if input_idx < 0 or output_idx < 0:
            logger.error("Could not find input/output bindings")
            return False

        # Test with a small input
        input_shape = (1, 8)
        input_np = np.random.randint(
            0, 4096, input_shape, dtype=np.int32
        )

        # Allocate via torch CUDA tensors
        d_input = torch.from_numpy(input_np).cuda().int()

        # Set shape
        context.set_input_shape(input_name, input_shape)
        output_shape = tuple(context.get_tensor_shape(output_name))
        d_output = torch.zeros(
            output_shape, dtype=torch.float32, device="cuda"
        )

        context.set_tensor_address(input_name, d_input.data_ptr())
        context.set_tensor_address(output_name, d_output.data_ptr())

        # Run inference
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        torch.cuda.synchronize()

        # Copy output back to verify
        output_np = d_output.cpu().numpy()
        assert output_np.shape == output_shape, (
            f"Shape mismatch: got {output_np.shape}, expected {output_shape}"
        )

        logger.info(
            f"TensorRT engine verification PASSED "
            f"(input={input_shape}, output={output_shape})"
        )
        return True

    except ImportError as e:
        logger.warning(f"Cannot verify engine (missing dependency): {e}")
        return False
    except Exception as e:
        logger.error(f"TensorRT engine verification failed: {e}")
        return False


def main():
    """Main entry point for TensorRT engine building."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Build TensorRT engine from ONNX model"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="models/simple_model.onnx",
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/simple_model.engine",
        help="Path to output TensorRT engine",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision (if GPU supports it)",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=1 << 30,
        help="GPU workspace size in bytes (default 1 GB)",
    )
    parser.add_argument(
        "--min-batch", type=int, default=1, help="Min batch size"
    )
    parser.add_argument(
        "--opt-batch", type=int, default=4, help="Optimal batch size"
    )
    parser.add_argument(
        "--max-batch", type=int, default=16, help="Max batch size"
    )
    parser.add_argument(
        "--min-seq-len", type=int, default=1, help="Min sequence length"
    )
    parser.add_argument(
        "--opt-seq-len", type=int, default=32, help="Optimal sequence length"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=128, help="Max sequence length"
    )
    args = parser.parse_args()

    try:
        output_path = build_trt_engine(
            onnx_path=args.onnx,
            output_path=args.output,
            fp16=args.fp16,
            workspace_size=args.workspace,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
            min_seq_len=args.min_seq_len,
            opt_seq_len=args.opt_seq_len,
            max_seq_len=args.max_seq_len,
        )
        verify_trt_engine(output_path)
        print()
        print(f"Build complete: {output_path}")
    except ImportError:
        print("Error: TensorRT is not installed. "
              "Install with: pip install tensorrt")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
