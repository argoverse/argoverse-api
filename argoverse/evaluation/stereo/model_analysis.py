from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("torch failed to import.")

from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis, flop_count_table, parameter_count
from rich.progress import track


def model_analysis(
    model: torch.nn.Module,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_dir: Path,
) -> None:
    """Compute number of parametes, number of activations, flops, and latency (inference time) of a given model.

    Args:
        model: The model to perform the analysis on.
        inputs: Inputs that are passed to the model.
        output_dir: The output directory to save the final txt report with the analysis.

    Raises:
        RuntimeError: If `cuda` is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Expected a 'cuda' device type to perform the analysis.")

    flops = FlopCountAnalysis(model, inputs)
    activations = ActivationCountAnalysis(model, inputs)
    parameters = parameter_count(model)

    count_table = flop_count_table(flops=flops, activations=activations, max_depth=3)
    time_stats = compute_forward_time(model, inputs)

    input_size_str = "\n".join(f"Input size {i}: {input.shape}" for i, input in enumerate(inputs))
    output_str = (
        f"Device name: {torch.cuda.get_device_name(0)} \n"
        f"#Parameters: {parameters['']} \n"
        f"#Flops: {flops.total()} \n"
        f"#Activations: {activations.total()} \n"
        f"Mean forward time (ms): {time_stats['mean_time_ms']} \n"
        f"Std forward time (ms): {time_stats['std_time_ms']} \n"
        f"{input_size_str} \n"
        f"{count_table}"
    )
    print(output_str)

    report_fpath = output_dir / "model_analysis_report.txt"
    with open(report_fpath, "w") as f:
        print(output_str, file=f)

    print(f"Saved report at {report_fpath}.")


def compute_forward_time(
    model: torch.nn.Module,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    num_forward_passes: int = 30,
) -> Dict[str, float]:
    """Compute the model forward time.

    Args:
        model: The model to perform the analysis on.
        inputs: Inputs that are passed to the model.
        num_forward_passes: Number of forward passes to compute the time statistics.

    Returns:
        A dictionary that records the time statistics (mean and std).
    """
    # Init time loggers
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Before we run the profiler, we warm-up CUDA to ensure accurate performance benchmarking
    model(*inputs)

    # Model forward passes
    model.eval()
    timings_ms = []
    with torch.no_grad():
        for _ in track(range(num_forward_passes), description="Forward passing..."):
            start.record()
            model(*inputs)
            end.record()

            # Wait for GPU synchronization
            torch.cuda.synchronize()

            time_ms = start.elapsed_time(end)
            timings_ms.append(time_ms)

    # Compute mean time and its standard deviation
    mean_time_ms = np.sum(timings_ms) / num_forward_passes
    std_time_ms = np.std(timings_ms)

    return {"mean_time_ms": mean_time_ms, "std_time_ms": std_time_ms}


if __name__ == "__main__":

    # Define dummy stereo model
    class DummyStereoModel(torch.nn.Module):  # type: ignore
        def __init__(self) -> None:
            super().__init__()
            self.conv_layer1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
            self.conv_layer2 = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3)

        def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
            output1 = self.conv_layer1(input1)
            output2 = self.conv_layer1(input2)
            output = self.conv_layer2(torch.cat((output1, output2), axis=1))
            return output

    model = DummyStereoModel().cuda()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Define dummy inputs
    input_shape = (1, 3, 10, 10)  # E.g., batch_size x num_channels x height x width
    input1 = torch.randn(input_shape, device=device, dtype=dtype)
    input2 = torch.randn(input_shape, device=device, dtype=dtype)
    dummy_input = (input1, input2)

    report_dir = Path("/home/ubuntu/")

    # Example of output report:
    # Device name: Tesla V100-SXM2-32GB
    # #Parameters: 1315
    # #Flops: 86400
    # #Activations: 2156
    # Mean forward time (ms): 0.21968213319778443
    # Std forward time (ms): 0.027638812549933046
    # Input size 0: torch.Size([1, 3, 10, 10])
    # Input size 1: torch.Size([1, 3, 10, 10])
    # | module               | #parameters or shape   | #flops   | #activations   |
    # |:---------------------|:-----------------------|:---------|:---------------|
    # | model                | 1.315K                 | 86.4K    | 2.156K         |
    # |  conv_layer1         |  0.448K                |  55.296K |  2.048K        |
    # |   conv_layer1.weight |   (16, 3, 3, 3)        |          |                |
    # |   conv_layer1.bias   |   (16,)                |          |                |
    # |  conv_layer2         |  0.867K                |  31.104K |  0.108K        |
    # |   conv_layer2.weight |   (3, 32, 3, 3)        |          |                |
    # |   conv_layer2.bias   |   (3,)                 |          |                |
    model_analysis(model, dummy_input, report_dir)
