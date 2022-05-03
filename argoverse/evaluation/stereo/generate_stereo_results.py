from pathlib import Path

import cv2
import numpy as np
from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis, flop_count_table, parameter_count
from tqdm import tqdm

try:
    import torch
    import torchvision
except ImportError:
    raise ImportError("torch failed to import.")

from argoverse.evaluation.competition_util import generate_stereo_zip


class ArgoverseStereoDataset(torch.utils.data.Dataset):  # type: ignore
    """Minimal Argoverse Stereo dataset.

    Args:
        data_dir: The directory of the Argoverse Stereo dataset.
        split_name: The split of the detaset (train, val, or test).
        transforms: Transforms (torchvision.transforms) to be applied to the input data.
    """

    def __init__(
        self,
        data_dir: Path,
        split_name: str = "test",
        transforms: torchvision.transforms = None,
    ):
        left_stereo_img_fpaths = sorted(
            (data_dir / "rectified_stereo_images_v1.1" / split_name).glob("*/stereo_front_left_rect/*.jpg")
        )
        right_stereo_img_fpaths = sorted(
            (data_dir / "rectified_stereo_images_v1.1" / split_name).glob("*/stereo_front_right_rect/*.jpg")
        )

        self.stereo_img_fpaths = (left_stereo_img_fpaths, right_stereo_img_fpaths)

        self.transforms = transforms

    def __len__(self):  # type: ignore
        return len(self.stereo_img_fpaths[0])

    def __getitem__(self, idx):  # type: ignore
        left_stereo_img_fpath = str(self.stereo_img_fpaths[0][idx])
        right_stereo_img_fpath = str(self.stereo_img_fpaths[1][idx])

        left_stereo_img = torchvision.io.read_image(left_stereo_img_fpath) / 255.0
        right_stereo_img = torchvision.io.read_image(right_stereo_img_fpath) / 255.0

        if self.transforms:
            left_stereo_img = self.transforms(left_stereo_img)
            right_stereo_img = self.transforms(right_stereo_img)

        return {
            "left_stereo_img": left_stereo_img,
            "right_stereo_img": right_stereo_img,
            "left_stereo_img_fpath": left_stereo_img_fpath,
            "right_stereo_img_fpath": right_stereo_img_fpath,
        }


@torch.no_grad()  # type: ignore
def generate_stereo_results(
    model: torch.nn.Module,
    data_dir: Path,
    output_dir: Path,
    transforms: torchvision.transforms = None,
) -> None:
    """Generate the stereo model predictions, benchmark the model, and create the submission file.
    Compute number of parametes, number of activations, flops, and latency (inference time).

    Args:
        model: The model to get predictions and benchmarking.
        data_dir: The directory of the Argoverse Stereo dataset.
        output_dir: The output directory to save the results.
        transforms: Transforms (torchvision.transforms) to be applied to the input data.

    Raises:
        RuntimeError: If `cuda` is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Expected a 'cuda' device type to perform the analysis.")

    test_dataset = ArgoverseStereoDataset(data_dir=data_dir, transforms=transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    model.cuda()
    model.eval()

    # Init time loggers
    time_start = torch.cuda.Event(enable_timing=True)
    time_end = torch.cuda.Event(enable_timing=True)

    timings_ms = []

    print("Forward passing the test dataset, computing latency, and saving predictions...")
    for i, sample in enumerate(tqdm(test_dataloader)):
        left_stereo_img = sample["left_stereo_img"].cuda()
        right_stereo_img = sample["right_stereo_img"].cuda()

        inputs = (left_stereo_img, right_stereo_img)

        if i == 0:
            # Warm-up CUDA to ensure accurate benchmarking
            model(*inputs)

        time_start.record()
        left_disparity_pred = model(*inputs)
        time_end.record()

        # Wait for GPU synchronization
        torch.cuda.synchronize()

        time_ms = time_start.elapsed_time(time_end)
        timings_ms.append(time_ms)

        final_img_height = 2056
        final_img_width = 2464
        output_shape = (final_img_height, final_img_width)
        if left_disparity_pred.shape != output_shape:
            transform = torchvision.transforms.Resize(size=output_shape, antialias=True)
            left_disparity_pred = transform(left_disparity_pred)

        # Encode the disparity values to uint16
        left_disparity_pred = left_disparity_pred[0].squeeze(0).cpu().numpy()
        left_disparity_pred = np.uint16(left_disparity_pred * 256.0)

        left_stereo_img_fpath = sample["left_stereo_img_fpath"][0]
        timestamp = int(Path(left_stereo_img_fpath).stem.split("_")[-1])
        log_id = left_stereo_img_fpath.split("/")[-3]

        save_dir_disp = output_dir / log_id
        save_dir_disp.mkdir(parents=True, exist_ok=True)
        left_disparity_fname = f"{save_dir_disp}/disparity_{timestamp}.png"

        # Write PNG file to disk
        cv2.imwrite(left_disparity_fname, left_disparity_pred)

    # Compute mean time and its standard deviation
    mean_time_ms = np.mean(timings_ms)
    std_time_ms = np.std(timings_ms)

    print("Computing number of flops, activations, and parameters...")
    flops = FlopCountAnalysis(model, inputs)
    activations = ActivationCountAnalysis(model, inputs)
    parameters = parameter_count(model)

    count_table = flop_count_table(flops=flops, activations=activations, max_depth=3)

    input_size_str = "\n".join(f"Input size {i}: {input.shape}" for i, input in enumerate(inputs))
    output_str = (
        f"Device name: {torch.cuda.get_device_name(0)} \n"
        f"#Parameters: {parameters['']} \n"
        f"#Flops: {flops.total()} \n"
        f"#Activations: {activations.total()} \n"
        f"Mean forward time (ms): {mean_time_ms} \n"
        f"Std forward time (ms): {std_time_ms} \n"
        f"{input_size_str} \n"
        f"{count_table}"
    )
    print(output_str)

    report_fpath = output_dir / "model_analysis_report.txt"
    with open(report_fpath, "w") as f:
        print(output_str, file=f)
    print(f"Saved report at {report_fpath}.")

    print("Checking outputs and generating submission file. Please wait...")
    generate_stereo_zip(output_dir, output_dir.parent)


if __name__ == "__main__":

    # Define dummy stereo model
    class DummyStereoModel(torch.nn.Module):  # type: ignore
        def __init__(self) -> None:
            super().__init__()
            self.conv_layer1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
            self.conv_layer2 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

        def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
            output1 = self.conv_layer1(input1)
            output2 = self.conv_layer1(input2)
            output = self.conv_layer2(torch.cat((output1, output2), axis=1))
            return output

    model = DummyStereoModel()

    # Example of paths
    data_dir = Path("/data/datasets/stereo/argoverse1/")
    output_dir = Path("/data/datasets/stereo/submission/stereo_output")

    # Example of transforms to be applied to the input data
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(528, 624)),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),  # ResNet mean image normalization
                std=(0.229, 0.224, 0.225),  # ResNet std image normalization
            ),
        ]
    )

    # Example of output report:
    # Device name: Tesla V100-SXM2-16GB
    # #Parameters: 737
    # #Flops: 379551744
    # #Activations: 10872576
    # Mean forward time (ms): 1.0553814383049989
    # Std forward time (ms): 0.05229927585160601
    # Input size 0: torch.Size([1, 3, 528, 624])
    # Input size 1: torch.Size([1, 3, 528, 624])
    # | module               | #parameters or shape   | #flops   | #activations   |
    # |:---------------------|:-----------------------|:---------|:---------------|
    # | model                | 0.737K                 | 0.38G    | 10.873M        |
    # |  conv_layer1         |  0.448K                |  0.285G  |  10.543M       |
    # |   conv_layer1.weight |   (16, 3, 3, 3)        |          |                |
    # |   conv_layer1.bias   |   (16,)                |          |                |
    # |  conv_layer2         |  0.289K                |  94.888M |  0.329M        |
    # |   conv_layer2.weight |   (1, 32, 3, 3)        |          |                |
    # |   conv_layer2.bias   |   (1,)                 |          |                |
    generate_stereo_results(model, data_dir, output_dir, transforms)
