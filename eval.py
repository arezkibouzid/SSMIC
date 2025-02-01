import argparse
import json
import math
import sys
import time

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms



from compressai.ops import compute_padding

import compressai
import torch 
import torchvision.transforms as transforms 
import tensorflow as tf 
from SSMIC import SSMIC
#from SSMIC_CW import SSMIC_CW


torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)



def collect_images(rootpath: str) -> List[str]:
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(image_files)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, max_val: int = 255
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr-rgb"] = psnr(org, rec).item()
    metrics["ms-ssim-rgb"] = ms_ssim(org, rec, data_range=max_val).item()
    metrics["msssim_db"] = -10. * tf.math.log(1 - metrics["ms-ssim-rgb"]) / tf.math.log(10.)

    
    return metrics


def read_image(filepath: str) -> torch.Tensor:
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    # Calculate the required padding
    if img.size[1]%256==0 : 
        pad_height =0
    else:
        pad_height = max(0, (int(img.size[1]/256)+1)*256 - img.size[1])
        

    if img.size[0]%256==0 : 
        pad_width=0
    else:
        pad_width = max(0, (int(img.size[0]/256)+1)*256 - img.size[0])
        

    # Pad equally on all sides
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padding = (pad_left, pad_top, pad_right, pad_bottom)

    # Define the padding transformation
    pad_transform = transforms.Pad(padding)
    img = pad_transform(img)
    return transforms.ToTensor()(img)

@torch.no_grad()
def inference(model, x):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2

    x_padded = F.pad(x, pad, mode="constant", value=0)
    start = time.time()
    out_enc = (
        model.compress(x_padded)
    )
    enc_time = time.time() - start

    start = time.time()
    out_dec = (
        model.decompress(out_enc["strings"], out_enc["shape"])
    )
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = (sum(len(s[0]) for s in out_enc["strings"])) / num_pixels

    return {
        "psnr-rgb": metrics["psnr-rgb"],
        "ms-ssim-rgb": metrics["ms-ssim-rgb"],
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

from torchvision.utils import save_image
@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = (
        model.forward(x)
    )
    elapsed_time = (time.time() - start)*1000

    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_net["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )
    return {
        "psnr-rgb": metrics["psnr-rgb"],
        "ms-ssim-rgb": metrics["ms-ssim-rgb"],
        "bpp": bpp.item(),
        "encoding_time": out_net["enc-time"]*1000, #'''  # broad estimation
        "decoding_time": out_net["dec-time"]*1000 #'''
    }




def load_checkpoint(arch: str, no_update: bool, checkpoint_path: str) -> nn.Module:
    # update model if need be
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    # compatibility with 'not updated yet' trained nets
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
    model_cls = SSMIC() if arch =="SSMIC" else SSMIC_CW()
    net = model_cls.from_state_dict(state_dict)
    if not no_update:
        net.update(force=True)

    return net.eval()

def eval_model(
    model: nn.Module,
    outputdir: Path,
    inputdir: Path,
    filepaths,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any,
) -> Dict[str, Any]:

    device = next(model.parameters()).device
    metrics = defaultdict(float)
    
    for filepath in filepaths:
        x = read_image(filepath).to(device)
        if not entropy_estimation:
            rv = (
                inference(model, x)
            )
        else:
            rv = (
                inference_entropy_estimation(model, x)
            )
        for k, v in rv.items():
            metrics[k] += v
        if args["per_image"]:
            if not Path(outputdir).is_dir():
                raise FileNotFoundError("Please specify output directory")

            output_subdir = Path(outputdir) / Path(filepath).parent.relative_to(
                inputdir
            )
            output_subdir.mkdir(parents=True, exist_ok=True)
            image_metrics_path = output_subdir / f"{filepath.stem}-{trained_net}.json"
            with image_metrics_path.open("wb") as f:
                output = {
                    "source": filepath.stem,
                    "name": args["architecture"],
                    "description": f"Inference ({description})",
                    "results": rv,
                }
                f.write(json.dumps(output, indent=2).encode())

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def setup_args():
    # Common options.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("dataset", type=str, help="dataset path")

    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=["SSMIC","SSMIC_CW"],
        help="model architecture",
        required=True,
    )

    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )

    parent_parser.add_argument(
        "-d",
        "--output_directory",
        type=str,
        default="",
        help="path of output directory. Optional, required for output json file, results per image. Default will just print the output results.",
    )
    parent_parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="",
        help="output json file name, (default: architecture-entropy_coder.json)",
    )
    parent_parser.add_argument(
        "--per-image",
        action="store_true",
        help="store results for each image of the dataset, separately",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="checkpoint_paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    checkpoint_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Disable the default update of the model entropy parameters before eval",
    )
    return parser


def main(argv):  # noqa: C901
    parser = setup_args()
    args = parser.parse_args(argv)

    if args.source not in ["checkpoint"]:
        print("Error: missing 'checkpoint' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    # create output directory
    if args.output_directory:
        Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    if args.source == "checkpoint":
        runs = args.checkpoint_paths
        opts = (args.architecture, args.no_update)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s} "

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(
                log_fmt.format(*opts, run=(run) )
            )
            sys.stderr.flush()

        model = load_func(*opts, run)

        run_ = run
        cpt_name = Path(run_).name[: -len(".tar.pth")]  # removesuffix() python3.9
        trained_net = f"{cpt_name}-{description}"

        print(f"Using trained model {trained_net}", file=sys.stderr)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
        args_dict = vars(args)
        metrics = eval_model(
            model,
            args.output_directory,
            args.dataset,
            filepaths,
            trained_net=trained_net,
            description=description,
            **args_dict,
        )
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": f"{args.architecture}-{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }
    if args.output_directory:
        output_file = (
            args.output_file
            if args.output_file
            else f"{args.architecture}-{description}"
        )

        with (Path(f"{args.output_directory}/{output_file}").with_suffix(".json")).open(
            "wb"
        ) as f:
            f.write(json.dumps(output, indent=2).encode())

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
