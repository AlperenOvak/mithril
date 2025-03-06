import argparse
import collections
import glob
import json
from pathlib import Path
import shutil

import torch

def llama(model_path, dtype: str):
    SHARD_FIRST = ["wv", "wq", "wk", "w1", "w3", "output"]
    SHARD_SECOND = ["tok_embeddings", "wo", "w2"]
    SHARD_WEIGHTS = set(SHARD_FIRST + SHARD_SECOND)

    def shard_key(k):
        keys = k.split(".")
        if len(keys) < 2:
            return None
        return keys[-2]

    def unshard(k, v):
        wn = shard_key(k)
        if wn not in SHARD_WEIGHTS:
            return v
        elif wn in SHARD_FIRST:
            axis = 0
        elif wn in SHARD_SECOND:
            axis = 1
        else:
            raise ValueError(f"Invalid weight name: {wn}")
        return torch.cat(v, dim=axis)

    torch_files = glob.glob(str(model_path / "consolidated.*.pth"))
    weights = collections.defaultdict(list)

    for wf in torch_files:
        state = torch.load(wf, map_location=torch.device("cpu"))
        for k, v in state.items():
            v = v.to(dtype=getattr(torch, dtype))
            state[k] = None  # free memory
            if shard_key(k) in SHARD_WEIGHTS:
                weights[k].append(v)
            else:
                weights[k] = v

    for k, v in weights.items():
        weights[k] = unshard(k, v)

    with open(model_path / "params.json", "r") as f:
        params = json.load(f)

    return weights, params


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    config = sanitize_config(config, weights)
    model = Llama(ModelArgs(**config))
    model.update(tree_unflatten(list(weights.items())))

    # Update the config:
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
    parser.add_argument(
        "--torch-path",
        type=str,
        help="Path to the PyTorch model.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Path to save the MLX model.",
    )
    parser.add_argument(
        "--model-name",
        help=(
            "Name of the model to convert. Use 'llama' for models in the "
            "Llama family distributed by Meta including Llama 1, Llama 2, "
            "Code Llama, and Llama chat."
        ),
        choices=["tiny_llama", "llama"],
        default="llama",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        help="Generate a quantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dtype",
        help="dtype for loading the torch model and input for quantization or saving the converted model. "
        "The original weights are stored in bfloat16.",
        type=str,
        default="float16",
    )

    args = parser.parse_args()

    torch_path = Path(args.torch_path)
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading")
    weights, params = globals()[args.model_name](torch_path, dtype=args.dtype)
    sorted_keys = sorted(weights.keys())
    for key in sorted_keys:
        print(f"{key}: {weights[key].shape}")
    print("***********")
    params["model_type"] = "llama"
    if args.quantize:
        print("[INFO] Quantizing")
        weights, params = quantize(weights, params, args)

    print("[INFO] Saving")
    shutil.copyfile(
        str(torch_path / "tokenizer.model"),
        str(mlx_path / "tokenizer.model"),
    )
    shards = make_shards(weights)

    if len(shards) == 1:
        torch.save(shards[0], mlx_path / "weights.pth")
    else:
        for i, shard in enumerate(shards):
            torch.save(shard, mlx_path / f"weights.{i:02d}.pth")

    # Save configuration
    with open(mlx_path / "config.json", "w") as fid:
        json.dump(params, fid, indent=4)