import argparse
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite

from model import final_net
from test_dataset import dehaze_test_dataset


def resolve_device(preferred):
    if preferred.startswith("cuda"):
        ok = False
        try:
            if torch.cuda.is_available():
                _ = torch.empty(1, device=preferred)
                ok = True
        except Exception:
            ok = False
        if not ok:
            print(f"Requested device {preferred} is not usable; falling back to cpu.")
            return "cpu"
    return preferred


def parse_args():
    parser = argparse.ArgumentParser(description="Shadow removal inference")
    parser.add_argument("--test_dir", type=str, default="/NTIRE2026/C4_ShdwRem/valid")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("-test_batch_size", help="Set testing batch size", default=1, type=int)

    parser.add_argument("--remove_ckpt", type=str, default=os.path.join("weights", "shadowremoval.pkl"))
    parser.add_argument("--enhance_ckpt", type=str, default=os.path.join("weights", "refinement.pkl"))
    parser.add_argument("--full_ckpt", type=str, default="", help="Single full checkpoint")
    parser.add_argument(
        "--ensemble_ckpts",
        type=str,
        default="",
        help="Comma-separated full checkpoints for output ensembling",
    )

    parser.add_argument("--tta", action="store_true", help="Enable x8 test-time augmentation")
    parser.add_argument("--tile", type=int, default=0, help="Tile size (0 = full-image inference)")
    parser.add_argument("--tile_overlap", type=int, default=32, help="Tile overlap in pixels")
    parser.add_argument("--min_divisible", type=int, default=64, help="Pad image so H,W are divisible by this")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_single_model(device: str, full_ckpt: str, remove_ckpt: str, enhance_ckpt: str) -> torch.nn.Module:
    model = final_net().to(device)
    if full_ckpt:
        model.load_state_dict(torch.load(full_ckpt, map_location=device), strict=True)
        print(f"Loaded full model: {full_ckpt}")
    else:
        model.remove_model.load_state_dict(torch.load(remove_ckpt, map_location=device), strict=True)
        model.enhancement_model.load_state_dict(torch.load(enhance_ckpt, map_location=device), strict=True)
        print(f"Loaded removal model: {remove_ckpt}")
        print(f"Loaded enhancement model: {enhance_ckpt}")
    model.eval()
    return model


def load_models(args) -> List[torch.nn.Module]:
    if args.ensemble_ckpts.strip():
        paths = [p.strip() for p in args.ensemble_ckpts.split(",") if p.strip()]
        if not paths:
            raise RuntimeError("--ensemble_ckpts was provided but no valid checkpoint paths were found")
        models = [load_single_model(args.device, p, "", "") for p in paths]
        print(f"Output ensembling enabled with {len(models)} checkpoints")
        return models

    model = load_single_model(args.device, args.full_ckpt, args.remove_ckpt, args.enhance_ckpt)
    return [model]


def pad_to_multiple(x: torch.Tensor, divisor: int):
    if divisor <= 1:
        return x, (0, 0)
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def unpad(x: torch.Tensor, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        x = x[..., :-pad_h, :]
    if pad_w > 0:
        x = x[..., :, :-pad_w]
    return x


def forward_tiled(model: torch.nn.Module, inp: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    b, c, h, w = inp.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return model(inp)

    stride = max(1, tile - overlap)
    out = torch.zeros((b, c, h, w), device=inp.device, dtype=inp.dtype)
    norm = torch.zeros((b, 1, h, w), device=inp.device, dtype=inp.dtype)

    for y in range(0, h, stride):
        y0 = min(y, h - tile)
        y1 = y0 + tile
        for x in range(0, w, stride):
            x0 = min(x, w - tile)
            x1 = x0 + tile
            patch = inp[..., y0:y1, x0:x1]
            pred = model(patch)
            out[..., y0:y1, x0:x1] += pred
            norm[..., y0:y1, x0:x1] += 1.0

    return out / torch.clamp(norm, min=1e-6)


def apply_transform(x: torch.Tensor, op: int) -> torch.Tensor:
    if op == 0:
        return x
    if op == 1:
        return torch.flip(x, dims=[-1])
    if op == 2:
        return torch.flip(x, dims=[-2])
    if op == 3:
        return torch.rot90(x, 1, dims=[-2, -1])
    if op == 4:
        return torch.rot90(x, 2, dims=[-2, -1])
    if op == 5:
        return torch.rot90(x, 3, dims=[-2, -1])
    if op == 6:
        return torch.flip(torch.rot90(x, 1, dims=[-2, -1]), dims=[-1])
    if op == 7:
        return torch.flip(torch.rot90(x, 1, dims=[-2, -1]), dims=[-2])
    raise ValueError(f"Unknown transform op: {op}")


def invert_transform(x: torch.Tensor, op: int) -> torch.Tensor:
    if op == 0:
        return x
    if op == 1:
        return torch.flip(x, dims=[-1])
    if op == 2:
        return torch.flip(x, dims=[-2])
    if op == 3:
        return torch.rot90(x, 3, dims=[-2, -1])
    if op == 4:
        return torch.rot90(x, 2, dims=[-2, -1])
    if op == 5:
        return torch.rot90(x, 1, dims=[-2, -1])
    if op == 6:
        return torch.rot90(torch.flip(x, dims=[-1]), 3, dims=[-2, -1])
    if op == 7:
        return torch.rot90(torch.flip(x, dims=[-2]), 3, dims=[-2, -1])
    raise ValueError(f"Unknown transform op: {op}")


def model_forward(model: torch.nn.Module, inp: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    if tile > 0:
        return forward_tiled(model, inp, tile, overlap)
    return model(inp)


def predict_one_model(model: torch.nn.Module, inp: torch.Tensor, tta: bool, tile: int, overlap: int):
    if not tta:
        return model_forward(model, inp, tile, overlap)

    preds = []
    for op in range(8):
        aug = apply_transform(inp, op)
        pred = model_forward(model, aug, tile, overlap)
        preds.append(invert_transform(pred, op))
    return torch.stack(preds, dim=0).mean(dim=0)


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    test_dataset = dehaze_test_dataset(args.test_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Device: {args.device}")
    models = load_models(args)

    with torch.no_grad():
        for input_tensor, name in test_loader:
            input_tensor = input_tensor.to(args.device)
            padded, pad_hw = pad_to_multiple(input_tensor, args.min_divisible)

            model_preds = []
            for model in models:
                pred = predict_one_model(model, padded, args.tta, args.tile, args.tile_overlap)
                model_preds.append(pred)

            output = torch.stack(model_preds, dim=0).mean(dim=0)
            output = unpad(output, pad_hw)
            output = torch.clamp(output, 0, 1)

            imwrite(output, os.path.join(args.output_dir, name[0]), value_range=(0, 1))


if __name__ == "__main__":
    main()
