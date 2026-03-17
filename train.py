import os
import random
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile, UnidentifiedImageError
import torchvision.transforms as transforms
from torchvision.utils import save_image
from model import final_net
from test_dataset import dehaze_test_dataset
from vainF_ssim import MS_SSIM
from utils import Vgg16


def resolve_device(preferred="cuda"):
    if preferred.startswith("cuda"):
        ok = False
        try:
            if torch.cuda.is_available():
                _ = torch.empty(1, device=preferred)
                ok = True
        except Exception:
            ok = False
        if not ok:
            print(f"Requested device {preferred} is not usable; falling back to cpu.", flush=True)
            return torch.device("cpu")
    return torch.device(preferred)


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="Safer shadow-removal fine-tuning with best-checkpoint tracking")
    parser.add_argument("--train_inp", type=str, default="/NTIRE2026/C4_ShdwRem/train_inp")
    parser.add_argument("--train_gt", type=str, default="/NTIRE2026/C4_ShdwRem/train_gt")
    parser.add_argument("--valid_dir", type=str, default="/NTIRE2026/C4_ShdwRem/valid")
    parser.add_argument("--paired_eval_root", type=str, default="/NTIRE2026/C4_ShdwRem/ntire26_shadow_removal_train")
    parser.add_argument("--paired_eval_images", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patch_size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_grad_checkpointing", action="store_true")
    parser.add_argument("--verify_dataset_on_start", action="store_true")
    parser.add_argument("--init_full_ckpt", type=str, default="weights/epoch_200.pth")
    parser.add_argument("--out_dir", type=str, default="weights")
    return parser.parse_args()


def run_validation(model, device, epoch, valid_dir):
    val_dataset = dehaze_test_dataset(valid_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    val_output_dir = f"val_epoch_{epoch + 1}"
    os.makedirs(val_output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for input_tensor, name in val_loader:
            input_tensor = input_tensor.to(device, non_blocking=True)
            output = model(input_tensor)
            output = torch.clamp(output, 0, 1)
            save_image(output, os.path.join(val_output_dir, name[0]))
    model.train()


# ---------------- Dataset ----------------
class ShadowDataset(Dataset):

    def __init__(self, inp_dir, gt_dir, patch_size=256, verify_dataset_on_start=False):
        self.inp_dir = inp_dir
        self.gt_dir = gt_dir
        all_files = sorted(os.listdir(inp_dir))
        self.files = []
        self.transform = transforms.ToTensor()
        self.patch_size = patch_size
        skipped = 0

        print(f"Scanning dataset directory: {inp_dir}", flush=True)
        for name in all_files:
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue

            gt_name = name.replace("_in", "_gt")
            gt_path = os.path.join(self.gt_dir, gt_name)

            if not os.path.exists(gt_path):
                skipped += 1
                continue

            if verify_dataset_on_start:
                inp_path = os.path.join(self.inp_dir, name)
                if self._is_valid_image(inp_path) and self._is_valid_image(gt_path):
                    self.files.append(name)
                else:
                    skipped += 1
            else:
                self.files.append(name)

        print(f"Dataset ready: {len(self.files)} valid pairs, {skipped} skipped.", flush=True)

    @staticmethod
    def _is_valid_image(path):
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
            return False

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        for _ in range(10):
            name = self.files[idx]
            try:
                inp = Image.open(os.path.join(self.inp_dir, name)).convert("RGB")
                gt_name = name.replace("_in", "_gt")
                gt = Image.open(os.path.join(self.gt_dir, gt_name)).convert("RGB")
                break
            except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
                idx = random.randint(0, len(self.files) - 1)
        else:
            raise RuntimeError("Failed to read a valid input/gt pair after multiple retries.")

        # Make dimensions divisible by 64
        w, h = inp.size
        new_w = (w // 64) * 64
        new_h = (h // 64) * 64
        if new_w == 0:
            new_w = w
        if new_h == 0:
            new_h = h

        inp = inp.crop((0, 0, new_w, new_h))
        gt = gt.crop((0, 0, new_w, new_h))

        # Use patches to control memory usage.
        if self.patch_size is not None and self.patch_size > 0:
            p = min(self.patch_size, new_w, new_h)
            p = (p // 64) * 64
            if p > 0 and new_w > p and new_h > p:
                x = random.randint(0, new_w - p)
                y = random.randint(0, new_h - p)
                inp = inp.crop((x, y, x + p, y + p))
                gt = gt.crop((x, y, x + p, y + p))

        inp = self.transform(inp)
        gt = self.transform(gt)

        return inp, gt


class PairedEvalDataset(Dataset):
    def __init__(self, data_root, max_images=0):
        self.transform = transforms.ToTensor()
        self.pairs = []
        for name in sorted(os.listdir(data_root)):
            if "_in." not in name:
                continue
            stem, ext = name.split("_in.", 1)
            gt_name = f"{stem}_gt.{ext}"
            in_path = os.path.join(data_root, name)
            gt_path = os.path.join(data_root, gt_name)
            if os.path.exists(gt_path):
                self.pairs.append((in_path, gt_path))
        if max_images > 0:
            self.pairs = self.pairs[:max_images]
        if not self.pairs:
            raise RuntimeError(f"No paired *_in/*_gt files found in {data_root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        in_path, gt_path = self.pairs[idx]
        inp = Image.open(in_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        return self.transform(inp), self.transform(gt)


def batch_psnr(pred, gt):
    pred = torch.clamp(pred, 0, 1)
    gt = torch.clamp(gt, 0, 1)
    mse = torch.mean((pred - gt) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-12)
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse))


def evaluate_psnr(model, loader, device):
    model.eval()
    vals = []
    with torch.no_grad():
        for inp, gt in loader:
            inp = inp.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            out = torch.clamp(model(inp), 0, 1)
            h = min(out.shape[-2], gt.shape[-2])
            w = min(out.shape[-1], gt.shape[-1])
            out = out[..., :h, :w]
            gt = gt[..., :h, :w]
            vals.append(batch_psnr(out, gt))
    model.train()
    return float(torch.cat(vals).mean().item()) if vals else float("nan")


# ---------------- Paths ----------------
args = parse_args()
device = resolve_device(args.device)

dataset = ShadowDataset(
    args.train_inp,
    args.train_gt,
    patch_size=args.patch_size,
    verify_dataset_on_start=args.verify_dataset_on_start,
)
print("Building DataLoader...", flush=True)
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

eval_loader = DataLoader(
    PairedEvalDataset(args.paired_eval_root, max_images=args.paired_eval_images),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)


# ---------------- Model ----------------
print("Building model...", flush=True)
model = final_net().to(device)
if args.use_grad_checkpointing and hasattr(model, "set_gradient_checkpointing"):
    model.set_gradient_checkpointing(True)
    print("Gradient checkpointing enabled.", flush=True)
print(f"Using device: {device}")

print("Loading pretrained weights...", flush=True)
if args.init_full_ckpt and os.path.exists(args.init_full_ckpt):
    model.load_state_dict(torch.load(args.init_full_ckpt, map_location=device), strict=True)
    print(f"Loaded full checkpoint: {args.init_full_ckpt}", flush=True)
else:
    model.remove_model.load_state_dict(
        torch.load("weights/shadowremoval.pkl", map_location=device)
    )
    model.enhancement_model.load_state_dict(
        torch.load("weights/refinement.pkl", map_location=device)
    )
    print("Loaded component checkpoints: shadowremoval.pkl + refinement.pkl", flush=True)
print("Model ready. Starting training loop...", flush=True)

model.train()

for param in model.parameters():
    param.requires_grad = True

feature_loss = Vgg16().to(device)
dssim = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)
# Keep FFT ops in FP32: cuFFT half precision requires power-of-two spatial sizes.
use_amp = False
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs
)


# ---------------- Training ----------------
log_every = 20

best_psnr = -1.0
os.makedirs(args.out_dir, exist_ok=True)

for epoch in range(args.epochs):
    total_loss = 0
    epoch_start = time.time()
    print(f"Epoch {epoch+1}/{args.epochs} started. Batches: {len(loader)}", flush=True)

    for step, (inp, gt) in enumerate(loader, start=1):
        inp = inp.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        output = None
        loss = None

        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(inp)
                loss_fl = feature_loss(output, gt, which="relu2")
                loss_ssim = 1 - dssim(output, gt)
                loss = (0.5 * loss_fl) + (0.5 * loss_ssim)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if step % log_every == 0 or step == 1 or step == len(loader):
                avg_loss = total_loss / step
                msg = f"Epoch {epoch+1}/{args.epochs} Step {step}/{len(loader)} Loss {avg_loss:.6f}"
                if device.type == "cuda":
                    mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    msg += f" GPU_mem {mem:.2f}GB"
                print(msg, flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                print("CUDA OOM on this batch, skipping and clearing cache.")
                optimizer.zero_grad(set_to_none=True)
                del output, loss, inp, gt
                torch.cuda.empty_cache()
                continue
            raise

    epoch_loss = total_loss / max(1, len(loader))
    elapsed = time.time() - epoch_start
    print(f"Epoch {epoch+1} done. Loss: {epoch_loss:.6f}. Time: {elapsed:.1f}s", flush=True)
    
    scheduler.step()

    if (epoch + 1) % args.eval_every == 0:
        val_psnr = evaluate_psnr(model, eval_loader, device)
        print(f"Epoch {epoch+1}: paired-eval PSNR {val_psnr:.4f} dB", flush=True)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = os.path.join(args.out_dir, "best_psnr.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best model: {best_path} ({best_psnr:.4f} dB)", flush=True)

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"epoch_{epoch + 1}.pth"))
        run_validation(model, device, epoch, args.valid_dir)

final_path = os.path.join(args.out_dir, "fine_tuned.pth")
torch.save(model.state_dict(), final_path)
print(f"Training Complete. Final model saved as {final_path}", flush=True)
if best_psnr >= 0:
    print(
        f"Best paired-eval checkpoint PSNR: {best_psnr:.4f} dB ({os.path.join(args.out_dir, 'best_psnr.pth')})",
        flush=True,
    )
