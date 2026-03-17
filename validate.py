import os
import re
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
from test_dataset import dehaze_test_dataset
from model import final_net

# ---------------- Params ----------------
ckpt_path = "weights/fine_tuned.pth"  # change to your epoch10 ckpt
test_dir = "/NTIRE2026/C4_ShdwRem/valid"
output_dir = "val_epoch10_results"
batch_size = 1

os.makedirs(output_dir, exist_ok=True)

# ---------------- Load Data ----------------
dataset = dehaze_test_dataset(test_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ---------------- Model ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = final_net().to(device)

print(f"Loading checkpoint: {ckpt_path}")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

with torch.no_grad():
    for batch_idx, (input_tensor, name) in enumerate(loader):
        input_tensor = input_tensor.to(device)

        output = model(input_tensor)

        # get numeric id
        ids = re.findall(r'\d+', str(name))
        if not ids:
            continue
        out_fname = f"{ids[0]}.png"

        imwrite(output, os.path.join(output_dir, out_fname), value_range=(0, 1))

print("Validation inference finished!")
