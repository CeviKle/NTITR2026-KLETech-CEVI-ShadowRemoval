from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os


class dehaze_test_dataset(Dataset):
    def __init__(self, test_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        lq_dir = os.path.join(test_dir, "LQ")
        # Accept either the original benchmark layout (test_dir/LQ)
        # or a flat directory of images such as a raw test_inp folder.
        self.root_hazy = lq_dir if os.path.isdir(lq_dir) else test_dir
        self.list_test_hazy = sorted(
            [
                name
                for name in os.listdir(self.root_hazy)
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            ]
        )
        self.file_len = len(self.list_test_hazy)

    def __getitem__(self, index, is_train=True):
        hazy = Image.open(os.path.join(self.root_hazy, self.list_test_hazy[index])).convert("RGB")
        hazy = self.transform(hazy)
        name = self.list_test_hazy[index]
        return hazy, name

    def __len__(self):
        return self.file_len
