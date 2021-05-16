import glob
from typing import Tuple, Any

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MaskedCelebADataset(Dataset):

    """
    MaskedCelebADataset contains all
    """

    def __init__(self, root_dir, mask_size=64, transform=None, mode="train"):
        """
        Construct MaskedCelebA dataset
        mode = (train|test)
        """
        super().__init__()

        self._root_dir = root_dir
        self._mask_size = mask_size
        self._mode = mode
        self._transforms = transforms.Compose(transform) if transform is not None else None
        self._images = sorted(glob.glob("%s/images/*.jpg" % root_dir))
        self._images = self._images[:-4000] if mode == "train" else self._images[-4000:]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index):
        img = Image.open(self._images[index]).convert('RGB')
        if self._transforms is not None:
            img = self._transforms(img)

        img = np.asarray(img)
        # work with numpy array from now on

        masked_img, masked_part = self._mask_image(img)

        return (img, masked_img, masked_part)

    def _mask_image(self, img):
        """
        Apply mask to the image
        """
        width = img.shape[1]
        y1, x1 = np.random.randint(0, width - self._mask_size, 2)
        y2, x2 = y1 + self._mask_size, x1 + self._mask_size

        masked_part = img[:, y1:y2, x1:x2]
        masked_img = np.array(img) # copy image array
        masked_img[:, y1:y2, x1:x2] = 255

        return masked_img, masked_part


# test
if __name__ == "__main__":
    ds = MaskedCelebADataset("dataset/celeba", transform=[transforms.Resize((128,128)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    print(f"{len(ds)} images loaded ") # should be 4000 images less than total files in there

    img, masked_img, masked_part = ds[0]
    print(f"img size: {img.shape}")
    print(f"masked_img size: {masked_img.shape}")
    print(f"masked_part size: {masked_part.shape}")
    #  Image.fromarray(np.moveaxis(img,0,2)).show()
    #  Image.fromarray(masked_img).show()
    #  Image.fromarray(masked_part).show()
