import codecs
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from vision import VisionDataset

class MNIST(VisionDataset):
    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
            "0 - zero",
            "1 - one",
            "2 - two",
            "3 - three",
            "4 - four",
            "5 - five",
            "6 - siz",
            "7 - seven",
            "8 - eight",
            "9 - nine",
    ]

    @property
    def train_labels(self):
        print('1')
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        print('2')
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        print('3')
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        print('4')
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        self.data, self.targets = self._load_data()

    def _load_data(self):
        print('5')
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        print('6')
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        print('7')
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        print('8')
        return os.path.join(self.root, '')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        print('9')
        return {_class: i for i, _class in enumerate(self.classes)}

    def extra_repr(self) -> str:
        print('10')
        return "Split: {}".format("Train" if self.train is True else "Test")

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)

SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype(">i2"), "i2"),
    12: (torch.int32, np.dtype(">i4"), "i4"),
    13: (torch.float32, np.dtype(">f4"), "f4"),
    14: (torch.float64, np.dtype(">f8"), "f8"),
}

def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    print('11')
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2])).view(*s)


def read_label_file(path: str) -> torch.Tensor:
    print('12')
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 1
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    print('13')
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 3
    return x

