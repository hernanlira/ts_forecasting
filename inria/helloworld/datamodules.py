from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from multiprocessing import cpu_count
from typing import Any, Optional, Union

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128, num_workers=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        if not num_workers:
            num_workers = cpu_count()
        self.num_workers = num_workers

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download data, train then test
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            mnist = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=10 * self.batch_size, num_workers=self.num_workers)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=10 * self.batch_size, num_workers=self.num_workers)
        return mnist_test


class Cifar10DataModule(CIFAR10DataModule):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = cpu_count(),
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> None:
        if "train_transforms" not in kwargs or kwargs["train_transforms"] is None:
            kwargs["train_transforms"] = Compose(
                [
                    RandomCrop(32, padding=4),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    cifar10_normalization(),
                ]
            )

        test_transforms = Compose(
            [
                ToTensor(),
                cifar10_normalization(),
            ]
        )

        if "test_transforms" not in kwargs or kwargs["test_transforms"] is None:
            kwargs["test_transforms"] = test_transforms

        if "val_transforms" not in kwargs or kwargs["val_transforms"] is None:
            kwargs["val_transforms"] = test_transforms

        super().__init__(
            data_dir, val_split, num_workers, normalize, batch_size, seed, shuffle, pin_memory, drop_last, *args, **kwargs
        )
