import logging

from pathlib import Path

from inria.helloworld.datamodules import MnistDataModule

logger = logging.getLogger(__name__)


def test_data_mnist__prepare_data(data_dir: Path):
    mnist = MnistDataModule(data_dir)
    mnist.prepare_data()

    assert list(data_dir.iterdir()), "Data dir contains something (i.e. the datasets)"


def test_data_mnist__setup_fit(data_dir: Path):
    mnist = MnistDataModule(data_dir)
    mnist.prepare_data()
    mnist.setup(stage="fit")

    assert mnist.mnist_train is not None and mnist.mnist_val is not None, "Train data should be ready"


def test_data_mnist__setup_test(data_dir: Path):
    mnist = MnistDataModule(data_dir)
    mnist.prepare_data()
    mnist.setup(stage="test")

    assert mnist.mnist_test is not None, "Test data should be ready"
