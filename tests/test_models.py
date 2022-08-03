import logging

from inria.helloworld.models import HelloWorldMlp, HelloWorldResnet


logger = logging.getLogger(__name__)


def test_data_mlp__create():
    HelloWorldMlp(in_dims=(1, 28, 28))


def test_data_resnet__create():
    HelloWorldResnet()
