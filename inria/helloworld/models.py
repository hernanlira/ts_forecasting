import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
import torchmetrics
import torchvision


class HelloWorldClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        # log hyperparameters
        self.save_hyperparameters()

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def loss(self, xs, ys):
        "convenient method to get the loss on a batch"

        logits = self(xs)  # this calls self.forward
        loss = F.nll_loss(logits, ys)
        return logits, loss

    def forward(self, x):
        out = self(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        xs, target = batch
        logits, loss = self.loss(xs, target)
        preds = torch.argmax(logits, 1)

        return {"loss": loss, "preds": preds, "target": target}

    def training_step_end(self, outputs):
        self.train_acc(outputs["preds"], outputs["target"])

        self.log("train/loss", outputs["loss"])
        self.log("train/accuracy", self.train_acc)
        self.log("train/learning_rate", self.hparams["lr"])

    def evaluate(self, batch, stage=None):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.test_acc(preds, ys)

        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/accuracy_epoch", self.test_acc, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        xs, target = batch
        logits, loss = self.loss(xs, target)
        preds = torch.argmax(logits, 1)
        return {"loss": loss, "preds": preds, "target": target}

    def validation_step_end(self, outputs):
        self.valid_acc(outputs["preds"], outputs["target"])

        self.log("valid/loss_epoch", outputs["loss"], on_step=False, on_epoch=True)  # default on val/test is on_epoch only
        self.log("valid/accuracy_epoch", self.valid_acc, on_step=False, on_epoch=True)


class HelloWorldMlp(HelloWorldClassifier):
    def __init__(self, in_dims, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-4):
        super().__init__()

        # we flatten the input Tensors and pass them through an MLP
        self.layer_1 = nn.Linear(np.prod(in_dims), n_layer_1)
        self.layer_2 = nn.Linear(n_layer_1, n_layer_2)
        self.layer_3 = nn.Linear(n_layer_2, n_classes)

    def forward(self, x):
        """
        Defines a forward pass using the Stem-Learner-Task
        design pattern from Deep Learning Design Patterns:
        https://www.manning.com/books/deep-learning-design-patterns
        """
        batch_size, *dims = x.size()

        # stem: flatten
        x = x.view(batch_size, -1)

        # learner: two fully-connected layers
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))

        # task: compute class logits
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])


class HelloWorldResnet(HelloWorldClassifier):
    def _create_model(self):
        model = torchvision.models.resnet18(pretrained=False, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        return model

    def __init__(self, lr=0.05):
        super().__init__()
        self.model = self._create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=5e-4,
        )
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(self.trainer.datamodule.dataset_train) // self.trainer.datamodule.batch_size,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
