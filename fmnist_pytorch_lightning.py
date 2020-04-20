import logging
from itertools import islice

import torch
from torchvision import datasets, transforms
import random
import numpy as np
import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F

seed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class LitFMNIST(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        self.trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,
                                              train=True, transform=transform)
        self.testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,
                                             train=False,
                                             transform=transform)

    def train_dataloader(self):
        # tdl = torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=False,
        #                                   worker_init_fn=random.seed(seed))
        # return islice(iter(tdl), 1)
        return torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=False,
                                        worker_init_fn=random.seed(seed))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return {'loss': loss}

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=False,
                                           worker_init_fn=random.seed(seed))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        val_acc = pred.eq(y.view_as(pred)).sum().to(dtype=torch.float)
        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['val_acc'] for x in outputs]) / len(self.val_dataloader().dataset)
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).sum() / len(self.val_dataloader().dataset)

        # avg_loss = outputs['val_loss'].mean()
        tensorboard_logs = {'val_loss': avg_loss}
        # tensorboard_logs = {'val_loss': avg_acc}
        logger = logging.getLogger(__name__)
        logger.info(f"avg_acc from logger {avg_acc}")
        print(avg_loss, avg_acc)
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.003)


model = LitFMNIST()

# trainer = pl.Trainer(max_steps=1)
trainer = pl.Trainer(nb_sanity_val_steps=0, max_epochs=3)
trainer.fit(model)