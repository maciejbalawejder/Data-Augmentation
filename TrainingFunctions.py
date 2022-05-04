import torch
import numpy as np

class Network:
    """
    Network class that takes pytorch model, learning rate and device as input.

    Parameters
    ----------
    loss -> loss function
    opt -> optimizer
    lr_scheduler -> learning rate scheduler which reduces lr when validation plateu

    Functions
    ---------

    - train_step -> takes train_loader and train model for one epoch, and update train accuracy and loss
    - validation_step -> takes val_loader and update validation accuracy and loss
    - test_step -> takes test_loader and return averaged accuracy
    - save_all -> saves traning, validation accuracy and loss to an '.npy' file

    """

    def __init__(
        self,
        model : torch.nn.Module,
        learning_rate : float,
        device : str,
        ):

        self.model = model
        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.device = device

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt,
                mode='min',
                patience=3,
                factor=0.5,
                min_lr=1e-6,
                verbose=True
        )

    def train_step(self, dataset):
        self.model.train()
        batch_loss = []
        batch_acc = []
        for batch in dataset:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            self.opt.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss(outputs, targets)
            loss.backward()
            self.opt.step()
            batch_loss.append(loss.item())
            batch_acc.append(self.batch_accuracy(outputs,targets))

        self.train_loss.append(np.mean(batch_loss))
        self.train_acc.append(np.mean(batch_acc))


    def validation_step(self, dataset):
        self.model.eval()
        batch_loss = []
        batch_acc = []
        with torch.no_grad():
            for batch in dataset:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outputs = self.model(inputs)

                loss = self.loss(outputs, targets)
                batch_loss.append(loss.item())
                batch_acc.append(self.batch_accuracy(outputs,targets))

        self.val_loss.append(np.mean(batch_loss))
        self.val_acc.append(np.mean(batch_acc))
        self.lr_scheduler.step(self.val_loss[-1])


    def test_step(self, dataset):
        self.model.eval()
        batch_acc = []
        with torch.no_grad():
            for batch in dataset:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outputs = self.model(inputs)
                batch_acc.append(self.batch_accuracy(outputs,targets))

        print("Accuracy : ", np.mean(batch_acc), "%")


    def batch_accuracy(self, output, target):
        # output shape: [batch, target]
        output = nn.functional.softmax(output, dim=1)
        output = output.argmax(1)
        acc = torch.sum(output==target) / output.shape[0]
        return acc.cpu()*100

    def save_all(self, name):
        np.save("val_acc{}.npy".format(name), self.val_acc)
        np.save("train_acc{}.npy".format(name), self.train_acc)
        np.save("val_loss{}.npy".format(name), self.val_loss)
        np.save("train_loss{}.npy".format(name), self.train_loss)
