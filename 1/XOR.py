import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


class mydata(Dataset):
    def __init__(self, file):
        data = np.loadtxt(file)
        self.xdata = torch.from_numpy(data[:, 0:-1])
        self.ydata = torch.from_numpy(data[:, [-1]])

    def __getitem__(self, item):
        return self.xdata[item], self.ydata[item]

    def __len__(self):
        return len(self.xdata)


class myModel(torch.nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = myModel().to(device)
criterion = torch.nn.MSELoss()


def trainNet():
    loader = DataLoader(mydata("data/1.txt"), batch_size=4, shuffle= True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0)
    for epoch in range(5000):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)
            y = model(inputs)
            loss = criterion(y, labels)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(epoch, inputs, labels, y)


def validation():
    set = mydata('data/2.txt')
    loader = DataLoader(set, batch_size=1, shuffle=False)
    model.eval()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(torch.float32), y.to(torch.float32)
        with torch.no_grad():
            pre = model(x)
            loss = criterion(pre, y)
        total_loss += loss.cpu().item() * len(x)
    avg_loss = total_loss / len(loader.dataset)
    print(avg_loss)


trainNet()
validation()
