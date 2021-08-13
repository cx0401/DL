import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class myData(Dataset):
    def __init__(self, file):
        data = np.loadtxt(file)
        self.x = data[:, 0:-1]
        self.y = data[:, [-1]]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = myModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.2, momentum=0)


def train():
    dataLoader = DataLoader(myData("data/1.txt"), batch_size=4, shuffle=True)
    model.train()
    for epoch in range(5000):
        for x, y in dataLoader:
            x, y = x.to(torch.float32), y.to(torch.float32)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


def validation():
    dataLoader = DataLoader(myData("data/2.txt"), batch_size=1, shuffle=False)
    model.eval()
    loss = 0
    for x, y in dataLoader:
        with torch.no_grad():
            x, y = x.to(torch.float32), y.to(torch.float32)
            loss += criterion(model(x), y).cpu().item() * len(x) # .cpu() means
            print(x, y, model(x))
    print(loss / len(dataLoader))


train()
validation()
