import torch.nn as nn
import torch.optim as optim


def train(loader, model, args):
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        for i, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            if (epoch + 1) % 1000 == 0:
                print(epoch + 1, i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()