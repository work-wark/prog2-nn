from torch import nn
import torch


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    

def test_accuracy(model, dataloader):
    n_corrects = 0 #正解の個数

    model.eval()
    for image_batch, label_batch in dataloader:
        with torch.no_grad():
            logits_batch = model(image_batch)

        predict_batch = logits_batch.argmax(dim=1)
        n_corrects += (label_batch == predict_batch).sum().item()

    accuracy = n_corrects / len(dataloader.dataset)
    return accuracy


# acc_test = models.test_accuracy(model, dataloader_test)
# print(f'test accuracy: {acc_test*100:.2f}%')

def train(model, dataloader, loss_fn, optimizer):
    model.train()
    for image_batch, label_batch in dataloader:
        logits_batch = model(image_batch)

        loss = loss_fn(logits_batch, label_batch)

        # 最適化
        optimizer.zero_grad()
        loss.backward() #   誤差逆伝播法(back propagation)
        optimizer.step() #パラメータをちょっと動かす

    return loss.item()

def test(model,dataloader, loss_fn):
    loss_total = 0.0

    model.eval()
    for image_batch, label_batch in dataloader:
        with torch.no_grad():
            logits_batch = model(image_batch)

        loss = loss_fn(logits_batch,  label_batch)
        loss_total += loss.item()
        
    # バッチ数を割って、平均値を返す
    return loss_total / len(dataloader)



