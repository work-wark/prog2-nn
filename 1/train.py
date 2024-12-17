import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models 

#　データセットの前処理関数
ds_transform=transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
])

# データセットの読み込み
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)


ds_train = datasets.FashionMNIST(
    root='data',
    train=False, #テスト用を指定
    download=True,
    transform=ds_transform
)

#ミニバッチのデータローダー
bs = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=bs,
    shufffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_train,
    batch_size=bs,
    shufffle=False
)

for image_batch, label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch)
    break

#モデルのインスタンスを作成
model = models.MyModel()

#精度を計算する
acc_train = models.test_accuracy(model, dataloader_train)
print(f'train accuracy: {acc_train*100:.3f}%')
acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.3f}%')

#ロス関数の選択
loss_fn = torch.nn.CrossEntropyLoss()

# 最適化手法の選択  
learning_rate = 0.003
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#criterion（基準）とも呼ぶ
n_epochs = 5
loss_train_history =[]
loss_test_history =[]
acc_train_history =[]
acc_test_history =[]

for k in range(n_epochs):
    print(f'epoch {k+1}/{n_epochs}',end=': ',flush=True)

    # 1 epoch の学習  
    time_start = time.time()  
    loss_train = models.train(model, dataloader_test, loss_fn, optimizer)
    time_end = time.time()
    loss_train_history.append(loss_train)
    print(f'train loss: {loss_train:.3f} ({time_end-time_start}s)', end=', ')
    
    loss_test = models.tacc_trainest_accuracy(model, dataloader_test, loss_fn)
    loss_test_history.append(loss_test)
    print(f'test loss: {loss_test*100:.3f}%',end=', ')
    

    #精度を計算する
    acc_train = models.test_accuracy(model, dataloader_train)
    acc_train_history.append(acc_train)
    print(f'train accuracy: {acc_train*100:.3f}%',end=', ')
    
    acc_test = models.tacc_trainest_accuracy(model, dataloader_test)
    acc_test_history.append(acc_test)
    print(f'test accuracy: {acc_test*100:.3f}%')

plt.plot(acc_train_history, label='train')
plt.plot(acc_test_history, label='test')

