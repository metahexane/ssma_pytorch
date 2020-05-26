from ssma import SSMA
from loader import DataLoader
import torch
import gc
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dl = DataLoader()

model = SSMA(22)
model.cuda()

lr = 10**-3
adam_opt = optim.Adam(model.parameters(), lr=lr)

iters = 100
all_iters = []
all_losses = []

for i in range(iters):
    model.train()
    adam_opt.zero_grad()

    mod1, mod2, gt_all = dl.sample_batch(2, i+1)

    a1, a2, res = model(mod1, mod2)

    # loss
    res_loss = nn.CrossEntropyLoss()
    aux1_loss = nn.CrossEntropyLoss()
    aux2_loss = nn.CrossEntropyLoss()

    batch_loss = res_loss(res, gt_all) + .6 * aux1_loss(a1, gt_all) + .5 * aux2_loss(a2, gt_all)
    batch_loss.backward()
    adam_opt.step()

    print("Train iteration " + str(i) + " with loss = " + str(batch_loss.item()))

    all_iters.append(i + 1)
    all_losses.append(batch_loss.item())

    if (i+1) % 5 == 0:
        img = torch.argmax(res[0], dim=0, keepdim=True)
        dl.result_to_image(img[0], i)

    del a1
    del a2
    del res

    torch.cuda.empty_cache()
    gc.collect()

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.plot(all_iters, all_losses)
plt.show()