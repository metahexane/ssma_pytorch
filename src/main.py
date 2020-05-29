import torch
from util.loader import DataLoader
from util.helper import *

dl = DataLoader()
print(dl.num_labels)

print("---- Stage 1: individual AdapNet++ training ----")
m1, m2 = train_stage_1(dl, bach_size=8)
torch.save(m1, "models/m1")
torch.save(m1, "models/m2")

print("---- Stage 2: fusion architecture, train SSMA ----")
mf = train_stage_2(dl, [m1, m2], batch_size=7)
torch.save(mf, "models/mf")

print("---- Stage 3: train SSMA and decoder ----")
mf_final = train_stage_3(dl, mf, batch_size=12)
torch.save(m1, "models/mf_final")
