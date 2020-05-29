import torch
from util.loader import DataLoader
from util.helper import *
import datetime

dl = DataLoader(num_examples=10)
date = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")

print("---- Stage 1: individual AdapNet++ training ----")
m1, m2 = train_stage_1(dl, 2, iters=5) # 8
torch.save(m1.state_dict(), "models/adapnet_mod1_" + date + ".pt")
torch.save(m2.state_dict(), "models/adapnet_mod2_" + date + ".pt")

print("---- Stage 2: fusion architecture, train SSMA ----")
mf = train_stage_2(dl, [m1, m2], 7) # 7
torch.save(mf.state_dict(), "models/adapnet_fusion1_" + date + ".pt")

print("---- Stage 3: train SSMA and decoder ----")
mf_final = train_stage_3(dl, mf, 12) # 12
torch.save(mf_final.state_dict(), "models/adapnet_fusion2_" + date + + ".pt")
