from util.loader import DataLoader
from util.helper import *

dl = DataLoader()

print("---- Stage 1: individual AdapNet++ training ----")
m1, m2 = train_stage_0(dl, iters=5)

print("---- Stage 2: fusion architecture, train SSMA ----")
mf = train_stage_1(dl, [m1, m2], iters=5)

print("---- Stage 3: train SSMA and decoder ----")
mf_final = train_stage_2(dl, mf, iters=5)
