import torch
from util.loader import DataLoader
from util.parser import *
from util.helper import *
import datetime

args = parse_args()

batch_sizes = eval(args.batch)
iterations = eval(args.iters)
enc_lr = eval(args.enc_lr)
dec_lr = eval(args.dec_lr)

dl = DataLoader(args.data)
date = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
dl.write_dataset(date)

print("---- Stage 1: individual AdapNet++ training ----")
s1_names = ["models/adapnet_mod1_" + date + ".pt", "models/adapnet_mod2_" + date + ".pt"]
m1, m2 = train_stage_1(dl, batch_sizes[0], s1_names, iters=iterations[0], enc_lr=enc_lr[0], dec_lr=dec_lr[0])

if int(args.save) == 1:
    torch.save(m1.state_dict(), s1_names[0])
    torch.save(m2.state_dict(), s1_names[1])

print("---- Stage 2: fusion architecture, train SSMA ----")
s2_names = ["models/adapnet_fusion1_" + date + ".pt"]
mf = train_stage_2(dl, [m1, m2], batch_sizes[1], s2_names, iters=iterations[1], enc_lr=enc_lr[1], dec_lr=dec_lr[1])

if int(args.save) == 1:
    torch.save(mf.state_dict(), s2_names[0])

print("---- Stage 3: train SSMA and decoder ----")
s3_names = ["models/adapnet_fusion2_" + date + ".pt"]
mf_final = train_stage_3(dl, mf, batch_sizes[2], s3_names, iters=iterations[2], enc_lr=enc_lr[2], dec_lr=dec_lr[2])

if int(args.save) == 1:
    torch.save(mf_final.state_dict(), s3_names[0])
