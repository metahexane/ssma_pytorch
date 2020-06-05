import torch
from util.loader import DataLoader
from util.parser import *
from util.helper import *
import datetime
from components.encoder import Encoder

args = parse_args()

batch_sizes = eval(args.batch)
iterations = eval(args.iters)
enc_lr = eval(args.enc_lr)
dec_lr = eval(args.dec_lr)
start_stage = int(args.start)
load_model = args.model


def stage_1():
    if len(load_model) == 0:
        dl = DataLoader(args.data)
        date = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
        dl.write_dataset(date)
    else:
        dl = DataLoader(args.data, date=load_model)
        date = load_model

    print("---- Stage 1: individual AdapNet++ training ----")
    s1_names = ["models/adapnet_mod1_" + date + ".tar", "models/adapnet_mod2_" + date + ".tar"]
    m1, m2 = train_stage_1(dl, batch_sizes[0], s1_names, iters=iterations[0], enc_lr=enc_lr[0], dec_lr=dec_lr[0], load_model=load_model)

    return m1, m2, date, dl


def stage_2(m1, m2, date, dl):
    print("---- Stage 2: fusion architecture, train SSMA ----")
    s2_names = ["models/adapnet_fusion1_" + date + ".tar"]
    mf = train_stage_2(dl, [m1, m2], batch_sizes[1], s2_names, iters=iterations[1], enc_lr=enc_lr[1], dec_lr=dec_lr[1])

    return mf


def stage_3(mf, date, dl):
    print("---- Stage 3: train SSMA and decoder ----")
    s3_names = ["models/adapnet_fusion2_" + date + ".tar"]
    mf_final = train_stage_3(dl, mf, batch_sizes[2], s3_names, iters=iterations[2], enc_lr=enc_lr[2], dec_lr=dec_lr[2])

    return mf_final


if start_stage == 1:
    m1, m2, date, dl = stage_1()
    mf = stage_2(m1, m2, date, dl)
    mf_final = stage_3(mf, date, dl)

if start_stage == 2:
    dl = DataLoader(args.data, date=load_model)
    m1 = AdapNet(dl.num_labels)
    m2 = AdapNet(dl.num_labels)
    m1_name = "models/adapnet_mod1_" + load_model + ".pt"
    m2_name = "models/adapnet_mod2_" + load_model + ".pt"

    m1.load_state_dict(torch.load(m1_name))
    m2.load_state_dict(torch.load(m2_name))

    m1.cuda()
    m2.cuda()

    mf = stage_2(m1, m2, load_model, dl)
    mf_final = stage_3(mf, load_model, dl)

if start_stage == 3:
    dl = DataLoader(args.data, date=load_model)
    mf = AdapNet(dl.num_labels, encoders=[Encoder(), Encoder()])
    mf_name = "models/adapnet_fusion1_" + load_model + ".pt"
    mf.load_state_dict(torch.load(mf_name))
    mf.cuda()

    mf_final = stage_3(mf, load_model, dl)
