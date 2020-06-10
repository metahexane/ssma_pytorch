import torch
from util.loader import DataLoader
from util.parser import *
from util.helper import *
import datetime
from components.encoder import Encoder
import gc

torch.cuda.empty_cache()

args = parse_args()

batch_sizes = eval(args.batch)
iterations = eval(args.iters)
enc_lr = eval(args.enc_lr)
dec_lr = eval(args.dec_lr)
start_stage = int(args.start)
load_model = args.model
resume_training = int(args.resume) == 1

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


def stage_2(models, date, dl, optim=None):
    print("---- Stage 2: fusion architecture, train SSMA ----")
    s2_names = ["models/adapnet_fusion1_" + date + ".tar"]
    mf = train_stage_2(dl, models, batch_sizes[1], s2_names, iters=iterations[1], enc_lr=enc_lr[1], dec_lr=dec_lr[1], optim=optim)

    return mf


def stage_3(mf, date, dl, optim=None):
    print("---- Stage 3: train SSMA and decoder ----")
    s3_names = ["models/adapnet_fusion2_" + date + ".tar"]
    mf_final = train_stage_3(dl, mf, batch_sizes[2], s3_names, iters=iterations[2], enc_lr=enc_lr[2], dec_lr=dec_lr[2], optim=optim)

    return mf_final


if start_stage == 1:
    m1, m2, date, dl = stage_1()
    gc.collect()
    mf = stage_2([m1, m2], date, dl)
    gc.collect()
    mf_final = stage_3(mf, date, dl)

if start_stage == 2:
    dl = DataLoader(args.data, date=load_model)
    if resume_training:
        mf = AdapNet(dl.num_labels, encoders=[Encoder(), Encoder()])
        mf_name = "models/adapnet_fusion1_" + load_model + ".tar"
        l_mfusion = torch.load(mf_name)
        mf.load_state_dict(l_mfusion['model_state_dict'])
        mf.cuda()

        print("Resuming training in stage 2 from iteration " + str(l_mfusion['iteration']))

        adam_opt = optim.Adam([
            {"params": mf.encoder_mod1.parameters()},
            {"params": mf.encoder_mod2.parameters()},
            {"params": mf.eASPP.parameters()},
            {"params": mf.ssma_s1.parameters()},
            {"params": mf.ssma_s2.parameters()},
            {"params": mf.ssma_res.parameters()},
            {"params": mf.decoder.parameters(), "lr": dec_lr[1]}], lr=enc_lr[1])

        adam_opt.load_state_dict(l_mfusion['optimizer_state_dict'])

        models = [mf]
        optim = adam_opt

        del l_mfusion
        torch.cuda.empty_cache()
        gc.collect()
    else:
        m1 = AdapNet(dl.num_labels)
        m2 = AdapNet(dl.num_labels)
        m1_name = "models/adapnet_mod1_" + load_model + ".tar"
        m2_name = "models/adapnet_mod2_" + load_model + ".tar"
        m1_load = torch.load(m1_name)
        m2_load = torch.load(m2_name)

        m1.load_state_dict(m1_load['model_state_dict'])
        m2.load_state_dict(m2_load['model_state_dict'])

        m1.cuda()
        m2.cuda()

        model_fusion = AdapNet(dl.num_labels, encoders=[m1.encoder_mod1, m2.encoder_mod1])
        model_fusion.cuda()

        del m1
        del m2
        del m1_load
        del m2_load
        torch.cuda.empty_cache()
        gc.collect()

        models = [model_fusion]
        optim = None

    mf = stage_2(models, load_model, dl, optim=optim)

    if adam_opt is not None:
        del adam_opt
    del optim
    torch.cuda.empty_cache()
    gc.collect()

    mf_final = stage_3(mf, load_model, dl)

if start_stage == 3:
    dl = DataLoader(args.data, date=load_model)

    if resume_training:
        mf = AdapNet(dl.num_labels, encoders=[Encoder(), Encoder()])
        mf_name = "models/adapnet_fusion2_" + load_model + ".tar"
        lmfinal = torch.load(mf_name)
        mf.load_state_dict(lmfinal['model_state_dict'])
        mf.cuda()

        print("Resuming training in stage 3 from iteration " + str(lmfinal["iteration"]))

        adam_opt = optim.Adam([
            {"params": mf.eASPP.parameters()},
            {"params": mf.ssma_s1.parameters()},
            {"params": mf.ssma_s2.parameters()},
            {"params": mf.ssma_res.parameters()},
            {"params": mf.decoder.parameters()},
            {"params": mf.encoder_mod1.parameters(), "lr": enc_lr[2]},
            {"params": mf.encoder_mod2.parameters(), "lr": enc_lr[2]}
        ], lr=dec_lr[2])
        adam_opt.load_state_dict(lmfinal['optimizer_state_dict'])
        optim = adam_opt

        del lmfinal
        torch.cuda.empty_cache()
        gc.collect()
    else:
        mf = AdapNet(dl.num_labels, encoders=[Encoder(), Encoder()])
        mf_name = "models/adapnet_fusion1_" + load_model + ".tar"
        mf.load_state_dict(torch.load(mf_name)['model_state_dict'])
        mf.cuda()
        optim = None

    mf_final = stage_3(mf, load_model, dl, optim=optim)
