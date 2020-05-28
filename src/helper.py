import torch
import torch.nn as nn
import torch.optim as optim
from ssma import SSMA
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_iou(target: torch.Tensor, pred: torch.Tensor, eps=10**-5):
    # ensure target dim = pred dim
    assert target.shape == pred.shape

    intersection = (target & pred).float().sum((1, 2))
    union = (target | pred).float().sum((1, 2))

    return (intersection + eps) / (union + eps)

def train_stage_0(dl, iters=150 * (10**3)):
    # mod1
    model_m1 = SSMA(22)
    model_m1.cuda()

    # mod2
    model_m2 = SSMA(22)
    model_m2.cuda()

    lr = 10 ** -3
    adam_opt_m1 = optim.Adam(model_m1.parameters(), lr=lr)
    adam_opt_m2 = optim.Adam(model_m2.parameters(), lr=lr)

    train_iteration(iters, [model_m1, model_m2], [adam_opt_m1, adam_opt_m2], dl)

    return model_m1, model_m2

def train_stage_1(dl, models, iters=100 * (10**3)):
    model_fusion = SSMA(22)
    model_fusion.cuda()

    lr_enc = 10**-4
    lr_dec = 10**-3

    adam_opt = optim.Adam([{ "params": model_fusion.encoder_mod1, "lr": lr_enc }], lr=lr_enc)
    train_iteration(iters, [model_fusion], [adam_opt], dl)

    return model_fusion

def train_stage_2(dl, model, iters=50 * (10**3)):
    lr_dec = 10**-5

    adam_opt = optim.Adam([{ }], lr=0)
    train_iteration(iters, [model], [adam_opt], dl)
    return model


def train_iteration(iters, models, opts, dl):
    for i in range(iters):
        for j, model in enumerate(models):
            res, batch_loss = train(model, opts[j], dl, i, stage=0, batch_size=2)

            del res
            torch.cuda.empty_cache()
            gc.collect()

def train(model, opt, dl, i, batch_size=2):
    model.train()
    opt.zero_grad()

    mod1, mod2, gt_all = dl.sample_batch(batch_size, i + 1)

    a1, a2, res = model(mod1, mod2)

    a1 = torch.softmax(a1, dim=1)
    a2 = torch.softmax(a2, dim=1)
    res = torch.softmax(res, dim=1)

    res_loss = nn.CrossEntropyLoss()
    aux1_loss = nn.CrossEntropyLoss()
    aux2_loss = nn.CrossEntropyLoss()

    batch_loss = res_loss(res, gt_all) + .6 * aux1_loss(a1, gt_all) + .5 * aux2_loss(a2, gt_all)
    batch_loss.backward()
    opt.step()

    del a1
    del a2

    return res, batch_loss