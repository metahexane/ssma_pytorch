import torch
import torch.nn as nn
import torch.optim as optim
from adapnet import AdapNet
import gc
from tqdm import tqdm
from util.eval import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_stage_0(dl, iters=150 * (10 ** 3)):
    # mod1
    model_m1 = AdapNet(dl.num_labels)
    model_m1.cuda()

    # mod2
    model_m2 = AdapNet(dl.num_labels)
    model_m2.cuda()

    lr = 10 ** -3
    adam_opt_m1 = optim.Adam(model_m1.parameters(), lr=lr)
    adam_opt_m2 = optim.Adam(model_m2.parameters(), lr=lr)

    train_iteration(iters, [model_m1, model_m2], [adam_opt_m1, adam_opt_m2], dl)

    return model_m1, model_m2


def train_stage_1(dl, models, iters=100 * (10 ** 3)):
    model_fusion = AdapNet(dl.num_labels, encoders=[models[0].encoder_mod1, models[1].encoder_mod1])
    model_fusion.cuda()

    lr_enc = 10 ** -4
    lr_dec = 10 ** -3

    adam_opt = optim.Adam([
        {"params": model_fusion.encoder_mod1.parameters()},
        {"params": model_fusion.encoder_mod2.parameters()},
        {"params": model_fusion.eASPP.parameters()},
        {"params": model_fusion.ssma_s1.parameters()},
        {"params": model_fusion.ssma_s2.parameters()},
        {"params": model_fusion.ssma_res.parameters()},
        {"params": model_fusion.decoder.parameters(), "lr": lr_dec}], lr=lr_enc)
    train_iteration(iters, [model_fusion], [adam_opt], dl)

    return model_fusion


def train_stage_2(dl, model, iters=50 * (10 ** 3)):
    lr_dec = 10 ** -5

    adam_opt = optim.Adam([
        {"params": model.eASPP.parameters()},
        {"params": model.ssma_s1.parameters()},
        {"params": model.ssma_s2.parameters()},
        {"params": model.ssma_res.parameters()},
        {"params": model.decoder.parameters()},
        {"params": model.encoder_mod1.parameters(), "lr": 0},
        {"params": model.encoder_mod2.parameters(), "lr": 0}
    ], lr=lr_dec)
    train_iteration(iters, [model], [adam_opt], dl)
    return model


def train_iteration(iters, models, opts, dl, batch_size=2):
    epochs = iters // batch_size
    for i in tqdm(range(iters)):
        mod1, mod2, gt_all = dl.sample_batch(batch_size, i + 1)

        input = [[mod1, mod2]]
        if len(models) > 1:
            input = [mod1, mod2]

        for j, model in enumerate(models):
            res, batch_loss = train(model, opts[j], input[j], gt_all, fusion=len(models) == 1)

            del res
            torch.cuda.empty_cache()
            gc.collect()

        if (i + 1) % epochs == 0 or i == iters - 1:
            iou = evaluate(models[0], dl, mode="validation")
            print("Evaluation of validation set @ " + str(i))
            print("mIoU: " + str(iou.mean().item()))
            print("IoU: " + str(iou))

    iou = evaluate(models[0], dl, mode="test")
    print("Evaluation of test set")
    print("mIoU: " + str(iou.mean().item()))
    print("IoU: " + str(iou))

def train(model, opt, input, target, fusion=False):
    model.train()
    opt.zero_grad()

    if fusion:
        a1, a2, res = model(input[0], input[1])
    else:
        a1, a2, res = model(input)

    a1 = torch.softmax(a1, dim=1)
    a2 = torch.softmax(a2, dim=1)
    res = torch.softmax(res, dim=1)

    res_loss = nn.CrossEntropyLoss()
    aux1_loss = nn.CrossEntropyLoss()
    aux2_loss = nn.CrossEntropyLoss()

    batch_loss = res_loss(res, target) + .6 * aux1_loss(a1, target) + .5 * aux2_loss(a2, target)
    batch_loss.backward()
    opt.step()

    del a1
    del a2

    return res, batch_loss
