import torch
import torch.nn as nn
import torch.optim as optim
from adapnet import AdapNet
import gc
from tqdm import tqdm
from util.eval import *
from util.parser import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_args()
should_evaluate = int(args.eval) == 1
save_checkpoint = int(args.save_checkpoint)

def train_stage_1(dl, batch_size, names, iters=150 * (10 ** 3), enc_lr=10**-3, dec_lr=10**-3):
    """Train stage 1 of AdapNet++

    The first stage of AdapNet++ trains two AdapNet models individually from each other, each with its own input. One
    takes the RGB-image as its input (modality 1), the other takes the Depth-image as its input (modality 2).

    :param dl: data loader
    :param batch_size: batch size
    :param iters: number of iterations
    :return: two trained AdapNet models
    """
    # mod1
    model_m1 = AdapNet(dl.num_labels)
    model_m1.cuda()

    # mod2
    model_m2 = AdapNet(dl.num_labels)
    model_m2.cuda()

    adam_opt_m1 = optim.Adam(model_m1.parameters(), lr=enc_lr)
    adam_opt_m2 = optim.Adam(model_m2.parameters(), lr=enc_lr)

    train_iteration(iters, [model_m1, model_m2], [adam_opt_m1, adam_opt_m2], dl, names, batch_size)

    return model_m1, model_m2


def train_stage_2(dl, models, batch_size, names, iters=100 * (10 ** 3), enc_lr=10**-4, dec_lr=10**-3):
    """Train stage 2 of AdapNet++

    The second stage of AdapNet++ trains a modified AdapNet model that has two encoders, each with their own modality
    as input. It uses the pre-trained weights of the encoders from stage 1 and adds Self-Supervised
    Model Adaptation (SSMA) to fuse the two encoders together

    :param dl: data loader
    :param models: pre-trained models from stage 1
    :param iters: number of iterations
    :return: fused model
    """
    model_fusion = AdapNet(dl.num_labels, encoders=[models[0].encoder_mod1, models[1].encoder_mod1])
    model_fusion.cuda()

    adam_opt = optim.Adam([
        {"params": model_fusion.encoder_mod1.parameters()},
        {"params": model_fusion.encoder_mod2.parameters()},
        {"params": model_fusion.eASPP.parameters()},
        {"params": model_fusion.ssma_s1.parameters()},
        {"params": model_fusion.ssma_s2.parameters()},
        {"params": model_fusion.ssma_res.parameters()},
        {"params": model_fusion.decoder.parameters(), "lr": dec_lr}], lr=enc_lr)
    train_iteration(iters, [model_fusion], [adam_opt], dl, names, batch_size)

    return model_fusion


def train_stage_3(dl, model, batch_size, names, iters=50 * (10 ** 3), enc_lr=0, dec_lr=10**-5):
    """Train stage 2 of AdapNet++

    The third and last stage of AdapNet++ trains the fused model from stage 2 again, but does not update the weights
    of the two encoders.

    :param dl: data loader
    :param model: fused model1
    :param iters: number of iterations
    :return: final model
    """

    adam_opt = optim.Adam([
        {"params": model.eASPP.parameters()},
        {"params": model.ssma_s1.parameters()},
        {"params": model.ssma_s2.parameters()},
        {"params": model.ssma_res.parameters()},
        {"params": model.decoder.parameters()},
        {"params": model.encoder_mod1.parameters(), "lr": enc_lr},
        {"params": model.encoder_mod2.parameters(), "lr": enc_lr}
    ], lr=dec_lr)
    train_iteration(iters, [model], [adam_opt], dl, names, batch_size)
    return model


def train_iteration(iters, models, opts, dl, names, batch_size=2):
    """Execute the training iterations

    :param iters: number of iterations
    :param models: array of models
    :param opts: array of optimizers
    :param dl: data loader
    :param batch_size: batch size
    :return: updates the models
    """
    epochs = dl.train_size // batch_size
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

        if (i + 1) % save_checkpoint == 0:
            print("Saving at iteration " + str(i+1))
            for u, model_name in enumerate(names):
                torch.save(models[u].state_dict(), model_name)

        if (i + 1) % epochs == 0 and should_evaluate:
            evaluate(models[0], dl, mode="validation")

    evaluate(models[0], dl, mode="validation")
    evaluate(models[0], dl, mode="test")

def train(model, opt, input, target, fusion=False):
    """Execute one training iteration

    :param model: a model
    :param opt: an optimizer
    :param input: a single input modality (at stage 1), or an array of input modalities (at stages 2 and 3)
    :param target: ground truth labels
    :param fusion: boolean if fusion applies (True for stages 2 and 3)
    :return: res: final result
             batch_loss: batch loss
    """
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
