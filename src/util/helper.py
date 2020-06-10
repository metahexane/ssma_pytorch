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


def train_stage_1(dl, batch_size, names, iters=150 * (10 ** 3), enc_lr=10 ** -3, dec_lr=10 ** -3, load_model=""):
    """Train stage 1 of AdapNet++

    The first stage of AdapNet++ trains two AdapNet models individually from each other, each with its own input. One
    takes the RGB-image as its input (modality 1), the other takes the Depth-image as its input (modality 2).

    :param dl: data loader
    :param batch_size: batch size
    :param iters: number of iterations
    :return: two trained AdapNet models
    """

    model_m1 = AdapNet(dl.num_labels)
    model_m2 = AdapNet(dl.num_labels)

    if len(load_model) == 0:
        model_m1.cuda()
        model_m2.cuda()

    adam_opt_m1 = optim.Adam(model_m1.parameters(), lr=enc_lr)
    adam_opt_m2 = optim.Adam(model_m2.parameters(), lr=enc_lr)

    if len(load_model) > 0:
        m1_name = "models/adapnet_mod1_" + load_model + ".tar"
        m2_name = "models/adapnet_mod2_" + load_model + ".tar"
        l_m1 = torch.load(m1_name)
        l_m2 = torch.load(m2_name)

        print("Resuming training from iteration " + str(l_m1['iteration']))

        # Uncomment for lower dim loading from larger dim model
        # singles = ["decoder.stage3.6.bias",
        #            "decoder.stage3.7.weight",
        #            "decoder.stage3.7.bias",
        #            "decoder.stage3.7.running_mean",
        #            "decoder.stage3.7.running_var",
        #            "decoder.stage3.8.bias",
        #            "decoder.stage3.9.weight",
        #            "decoder.stage3.9.bias",
        #            "decoder.stage3.9.running_mean",
        #            "decoder.stage3.9.running_var",
        #            "decoder.aux_conv1.bias",
        #            "decoder.aux_conv1_bn.weight",
        #            "decoder.aux_conv1_bn.bias",
        #            "decoder.aux_conv1_bn.running_mean",
        #            "decoder.aux_conv1_bn.running_var",
        #            "decoder.aux_conv2.bias",
        #            "decoder.aux_conv2_bn.weight",
        #            "decoder.aux_conv2_bn.bias",
        #            "decoder.aux_conv2_bn.running_mean",
        #            "decoder.aux_conv2_bn.running_var"]
        #
        # mults = ["decoder.stage3.6.weight",
        #          "decoder.aux_conv1.weight",
        #          "decoder.aux_conv2.weight"]
        #
        # for s in singles:
        #     l_m1['model_state_dict'][s] = l_m1['model_state_dict'][s][:dl.num_labels]
        #
        # for s in mults:
        #     l_m1['model_state_dict'][s] = l_m1['model_state_dict'][s][:dl.num_labels, :]
        #
        # l_m1['model_state_dict']["decoder.stage3.8.weight"] = l_m1['model_state_dict']["decoder.stage3.8.weight"][:dl.num_labels, :dl.num_labels, :]
        #
        # for s in singles:
        #     l_m2['model_state_dict'][s] = l_m2['model_state_dict'][s][:dl.num_labels]
        #
        # for s in mults:
        #     l_m2['model_state_dict'][s] = l_m2['model_state_dict'][s][:dl.num_labels, :]
        #
        # l_m2['model_state_dict']["decoder.stage3.8.weight"] = l_m2['model_state_dict']["decoder.stage3.8.weight"][:dl.num_labels, :dl.num_labels, :]

        model_m1.load_state_dict(l_m1['model_state_dict'])
        model_m2.load_state_dict(l_m2['model_state_dict'])

        model_m1.cuda()
        model_m2.cuda()

        # adam_opt_m1.load_state_dict(l_m1['optimizer_state_dict'])
        # adam_opt_m2.load_state_dict(l_m2['optimizer_state_dict'])

    train_iteration(iters, [model_m1, model_m2], [adam_opt_m1, adam_opt_m2], dl, names, batch_size)

    del adam_opt_m1
    del adam_opt_m2
    torch.cuda.empty_cache()

    return model_m1, model_m2


def train_stage_2(dl, models, batch_size, names, iters=100 * (10 ** 3), enc_lr=10 ** -4, dec_lr=10 ** -3, optim=None):
    """Train stage 2 of AdapNet++

    The second stage of AdapNet++ trains a modified AdapNet model that has two encoders, each with their own modality
    as input. It uses the pre-trained weights of the encoders from stage 1 and adds Self-Supervised
    Model Adaptation (SSMA) to fuse the two encoders together

    :param dl: data loader
    :param models: pre-trained models from stage 1
    :param iters: number of iterations
    :return: fused model
    """

    if len(models) == 2:
        model_fusion = AdapNet(dl.num_labels, encoders=[models[0].encoder_mod1, models[1].encoder_mod1])
        model_fusion.cuda()

        del models[1]
        del models[0]
        torch.cuda.empty_cache()
    else:
        model_fusion = models[0]

    if optim is None:
        adam_opt = torch.optim.Adam([
            {"params": model_fusion.encoder_mod1.parameters()},
            {"params": model_fusion.encoder_mod2.parameters()},
            {"params": model_fusion.eASPP.parameters()},
            {"params": model_fusion.ssma_s1.parameters()},
            {"params": model_fusion.ssma_s2.parameters()},
            {"params": model_fusion.ssma_res.parameters()},
            {"params": model_fusion.decoder.parameters(), "lr": dec_lr}], lr=enc_lr)
    else:
        adam_opt = optim

    train_iteration(iters, [model_fusion], [adam_opt], dl, names, batch_size)

    del adam_opt
    del optim
    torch.cuda.empty_cache()

    return model_fusion


def train_stage_3(dl, model, batch_size, names, iters=50 * (10 ** 3), enc_lr=0, dec_lr=10 ** -5, optim=None):
    """Train stage 2 of AdapNet++

    The third and last stage of AdapNet++ trains the fused model from stage 2 again, but does not update the weights
    of the two encoders.

    :param dl: data loader
    :param model: fused model1
    :param iters: number of iterations
    :return: final model
    """

    if optim is None:
        adam_opt = torch.optim.Adam([
            {"params": model.eASPP.parameters()},
            {"params": model.ssma_s1.parameters()},
            {"params": model.ssma_s2.parameters()},
            {"params": model.ssma_res.parameters()},
            {"params": model.decoder.parameters()},
            {"params": model.encoder_mod1.parameters(), "lr": enc_lr},
            {"params": model.encoder_mod2.parameters(), "lr": enc_lr}
        ], lr=dec_lr)
    else:
        adam_opt = optim

    train_iteration(iters, [model], [adam_opt], dl, names, batch_size)

    del adam_opt
    torch.cuda.empty_cache()
    gc.collect()

    return model


def create_snapshot(iter, names, models, opts):
    for u, model_name in enumerate(names):
        model_snapshot = {
            'iteration': iter + 1,
            'model_state_dict': models[u].state_dict(),
            'optimizer_state_dict': opts[u].state_dict()
        }
        torch.save(model_snapshot, model_name)
        del model_snapshot
        torch.cuda.empty_cache()
        gc.collect()


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
            train(model, opts[j], input[j], gt_all, fusion=len(models) == 1)

            torch.cuda.empty_cache()
            gc.collect()

        if (i + 1) % save_checkpoint == 0:
            print("Saving at iteration " + str(i + 1))
            create_snapshot(i, names, models, opts)

        if (i + 1) % epochs == 0 and should_evaluate:
            evaluate(models[0], dl, mode="validation")

    if int(args.save) == 1:
        print("Saving trained model after training")
        create_snapshot(iters, names, models, opts)

    evaluate(models[0], dl, mode="validation")
    evaluate(models[0], dl, mode="test")

    torch.cuda.empty_cache()
    gc.collect()


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

    del res
    del batch_loss
    del aux1_loss
    del res_loss
    del aux2_loss
