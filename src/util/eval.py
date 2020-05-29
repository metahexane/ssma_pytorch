from util.loader import DataLoader
import torch
from adapnet import AdapNet
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics import IoU
import torch.nn as nn
import numpy as np
def evaluate(model: AdapNet, dl: DataLoader, mode, batch_size=2):
    if mode == "test":
        set = dl.test_set
    else:
        set = dl.validation_set

    reps = len(set) // batch_size
    cm = ConfusionMatrix(dl.num_labels)
    iou_cur = IoU(cm)

    # total_iou = torch.zeros(dl.num_labels)
    with torch.no_grad():
        for _ in range(reps):
            m1, m2, gt = dl.sample_batch(batch_size, mode=mode)
            _, _, res = model(m1, m2)
            # res = torch.softmax(res, dim=1)
            s = nn.Softmax2d()
            res = s(res)
            dl.result_to_image(torch.argmax(res[0], dim=0), np.random.randint(500))
            cm.update((res, gt))
            # total_iou += iou_cur.compute()
    return iou_cur.compute()
