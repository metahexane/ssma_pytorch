from util.loader import DataLoader
import torch
from adapnet import AdapNet
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics import IoU
import numpy as np

def evaluate(model: AdapNet, dl: DataLoader, mode, batch_size=2):
    model.eval()

    if mode == "test":
        set = dl.test_set
    else:
        set = dl.validation_set

    reps = len(set) // batch_size
    cm = ConfusionMatrix(dl.num_labels)
    iou_cur = IoU(cm)

    with torch.no_grad():
        for _ in range(reps):
            m1, m2, gt = dl.sample_batch(batch_size, mode=mode)
            _, _, res = model(m1, m2)
            res = torch.softmax(res, dim=1)

            # dl.result_to_image(res.argmax(dim=1)[0], np.random.randint(0, 1000))

            cm.update((res, gt))
    return iou_cur.compute()
