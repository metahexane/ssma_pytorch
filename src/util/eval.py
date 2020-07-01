import torch
from ignite.metrics import IoU
from ignite.metrics.confusion_matrix import ConfusionMatrix

from adapnet import AdapNet
from util.loader import DataLoader


def evaluate(model: AdapNet, dl: DataLoader, mode, batch_size=2):
    """
    Evaluates the model, uses IoU as the metric
    :param model: The model to evaluate
    :param dl: The DataLoader of the model
    :param mode: The evaluations mode, one of "test" or "validation"
    :param batch_size: The batch size for the evaluation
    :return:
    """
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
            cm.update((res, gt))

    iou_score = iou_cur.compute()

    print("Evaluation of " + mode + " set")
    print("mIoU: " + str(iou_score.mean().item()))
    print("IoU: " + str(iou_score))

    return iou_score
