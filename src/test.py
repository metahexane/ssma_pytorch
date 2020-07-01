import torch

from adapnet import AdapNet
from util.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def acquire_results(model, samples, dl):
    model.eval()
    with torch.no_grad():
        for x in range(0, len(samples), 2):
            batch_rgb, batch_mod = [], []
            b1_m1, b1_m2, _ = dl.sample(samples[x])
            b2_m1, b2_m2, _ = dl.sample(samples[x + 1])
            batch_rgb.append(torch.from_numpy(b1_m1).float().to(device))
            batch_rgb.append(torch.from_numpy(b2_m1).float().to(device))

            batch_mod.append(torch.from_numpy(b1_m2).float().to(device))
            batch_mod.append(torch.from_numpy(b2_m2).float().to(device))

            batch_rgb = torch.stack(batch_rgb)
            batch_mod = torch.stack(batch_mod)

            _, _, res = model(batch_rgb, batch_mod)

            dl.result_to_image(res.argmax(dim=1)[0], samples[x])
            dl.result_to_image(res.argmax(dim=1)[1], samples[x + 1])

if __name__ == "__main__":
    samples = [33, 1237, 1253, 2343, 4781, 7110]
    load_model = "02_06_2020_02_57"
    dl = DataLoader("data/RAND_CITYSCAPES/", date=load_model)

    mf = AdapNet(12)
    mf_name = "models/adapnet_mod1_" + load_model + ".tar"
    lmfinal = torch.load(mf_name)
    mf.load_state_dict(lmfinal['model_state_dict'])
    mf.cuda()

    acquire_results(mf, samples, dl)