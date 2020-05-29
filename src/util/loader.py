import cv2
import numpy as np
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader():

    def __init__(self, num_examples=9400, train_size=0.6, test_size=0.3):
        self.color_map = {}
        classes = np.loadtxt("data/classes.txt")
        for x in classes:
            self.color_map[x[3]] = [x[0], x[1], x[2]]

        self.num_labels = len(np.unique(classes[:, 3]))
        self.num_examples = num_examples
        self.train_size = int(train_size * self.num_examples)
        self.test_size = int(test_size * self.num_examples)

        all_samples = np.arange(0, self.num_examples)
        self.train_set = np.random.choice(all_samples, self.train_size, replace=False)
        all_samples = np.delete(all_samples, self.train_set)
        self.test_set = [ 1, 2, 3, 4 ]#np.random.choice(all_samples, 50, replace=False) # self.test_size
        self.validation_set = [5, 6, 7, 8] #np.random.choice(all_samples, 50, replace=False)

    def get_color(self, x):
        return self.color_map[x]

    def result_to_image(self, result, iter):
        b = result.cpu().numpy()
        bs = b.shape
        data = np.zeros((bs[0], bs[1], 3), dtype=np.uint8)
        for y in range(bs[0]):
            for x in range(bs[1]):
                data[y, x] = self.get_color(b[y, x])

        img = Image.fromarray(data, 'RGB')
        img.save('results/ssma ' + str(iter + 1) + '.png')

    def sample_batch(self, batch_size, mode="train"):
        batch_mod1 = []
        batch_mod2 = []
        batch_gt = []
        for x in range(batch_size):
            cur_sample = np.random.choice(self.train_set, 1)[0]
            if mode == "test":
                cur_sample = np.random.choice(self.test_set, 1)[0]
            elif mode == "validation":
                cur_sample = np.random.choice(self.validation_set, 1)[0]

            m1, m2, gt = self.sample(cur_sample)

            batch_mod1.append(torch.from_numpy(m1).float().to(device))
            batch_mod2.append(torch.from_numpy(m2).float().to(device))
            batch_gt.append(torch.from_numpy(gt).long().to(device))

        return torch.stack(batch_mod1), torch.stack(batch_mod2), torch.stack(batch_gt).long()

    def sample(self, sample_id, path="data/RAND_CITYSCAPES/"):
        a = '%07d' % sample_id
        a = a + ".png"
        imgRGB = cv2.imread(path + "RGB/" + a)
        imgDep = cv2.imread(path + "Depth/Depth/" + a)
        imgGT = cv2.imread(path + "GT/LABELS/" + a)
        modRGB = cv2.resize(imgRGB, dsize=(768, 384), interpolation=cv2.INTER_LINEAR) / 255
        modDepth = cv2.resize(imgDep, dsize=(768, 384), interpolation=cv2.INTER_NEAREST) / 255
        modGT = cv2.resize(imgGT, dsize=(768, 384), interpolation=cv2.INTER_NEAREST)

        # if iter % 5 == 0:
        #     print("Loading in '" + a + "'...")

        # opencv saves it as BGR instead of RGB
        return np.array([modRGB[:,:,2], modRGB[:,:,1], modRGB[:,:,0]]), \
               np.array([modDepth[:,:,2], modDepth[:,:,1], modDepth[:,:,0]]), \
               modGT[:,:,1]

