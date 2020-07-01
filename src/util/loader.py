import json

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader():

    def __init__(self, path, num_examples=10, train_size=0.6, test_size=0.3, date=None):
        """
        Initializes the data loader
        :param path: the path to the data
        :param num_examples: The number of examples to use
        :param train_size: The size (in percentage) of the train set
        :param test_size: The size (in percentage) of the test set
        :param date: The signature/date of the model
        """
        self.path = path
        self.color_map = {}
        classes = np.loadtxt("data/classes.txt")
        for x in classes:
            self.color_map[x[3]] = [x[0], x[1], x[2]]

        self.num_labels = 12 # len(np.unique(classes[:, 3]))
        self.num_examples = num_examples
        self.train_size = int(train_size * self.num_examples)
        self.test_size = int(test_size * self.num_examples)

        if date is None:
            all_samples = np.arange(0, self.num_examples)
            self.train_set = np.random.choice(all_samples, self.train_size, replace=False)
            all_samples = np.delete(all_samples, self.train_set)
            self.test_set = np.random.choice(all_samples, self.test_size, replace=False)
            all_samples = np.delete(all_samples, self.test_set)
            self.validation_set = all_samples
        else:
            dt_name = "models/dataset_" + date + ".txt"
            with open(dt_name, 'r') as file:
                data_sets = json.load(file)
                self.train_set = data_sets["train"]
                self.test_set = data_sets["test"]
                self.validation_set = data_sets["validation"]


    def write_dataset(self, suffix):
        """
        Saves the data loader
        :param suffix:
        :return:
        """
        dt = {
            "train": [int(x) for x in list(self.train_set)],
            "test": [int(x) for x in list(self.test_set)],
            "validation": [int(x) for x in list(self.validation_set)]
        }
        dt_name = "models/dataset_" + suffix + ".txt"
        with open(dt_name, 'w') as file:
            file.write(json.dumps(dt))

    def get_color(self, x):
        return self.color_map[x]

    def result_to_image(self, result, iter):
        """
        Converts the output of the network to an actual image
        :param result: The output of the network (with torch.argmax)
        :param iter: The name of the file to save it to
        :return:
        """
        b = result.cpu().numpy()
        bs = b.shape
        data = np.zeros((bs[0], bs[1], 3), dtype=np.uint8)
        for y in range(bs[0]):
            for x in range(bs[1]):
                data[y, x] = self.get_color(b[y, x])

        img = Image.fromarray(data, 'RGB')
        img.save('results/ssma ' + str(iter + 1) + '.png')

    def sample_batch(self, batch_size, mode="train"):
        """
        Samples a batch of images
        :param batch_size: The batch size of the image
        :param mode: The mode for sampling, one of "train", "test" or "validation"
        :return:
        """
        batch_mod1 = []
        batch_mod2 = []
        batch_gt = []
        while len(batch_mod1) < batch_size:
            cur_sample = np.random.choice(self.train_set, 1)[0]
            if mode == "test":
                cur_sample = np.random.choice(self.test_set, 1)[0]
            elif mode == "validation":
                cur_sample = np.random.choice(self.validation_set, 1)[0]

            m1, m2, gt = self.sample(cur_sample)

            if m1 is False:
                continue

            batch_mod1.append(torch.from_numpy(m1).float().to(device))
            batch_mod2.append(torch.from_numpy(m2).float().to(device))
            batch_gt.append(torch.from_numpy(gt).long().to(device))

        return torch.stack(batch_mod1), torch.stack(batch_mod2), torch.stack(batch_gt).long()

    def sample(self, sample_id):
        """
        Samples a single image
        :param sample_id: The ID of the image
        :return:
        """
        a = '%07d' % sample_id
        a = a + ".png"

        try:
            pilRGB = Image.open(self.path + "RGB/" + a).convert('RGB')
            pilDep = Image.open(self.path + "Depth/Depth/" + a).convert('RGB')

            pilRGB = self.data_augmentation(pilRGB)
            pilDep = self.data_augmentation(pilDep)

            imgRGB = np.array(pilRGB)[:, :, ::-1]
            imgDep = np.array(pilDep)[:, :, ::-1]

            imgGT = cv2.imread(self.path + "GT/LABELS/" + a, cv2.IMREAD_UNCHANGED).astype(np.int8)
            modRGB = cv2.resize(imgRGB, dsize=(768, 384), interpolation=cv2.INTER_LINEAR) / 255
            modDepth = cv2.resize(imgDep, dsize=(768, 384), interpolation=cv2.INTER_NEAREST) / 255
            modGT = cv2.resize(imgGT, dsize=(768, 384), interpolation=cv2.INTER_NEAREST)
            modGT = modGT[: , :, 2]

            modGT = np.where(modGT == 16, 6, modGT)
            modGT = np.where(modGT == 18, 8, modGT)
            modGT = np.where(modGT == 19, 8, modGT)
            modGT = np.where(modGT == 20, 8, modGT)

            modGT = np.where(modGT == 15, 9, modGT)
            modGT = np.where(modGT == 14, 9, modGT)

            modGT = np.where(modGT == 12, 11, modGT)
            modGT = np.where(modGT == 17, 11, modGT)

            modGT = np.where(modGT == 21, 2, modGT)
            modGT = np.where(modGT == 13, 3, modGT)
            modGT = np.where(modGT == 22, 3, modGT)

            # opencv saves it as BGR instead of RGB
            return np.array([modRGB[:, :, 2], modRGB[:, :, 1], modRGB[:, :, 0]]), \
                   np.array([modDepth[:, :, 2], modDepth[:, :, 1], modDepth[:, :, 0]]), \
                   modGT
        except IOError:
            print("Error loading " + a)
        return False, False, False

    def data_augmentation(self, mods):
        """
        Augments the data
        :param mods:
        :return:
        """
        rand_crop = np.random.uniform(low=0.8, high=0.9)
        rand_scale = np.random.uniform(low=0.5, high=2.0)
        rand_bright = np.random.uniform(low=0, high=0.4)
        rand_cont = np.random.uniform(low=0, high=0.5)
        transform = transforms.RandomApply([
            transforms.RandomApply([transforms.RandomRotation((-13, 13))], p=0.25),
            transforms.RandomApply([transforms.ColorJitter(brightness=rand_bright)], p=0.25),
            transforms.RandomApply([transforms.ColorJitter(contrast=rand_cont)], p=0.25),
            transforms.RandomApply([transforms.RandomCrop((int(768 * rand_crop), int(384 * rand_crop))),
                                    transforms.Resize((768, 384))], p=0.25),
            transforms.RandomApply([transforms.Resize((int(768 * rand_scale), int(384 * rand_crop))),
                                    transforms.Resize((768, 384))], p=0.25),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=.25),
        ], p=.25)
        transformed_img = transform(mods)

        return transformed_img