import numpy as np
import cv2
import os
from scipy.ndimage import rotate
import scipy.misc
from tqdm import tqdm
import pandas as pd
import time
import matplotlib.pyplot as plt

def load_miniImageNet():
    """
    :return: train, val, test tensor with shape (X, 600, 84, 84, 3)
    """
    print("Loading resized MiniImageNet from jpg images")
    # resizetargetpath = '/home/hy/DataSets/miniImageNet/resizedminiImages/'
    resizetargetpath = './DataSets/miniImageNet/resizedminiImages/'
    csv_file_dir = './data/miniImagenet'
    def data_loader(csv_file):
        data = pd.read_csv(csv_file, sep=',')
        data = data.filename.tolist()
        img_list = []
        for file in tqdm(data):
            img = scipy.misc.imread(resizetargetpath+file).astype(np.float32)
            img_list.append(img)
        imgs = np.concatenate(img_list, axis=0)
        imgs = np.reshape(imgs, [-1, 600, 84, 84, 3])
        return imgs
    start = time.time()
    train = data_loader(os.path.join(csv_file_dir, 'train.csv'))
    val = data_loader(os.path.join(csv_file_dir, 'val.csv'))

    # fake data to accelerate testing without loading data
    # train = np.ones((64, 600, 84, 84, 3))
    # val = np.ones((16, 600, 84, 84, 3))
    test = data_loader(os.path.join(csv_file_dir, 'test.csv'))
    print("Loading from raw images data cost %.5f s" % (time.time()-start))
    print(train.shape, val.shape, test.shape)

    return train, val, test


class MiniImageNetDataSet():
    def __init__(self, batch_size, classes_per_set=20, samples_per_class=5, seed=2591, shuffle_classes=False):
        '''
        Construct a N-shot MiniImageNet Dataset
        :param batch_size:
        :param classes_per_set:
        :param samples_per_class:
        :param seed:
        :param shuffle_classes:
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
        '''
        np.random.seed(seed)
        self.x_train, self.x_val, self.x_test = load_miniImageNet()
        if shuffle_classes:
            class_ids = np.arange(self.x_train.shape[0])
            np.random.shuffle(class_ids)
            self.x_train = self.x_train[class_ids]
            class_ids = np.arange(self.x_val.shape[0])
            np.random.shuffle(class_ids)
            self.x_val = self.x_val[class_ids]
            class_ids = np.arange(self.x_test.shape[0])
            np.random.shuffle(class_ids)
            self.x_test = self.x_test[class_ids]

        # self.mean = np.mean(list(self.x_train)+list(self.x_val))
        self.mean = 113.77 # precomputed
        # self.std = np.std(list(self.x_train)+list(self.x_val))
        self.std = 70.1899 # precomputed
        print("mean ", self.mean, " std ", self.std)
        self.batch_size = batch_size
        self.n_classes = 100
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test}

    def preprocess_batch(self, x_batch):
        x_batch = (x_batch-self.mean) / self.std
        return x_batch

    def sample_new_batch(self, data_pack):
        """
        Collect batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return:  A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                  data_pack.shape[3], data_pack.shape[4]), dtype=np.float32)
        support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), dtype=np.float32)
        target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]),
                            dtype=np.float32)
        target_y = np.zeros((self.batch_size, ), dtype=np.float32)
        # for each task, there is only one target image for test, for example, 5-way-1-shot,
        # support set contains 5 images and target set contains 1 image.
        for i in range(self.batch_size):
            # Each idx in batch contains a task
            classes_idx = np.arange(data_pack.shape[0])
            samples_idx = np.arange(data_pack.shape[1])
            # not select replicate samples
            choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
            choose_label = np.random.choice(self.classes_per_set, size=1)
            choose_samples = np.random.choice(samples_idx, size=self.samples_per_class+1, replace=False)

            # select out the chosen classes as the task labels, make sure the images are correct
            x_temp = data_pack[choose_classes]
            x_temp = x_temp[:, choose_samples]
            y_temp = np.arange(self.classes_per_set)
            support_set_x[i] = x_temp[:, :-1]
            support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
            # the target of the one-shot learning task, only choose one labels
            target_x[i] = x_temp[choose_label, -1]
            target_y[i] = y_temp[choose_label]

        return support_set_x, support_set_y, target_x, target_y

    def get_batch(self, dataset_name, augment=False):
        """
        Gets next batch from the dataset with name.
        :param dataset_name: The name of the dataset (one of "train", "val", "test")
        :return:
        """
        x_support_set, y_support_set, x_target, y_target = self.sample_new_batch(self.datasets[dataset_name])
        if augment:
            # todo image data augmentation
            # k = np.random.randint(0, 4, size=(self.batch_size, self.classes_per_set))
            k = np.random.choice(a=[-1, -0.25, 0, 0.25, 1], size=(self.batch_size, self.classes_per_set), replace=True)

            x_augmented_support_set = []
            x_augmented_target_set = []
            for b in range(self.batch_size):
                temp_class_support = []

                for c in range(self.classes_per_set):
                    x_temp_support_set = self.rotate_batch(x_support_set[b, c], axis=(1, 2), k=k[b, c])
                    if y_target[b] == y_support_set[b, c, 0]:
                        x_temp_target = self.rotate_batch(x_target[b], axis=(0, 1), k=k[b, c])

                    temp_class_support.append(x_temp_support_set)

                x_augmented_support_set.append(temp_class_support)
                x_augmented_target_set.append(x_temp_target)

            x_support_set = np.array(x_augmented_support_set)
            x_target = np.array(x_augmented_target_set)
        x_support_set = self.preprocess_batch(x_support_set)
        x_target = self.preprocess_batch(x_target)

        return x_support_set, y_support_set, x_target, y_target

    def rotate_batch(self, x_batch, axis, k):
        # print(x_batch.shape, axis, k)
        # x_batch = rotate(x_batch, k*90, reshape=False, axes=axis, mode="nearest")
        x_batch = rotate(x_batch, k*45, reshape=False, axes=axis, mode="nearest")
        return x_batch

    def get_train_batch(self, augment=False):

        """
        Get next training batch
        :return: Next training batch
        """
        return self.get_batch("train", augment)

    def get_test_batch(self, augment=False):

        """
        Get next test batch
        :return: Next test_batch
        """
        return self.get_batch("test", augment)

    def get_val_batch(self, augment=False):

        """
        Get next val batch
        :return: Next val batch
        """
        return self.get_batch("val", augment)

if __name__ == '__main__':
    mini = MiniImageNetDataSet(batch_size=32)
    x_support_set, y_support_set, x_target, y_target = mini.get_test_batch(augment=True)
    print(np.min(x_support_set), np.max(x_support_set)) # -1.62089 2.01211
    print(x_support_set.shape, y_support_set.shape, x_target.shape, y_target.shape)

    for i in range(10):
        for img in x_support_set[i, 2]:
            plt.imshow(img)
            plt.show()
