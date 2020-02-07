import os
import os.path
import numpy as np
import pickle

import torch.utils.data as data

class IMAGENET(data.Dataset):
    base_folder = ''
    train_list = []
    test_list = []

    def __init__(self, root, train=True, transform=None, target_function=None, image_size=32):
        self.root = root
        self.transform = transform
        self.target_function = target_function
        self.train = train
        self.image_size = image_size
        image_size_2x = self.image_size * self.image_size

        # now load the picked numpy arrays
        if self.train:

            # create list of pickle files
            self.base_folder = 'train'
            self.train_list = [i for i in os.listdir(os.path.join(self.root, self.base_folder)) if
                               os.path.isfile(os.path.join(self.root, self.base_folder, i))]
            self.train_list = [self.train_list[0]] + [self.train_list[1]]
            print(self.train_list)

            self.train_data = []
            self.train_labels = []
            data_size = 0
            self.img_mean = None


            for f in self.train_list:
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                entry = pickle.load(fo)
                x = entry['data']
                y = entry['labels']
                mean_img = entry['mean']
                fo.close()

                x = x / np.float32(255)
                mean_img = mean_img / np.float32(255)

                if self.img_mean is None:
                    self.img_mean = mean_img
                else:
                    self.img_mean += mean_img


                y = [i - 1 for i in y]
                self.train_labels += y

                data_size += x.shape[0]

                x -= mean_img
                x = np.dstack((x[:, :image_size_2x], x[:, image_size_2x:2 * image_size_2x], x[:, 2 * image_size_2x:]))
                x = x.reshape((x.shape[0], self.image_size, self.image_size, 3))


                self.train_data.append(x)

            self.train_data = np.concatenate(self.train_data, axis=0)
            self.img_mean = self.img_mean / len(self.train_list)
            print('data size\n', data_size)


        else:
            self.base_folder = 'val'

            self.test_list = [i for i in os.listdir(os.path.join(self.root, self.base_folder)) if
                               os.path.isfile(os.path.join(self.root, self.base_folder, i))]

            f = self.test_list[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            x = entry['data']
            self.test_labels = entry['labels']
            fo.close()

            self.test_labels = [i - 1 for i in self.test_labels]

            x = x / np.float32(255)
            x -= self.img_mean

            x = np.dstack((x[:, :image_size_2x], x[:, image_size_2x:2 * image_size_2x], x[:, 2 * image_size_2x:]))
            x = x.reshape((x.shape[0], self.image_size, self.image_size, 3)).transpose(0, 3, 1, 2)

            self.test_data = x


    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index]. self.test_labels[index]


        if self.transform is not None:
            img = self.transform(img)

        if self.target_function is not None:
            target = self.target_function(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_function.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



