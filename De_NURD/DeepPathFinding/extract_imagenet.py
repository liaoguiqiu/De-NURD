import os
import pickle
import numpy as np
from skimage import io


path = '/media/annusha/BigPapa/Study/DL/ImageNet32/'
folder = 'train'
list_parts = [i for i in os.listdir(os.path.join(path, folder))
              if os.path.isfile(os.path.join(path, folder, i))]

# list_parts = [list_parts[0]]

out_path = '/media/annusha/BigPapa/Study/DL/ImageNet_images/'

if os.path.isdir(out_path):
    pass
else:
    os.mkdir(out_path)

counter = 0
label_path = '/media/annusha/BigPapa/Study/DL/label_imagenet'

for f in list_parts:
    file = os.path.join(path, folder, f)
    fo = open(file, 'rb')
    entry = pickle.load(fo)
    x = entry['data']
    y = entry['labels']
    mean_img = entry['mean']

    fo.close()

    x = x / np.float32(255)
    # mean_img = mean_img / np.float32(255)

    y = [i - 1 for i in y]

    # x -= mean_img
    image_size = 32
    image_size_2x = image_size * image_size
    x = np.dstack((x[:, :image_size_2x], x[:, image_size_2x:2 * image_size_2x], x[:, 2 * image_size_2x:]))
    x = x.reshape((x.shape[0], image_size, image_size, 3))
    print(x.shape)

    for im, im_y in zip(x, y):
        io.imsave(os.path.join(out_path, '%d.jpeg'%counter), im)
        with open(label_path, '+a') as fl:
            fl.write(os.path.join(out_path, '%d.png  %d\n'%(counter, im_y)))

        counter += 1

        if counter % 1000 == 0:
            print('done %d\n'%counter)

