from sklearn import svm
import os
import torch.utils.data
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import gan_body
from arg_parse import opt
import numpy as np
from sklearn.externals import joblib


# python3 classifier_svm.py --dataroot=/media/annusha/BigPapa/Study/DL/food-101/images --imageSize=32
# --netD=/media/annusha/BigPapa/Study/DL/out_imagenet/netD_epoch_7.pth

transform = transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.CenterCrop(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])

if __name__ == '__main__':

    fname = '50_features_food_101'

    if not opt.train_svm :
        print('save features')
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transform)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                                 shuffle=True, num_workers=int(opt.workers))

        netD = gan_body._netD()
        if opt.netD == '':
            print('load weights for discriminator')
            exit(-1)
        netD.load_state_dict(torch.load(opt.netD))
        print(netD)
        netD.cuda()

        number = len(os.listdir(opt.outf))

        netD.eval()

        input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize).cuda()

        features = np.array([])
        labels = np.array([])


        for i, data in enumerate(dataloader):
            imgs, label = data
            imgs = imgs.cuda()
            input.resize_as_(imgs).copy_(imgs)
            input_v = Variable(input)

            feature = netD.get_features(input_v)
            feature = feature.data.cpu().numpy()
            feature = feature.astype(np.float16)
            if features.size == 0:
                features = feature
                labels = label
            else:
                features = np.concatenate((features, feature), axis=0)
                labels = np.concatenate((labels, label), axis=0)
            if i % 5 == 0 and i :
                print('processed ', i)
                break
            if i == 50:
                break

        # split to train and validation sets
        indexes = list(range(len(labels)))
        print('number of samples ', labels.shape)
        print(features.shape)

        features = np.concatenate((features, labels[:, np.newaxis]), axis=1)

        features = features.astype(np.float16)

        np.savetxt(fname, features)

    else:
        # if load features from file
        print('load features')
        data = np.loadtxt(fname, dtype=np.float16)
        features, labels = data[:, : -1], data[:, -1: ]

        print('number of samples ', labels.shape)
        print(features.shape)

        indexes = list(range(len(labels)))
        np.random.shuffle(indexes)

        ratio = int(0.3 * len(labels))

        train_data, train_labels = features[indexes[: ratio]], labels[indexes[: ratio]]
        val_data, val_labels = features[indexes[ratio :]], labels[indexes[ratio :]]
        print('len train :', len(train_labels))
        print('len val: ', len(val_labels))

        # print('train svm')
        # clf = svm.SVC(decision_function_shape='ovo')
        # clf.fit(train_data, train_labels)
        # joblib.dump(clf, 'svm.pkl')

        print('download svm')
        clf = joblib.load('svm.pkl')


        print('predict svm')
        val_labels = val_labels.squeeze()
        predicted_labels = clf.predict(val_data)
        print(len(val_labels), len(predicted_labels))
        a = predicted_labels == val_labels
        print(np.sum(a))
        accuracy = np.sum(predicted_labels == val_labels) / len(val_labels)

        print('svm results: %.4f accuracy'%accuracy )

        # precision and recall
        uniq_labels = set(val_labels)
        for label in uniq_labels:
            tp = np.sum(predicted_labels[predicted_labels == label] ==
                        val_labels[predicted_labels == label])
            fn_tp = np.sum(val_labels == label)
            fn = fn_tp - tp
            fp = np.sum(predicted_labels[predicted_labels == label] !=
                        val_labels[predicted_labels == label])

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            print('class %d :: precision %.4f  recall %.4f '%(label, precision, recall))











