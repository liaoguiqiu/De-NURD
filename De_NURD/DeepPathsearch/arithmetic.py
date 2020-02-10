import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import gan_body
from arg_parse import opt
from matplotlib import pyplot as plt

# python3 arithmetic.py --netG=/media/annusha/BigPapa/Study/DL/out/netG_epoch_24.pth
# --nz=100 --ngf=64


netG = gan_body._netG()
if opt.netG == '':
    print('load weights for generator')
    exit(-1)
netG.load_state_dict(torch.load(opt.netG))
print(netG)
netG.cuda()


def preprocess_img(img_v):
    img = img_v.data.cpu().numpy()
    img = img.transpose(0, 2, 3, 1).squeeze()
    img += 1
    img /= 2
    return img

def close_event():
    plt.close()

fig = plt.figure()
timer = fig.canvas.new_timer(interval=3000)
timer.add_callback(close_event)
def choose_pic():
    print('0 - next image\n1 - choose this one')
    while True:
        noise = torch.FloatTensor(1, opt.nz, 1, 1).normal_(0, 1)
        noise_v = Variable(noise).cuda()
        fake_v = netG(noise_v)
        fake = preprocess_img(fake_v)
        print(fake.shape)
        plt.imshow(fake)

        # if you don't want to choose images, will be random
        user_input = '1'

        #if you want to choose images
        timer.start()
        plt.show()
        user_input = input()

        print('user input', user_input)
        if user_input == '1':
            print('out')
            return noise


if __name__ == '__main__' :
    noise_A = torch.FloatTensor(3, opt.nz, 1, 1)
    noise_B = torch.FloatTensor(3, opt.nz, 1, 1)
    noise_C = torch.FloatTensor(3, opt.nz, 1, 1)


    for i in range(3):
        A = choose_pic()
        noise_A[i] = A

        B = choose_pic()
        noise_B[i] = B


    mean_A = torch.mean(noise_A, 0, keepdim=True)
    mean_B = torch.mean(noise_B, 0, keepdim=True)

    final_noise = mean_A.add(mean_B)

    noise_A_v = Variable(noise_A).cuda()
    image_A_v = netG(noise_A_v)

    noise_B_v = Variable(noise_B).cuda()
    image_B_v = netG(noise_B_v)

    image_C = None

    mean_A_v = Variable(mean_A).cuda()
    mean_A_im_v = netG(mean_A_v)
    image_A_v = torch.cat((image_A_v, mean_A_im_v), 0)
    image_A = preprocess_img(image_A_v)

    mean_B_v = Variable(mean_B).cuda()
    mean_B_im_v = netG(mean_B_v)
    image_B_v = torch.cat((image_B_v, mean_B_im_v), 0)
    image_B = preprocess_img(image_B_v)


    final_noise_v = Variable(final_noise).cuda()
    final_im_v = netG(final_noise_v)
    final_im = preprocess_img(final_im_v)


    plt.figure(1)

    #number of set of 3 images
    num_of_im = 2
    for i in range(num_of_im):
        if i == 0:
            im_show = image_A
            text = 'A%d'
        if i == 1:
            im_show = image_B
            text = 'B%d'

        plt.subplot(num_of_im, 4, i * 4 + 1)
        plt.axis('off')
        plt.title(text%(1))
        plt.imshow(im_show[0])
        plt.subplot(num_of_im, 4, i * 4 + 2)
        plt.axis('off')
        plt.title(text % (2))
        plt.imshow(im_show[1])
        plt.subplot(num_of_im, 4, i * 4 + 3)
        plt.axis('off')
        plt.title(text % (3))
        plt.imshow(im_show[2])
        plt.subplot(num_of_im, 4, i * 4 + 4)
        plt.axis('off')
        plt.title(text[0] + ' mean')
        plt.imshow(im_show[3])
        plt.savefig('means.png')

    plt.figure(2)
    plt.axis('off')
    plt.title('A_mean + B_mean')
    plt.imshow(final_im)
    plt.savefig('A_B.png')
    plt.show()







