import torchvision
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as t_func
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import random


def imshow(img_original, img_final, interpolation_type, steps):
    npimg_original = img_original.numpy()
    npimg_final = img_final.numpy()
    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(np.transpose(npimg_original, (1, 2, 0)))
    axis[0].set_title('Img original')
    axis[1].imshow(np.transpose(npimg_final, (1, 2, 0)))
    axis[1].set_title(f'Img {steps} rotations')
    fig.suptitle(f'Rotations with {interpolation_type}', fontsize=16)
    plt.show()


def calculate_energy(image):
    energy = torch.sum(image ** 2)
    return energy


def make_interpolation(input_image, output_image, interpolation_type, steps):
    energies = []
    for i in range(steps):
        angle = random.randint(0, 360)
        output_image = t_func.rotate(output_image, angle=angle, interpolation=interpolation_type, fill=None)
        output_image = t_func.rotate(output_image, angle=-angle, interpolation=interpolation_type, fill=None)
        energies.append(calculate_energy(output_image).item())
    imshow(torchvision.utils.make_grid(input_image),
           torchvision.utils.make_grid(output_image), interpolation_type, steps)

    plt.plot(range(steps), energies)
    plt.title('Energy drop for each iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.show()


if __name__ == '__main__':
    im = Image.open('./image.png')

    transform = transforms.Compose([transforms.ToTensor()])
    im = transform(im)

    img_org = torch.nn.functional.pad(input=im, pad=(10, 10, 10, 10), mode='constant', value=0)
    img = img_org

    iterations = 500

    make_interpolation(img_org, img, InterpolationMode.BILINEAR, iterations)
    make_interpolation(img_org, img, InterpolationMode.NEAREST, iterations)
