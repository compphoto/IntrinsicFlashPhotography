"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from skimage.color import rgb2lab
import random


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
        elif image_numpy.shape[0] != 1:
            image_numpy = np.transpose(image_numpy, (1, 2, 0))  #
            image_numpy = gamma_correct(image_numpy)
            image_numpy *= 255
        image_numpy = np.where(image_numpy > 255, 255, image_numpy)

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def gamma_correct(rgb):
    """
    converts a linear RGB image to gamma encoded RGB image.
    Parameters:
        rgb (np.array): input image numpy array
    """
    srgb = np.zeros_like(rgb)
    mask1 = (rgb > 0) * (rgb < 0.0031308)
    mask2 = (1 - mask1).astype(bool)
    srgb[mask1] = 12.92 * rgb[mask1]
    srgb[mask2] = 1.055 * np.power(rgb[mask2], 0.41666) - 0.055
    srgb[srgb < 0] = 0
    srgb[srgb > 1] = 1
    return srgb


def average_brightness(image):
    """
    calculates the average brightness based on the L channel in Lab color space
    Parameters:
        image (numpy array)
    Return:
        average brightness
    """
    lab = rgb2lab(image)
    w = len(image)
    h = len(image[0])
    L = lab[:, :, 0]
    L = np.array(L)
    L_flat = np.reshape(L, (w * h))
    L_flat = np.sort(L_flat)
    len_all = len(L_flat)
    leave_out = int(len(L_flat) / 10)
    # leave 20 percent of brightest and darkest pixels
    brightness = sum(L_flat[leave_out:-leave_out])
    brightness = brightness / (len_all * 8 / 10)
    return brightness


def lin(srgb):
    """
    linearize a SRGB image to RGB.
    Parameters:
        srgb (numpy array)
    Returns:
        rgb (numpy array)
    """

    srgb = srgb.astype(np.float)
    rgb = np.zeros_like(srgb).astype(np.float)
    srgb = srgb
    mask1 = srgb <= 0.04045
    mask2 = (1 - mask1).astype(bool)
    rgb[mask1] = srgb[mask1] / 12.92
    rgb[mask2] = ((srgb[mask2] + 0.055) / 1.055) ** 2.4
    rgb = rgb
    return rgb


def lin_pixel(srgb):
    """
    linearize a single value pixel from SRGB to RGB
    Parameters:
        srgb (float)
    Returns:
        rgb (float)
    """
    if srgb <= 0.04045:
        rgb = srgb / 12.92
    else:
        rgb = ((srgb + 0.055) / 1.055) ** 2.4
    return rgb


def shading_color(lin):
    """
    Randomly changes temperature color of an RGB image
    Parameters:
        lin (numpy array) : input image in SRG
    Returns:
        color_rgb (array): Random color assigned to the image in RGB
        colored_lin (numpy array): image with new color temperature
        normalized_temp (int): normalized value of the new color temperature
    """
    kelvin_table = {
        2700: (255, 169, 87),
        2900: (255, 177, 101),
        3100: (255, 184, 114),
        3300: (255, 190, 126),
        3500: (255, 196, 137),
        3700: (255, 201, 148),
        3900: (255, 206, 159),
        4100: (255, 211, 168),
        4300: (255, 215, 177),
        4500: (255, 219, 186),
        4700: (255, 223, 194),
        4900: (255, 227, 202),
        5100: (255, 230, 210),
        5300: (255, 233, 217),
        5500: (255, 236, 224),
        5700: (255, 239, 230),
        5900: (255, 242, 236),
        6100: (255, 244, 242),
        6300: (255, 246, 247),
        6500: (255, 249, 253),
        6700: (252, 247, 255),
        6900: (247, 245, 255),
        7100: (243, 242, 255),
        7300: (239, 240, 255),
        7500: (235, 238, 255),
        7700: (231, 236, 255),
        7900: (228, 234, 255),
        8100: (225, 232, 255),
        8300: (222, 230, 255),
        8500: (220, 229, 255),
        8700: (217, 227, 255),
        8900: (215, 226, 255),
        9100: (212, 225, 255),
        9300: (210, 223, 255),
        9500: (208, 222, 255),
        9700: (207, 221, 255),
        9900: (205, 220, 255),
        10100: (207, 218, 255),
        10300: (205, 217, 255),
        10500: (204, 216, 255),
        10700: (202, 202, 255),

    }
    kelvin_list = [2700, 2900, 3100, 3300, 3500, 3700, 3900,
                   4100, 4300, 4500, 4700, 4900,
                   5100, 5300, 5500, 5700, 5900,
                   6100, 6300, 6500, 6700, 6900,
                   7100, 7300, 7500, 7700, 7900,
                   8100, 8300, 8500, 8700, 8900,
                   9100, 9300, 9500, 9700, 9900,
                   10100, 10300, 10500, 10700,
                   ]
    rand = random.randint(0, len(kelvin_list) - 1)
    kelvin_value = kelvin_list[rand]
    temp = kelvin_table[kelvin_value]
    r, g, b = temp
    color_rgb = r / 255, g / 255, b / 255
    color = (lin_pixel(color_rgb[0]), lin_pixel(color_rgb[1]), lin_pixel(color_rgb[2]))
    colored_lin = lin.copy()
    colored_lin[:, :, 0] = lin[:, :, 0] * color[0]
    colored_lin[:, :, 1] = lin[:, :, 1] * color[1]
    colored_lin[:, :, 2] = lin[:, :, 2] * color[2]

    normalized_temp = (kelvin_value - kelvin_list[0]) / (kelvin_list[-1] - kelvin_list[0])

    return color_rgb, colored_lin, normalized_temp


def get_brightness(rgb, mode='numpy'):
    """
    CCIR601 YIQ" method for computing brightness
    Parameters:
        rgb: input image
        mode: numpy or torch array
    Returns:
        brightness value
    """

    if mode == 'numpy':
        brightness = (0.3 * rgb[:, :, 0]) + (0.59 * rgb[:, :, 1]) + (0.11 * rgb[:, :, 2])
        return brightness[:, :, np.newaxis]
    if mode == 'torch':
        brightness = (0.3 * rgb[0, :, :]) + (0.59 * rgb[1, :, :]) + (0.11 * rgb[2, :, :])
        return brightness.unsqueeze(0)


def diagnose_network(net, name='network'):
    """
    Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count



def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """
    Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """
    create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """
    create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
