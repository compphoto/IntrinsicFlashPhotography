import os
from pymatreader import read_mat
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import skimage
import numpy as np
import cv2
import argparse
import random
import torchvision.transforms as transforms
import torch
import copy
from util.util import average_brightness, shading_color, get_brightness


class IntrinsicsFlashDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        # 10000 is the max dataset size
        if opt.phase == 'test':
            # load 3 test sub datasets
            self.dir_FAID = os.path.join(opt.dataroot, 'FAID_test')
            self.images_dir_FAID = sorted(make_dataset(self.dir_FAID, 100000))

            self.dir_MID = os.path.join(opt.dataroot, 'MID_test')
            self.images_dir_MID = sorted(make_dataset(self.dir_MID + '/2', 100000))

            self.dir_DPD = os.path.join(opt.dataroot, 'DPD_test')
            self.images_dir_DPD = sorted(make_dataset(self.dir_DPD + '/1', 100000))
        else:
            # load 3 train sub datasets
            self.dir_FAID = os.path.join(opt.dataroot, 'FAID_train')
            self.images_dir_FAID = sorted(make_dataset(self.dir_FAID, 100000))

            self.dir_MID = os.path.join(opt.dataroot, 'MID_train')
            self.images_dir_MID = sorted(make_dataset(self.dir_MID + '/1', 100000))
            self.images_dir_MID = self.images_dir_MID * 2

            self.dir_DPD = os.path.join(opt.dataroot, 'DPD_train')
            self.images_dir_DPD = sorted(make_dataset(self.dir_DPD + '/1', 100000))
            self.images_dir_DPD = self.images_dir_DPD * 4

        # load images
        all_images = self.images_dir_MID + self.images_dir_DPD + self.images_dir_FAID
        self.images_dir_all = []
        for image in all_images:
            if "flashphoto" in image:
                self.images_dir_all.append(image)

        # load exif files for FAID
        self.exif_dir = os.path.join(opt.dataroot, 'EXIFs')

        self.data_size = opt.load_size
        self.data_root = opt.dataroot

    def __getitem__(self, index):

        if self.opt.normalize_flash == 1:
            norm_flash = 35
        elif self.opt.normalize_flash == 2:
            norm_flash = 5.5

        norm_amb = 15

        image_path_temp = self.images_dir_all[index]

        image_name = image_path_temp.split('/')[-1]
        image_name = image_name.replace("_flashphoto", "")

        if self.opt.phase == 'test':
            if 'FAID_test' in image_path_temp:
                image_path = self.data_root + '/FAID_test' + '/{}'.format(image_name)
                components_path = self.data_root + '/FAID_test' + '_components' + '/{}'.format(
                    image_name)
                depth_path = self.data_root + '/FAID_test' + '_depth' + '/{}'.format(
                    image_name)
            elif 'MID_test' in image_path_temp:
                multi_select = random.randint(2, 2)
                image_path = self.data_root + '/MID_test' + '/{}'.format(
                    multi_select) + '/{}'.format(
                    image_name)
                components_path = self.data_root + '/MID_test' + '_components' + '/{}'.format(
                    multi_select) + '/{}'.format(
                    image_name)
                med_alb_path = self.data_root + '/MID_test' + '_components' + '/{}'.format(
                    1) + '/{}'.format(
                    image_name)
                depth_path = self.data_root + '/MID_test' + '_depth' + '/{}'.format(
                    multi_select) + '/{}'.format(
                    image_name)
            elif 'DPD_test' in image_path_temp:
                portrait_select = random.randint(1, 1)
                image_path = self.data_root + '/DPD_test' + '/{}'.format(
                    portrait_select) + '/{}'.format(
                    image_name)
                components_path = self.data_root + '/DPD_test' + '_components' + '/{}'.format(
                    portrait_select) + '/{}'.format(
                    image_name)
                depth_path = self.data_root + '/DPD_test' + '_depth' + '/{}'.format(
                    portrait_select) + '/{}'.format(
                    image_name)
        else:
            if 'FAID_train' in image_path_temp:
                image_path = self.data_root + '/FAID_train' + '/{}'.format(image_name)
                components_path = self.data_root + '/FAID_train' + '_components' + '/{}'.format(
                    image_name)
                depth_path = self.data_root + '/FAID_train' + '_depth' + '/{}'.format(
                    image_name)
            elif 'MID_train' in image_path_temp:
                multi_select = random.randint(1, 19)
                image_path = self.data_root + '/MID_train' + '/{}'.format(
                    multi_select) + '/{}'.format(image_name)
                components_path = self.data_root + '/MID_train' + '_components' + '/{}'.format(
                    multi_select) + '/{}'.format(
                    image_name)
                med_alb_path = self.data_root + '/MID_train' + '_components' + '/{}'.format(
                    1) + '/{}'.format(
                    image_name)
                depth_path = self.data_root + '/MID_train' + '_depth' + '/{}'.format(
                    multi_select) + '/{}'.format(
                    image_name)
            elif 'DPD_train' in image_path_temp:
                portrait_select = random.randint(1, 20)
                image_path = self.data_root + '/DPD_train' + '/{}'.format(
                    portrait_select) + '/{}'.format(
                    image_name)
                components_path = self.data_root + '/DPD_train' + '_components' + '/{}'.format(
                    portrait_select) + '/{}'.format(
                    image_name)
                depth_path = self.data_root + '/DPD_train' + '_depth' + '/{}'.format(
                    portrait_select) + '/{}'.format(
                    image_name)
        # Load images in rgb
        ambient = Image.open(image_path.replace(".png", "_ambient.png"))
        flash = Image.open(image_path.replace(".png", "_flash.png"))

        # load components
        flashphoto_depth = Image.open(depth_path.replace(".png", "_flash.png"))
        ambient_depth = Image.open(depth_path.replace(".png", "_ambient.png"))
        albedo_flshpht = Image.open(components_path.replace(".png", "_flshpht_alb.png"))
        albedo_amb = Image.open(components_path.replace(".png", "_amb_alb.png"))
        normals_amb = Image.open(components_path.replace(".png", "_ambi_nrm.png"))
        normals_flshpht = Image.open(components_path.replace(".png", "_flshpht_nrm.png"))
        if 'MID' in image_path:
            albedo_med = Image.open(med_alb_path.replace(".png", "_med_alb.png"))
        else:
            albedo_med = Image.open(components_path.replace(".png", "_med_alb.png"))


        # resize the images
        ambient = ambient.resize((self.data_size, self.data_size))
        flash = flash.resize((self.data_size, self.data_size))
        albedo_flshpht = albedo_flshpht.resize((self.data_size, self.data_size))
        albedo_amb = albedo_amb.resize((self.data_size, self.data_size))
        albedo_med = albedo_med.resize((self.data_size, self.data_size))
        normals_amb = normals_amb.resize((self.data_size, self.data_size))
        normals_flshpht = normals_flshpht.resize((self.data_size, self.data_size))
        ambient_depth = ambient_depth.resize((self.data_size, self.data_size))
        flashphoto_depth = flashphoto_depth.resize((self.data_size, self.data_size))

        # convert pil images to float
        ambient_float = skimage.img_as_float(ambient)
        flash_float = skimage.img_as_float(flash)
        albedo_med = skimage.img_as_float(albedo_med)
        albedo_amb = skimage.img_as_float(albedo_amb)
        albedo_flshpht = skimage.img_as_float(albedo_flshpht)

        # compute implied shadings
        flsh_impl_shd_med = flash_float / albedo_med.clip(1 / 255.)
        flsh_impl_shd_amb = flash_float / albedo_med.clip(1 / 255.)
        ambi_impl_shd = ambient_float / albedo_med.clip(1 / 255.)

        # grayscale the shadings
        flsh_impl_shd_med = get_brightness(flsh_impl_shd_med)
        flsh_impl_shd_amb = get_brightness(flsh_impl_shd_amb)
        ambi_impl_shd = get_brightness(ambi_impl_shd)

        # inverse the shadings
        flsh_impl_shd_med = 1. / (flsh_impl_shd_med + 1.)
        ambi_impl_shd = 1. / (ambi_impl_shd + 1.)
        flsh_impl_shd_amb = 1. / (flsh_impl_shd_amb + 1.)

        # normalize ambient
        if self.opt.normalize_ambient == 1:
            ambient_brightness = average_brightness(ambient_float)
            ambient_float = ambient_float * norm_amb / ambient_brightness
            ambient_float[ambient_float < 0] = 0
            ambient_float[ambient_float > 1] = 1

        # normalize flash
        if self.opt.normalize_flash == 1:
            flash_brightness = average_brightness(flash_float)
        elif self.opt.normalize_flash == 2:
            if "FAID" in image_path:
                exif_path = self.exif_dir + "/" + image_name.replace(".png", "_flash.mat")
                exif = read_mat(exif_path)
                flash_brightness = exif['metadata']['DigitalCamera']['BrightnessValue']
            elif "DPD" in image_path:
                flash_brightness = 5
            elif "MID" in image_path:
                flash_brightness = 9.26
        if self.opt.normalize_flash > 0:
            flash_float = flash_float * norm_flash / flash_brightness
            flash_float[flash_float < 0] = 0
            flash_float[flash_float > 1] = 1

        # compute white balanced flash photo
        flashPhoto_float_wb = flash_float + ambient_float
        flashPhoto_float_wb[flashPhoto_float_wb < 0] = 0
        flashPhoto_float_wb[flashPhoto_float_wb > 1] = 1

        # compute ambient color and color ambient
        ambient_shading_color, ambient_colored_float, ambient_shading_temp = shading_color(ambient_float.copy())

        # compute flashPhoto
        flashPhoto_float = flash_float + ambient_colored_float
        flashPhoto_float[flashPhoto_float < 0] = 0
        flashPhoto_float[flashPhoto_float > 1] = 1

        # compute albedo of flashPhoto
        albedo_flshpht[:, :, 0] = albedo_flshpht[:, :, 0] * ambient_shading_color[0] / 2 + albedo_flshpht[:, :, 0] / 2
        albedo_flshpht[:, :, 1] = albedo_flshpht[:, :, 1] * ambient_shading_color[1] / 2 + albedo_flshpht[:, :, 1] / 2
        albedo_flshpht[:, :, 2] = albedo_flshpht[:, :, 2] * ambient_shading_color[2] / 2 + albedo_flshpht[:, :, 2] / 2

        # convert numpy arrays to Image
        albedo_med = Image.fromarray((albedo_med * 255).astype('uint8'))
        albedo_amb = Image.fromarray((albedo_amb * 255).astype('uint8'))
        albedo_flshpht = Image.fromarray((albedo_flshpht * 255).astype('uint8'))
        ambi_impl_shd = Image.fromarray((np.squeeze(ambi_impl_shd) * 255).astype('uint8'))
        flsh_impl_shd_med = Image.fromarray((np.squeeze(flsh_impl_shd_med) * 255).astype('uint8'))
        flsh_impl_shd_amb = Image.fromarray((np.squeeze(flsh_impl_shd_amb) * 255).astype('uint8'))
        ambient = Image.fromarray((ambient_colored_float * 255).astype('uint8'))
        ambient_wb = Image.fromarray((ambient_float * 255).astype('uint8'))
        flashPhoto = Image.fromarray((flashPhoto_float * 255).astype('uint8'))
        flashPhoto_wb = Image.fromarray((flashPhoto_float_wb * 255).astype('uint8'))

        # transform to tensors
        transform_params = get_params(self.opt, ambient.size)
        rgb_transform = get_transform(self.opt, transform_params, grayscale=False)
        grayscale_transform = get_transform(self.opt, transform_params, grayscale=True)

        ambient_depth = grayscale_transform(ambient_depth)
        flashphoto_depth = grayscale_transform(flashphoto_depth)
        ambient = rgb_transform(ambient)
        flashPhoto = rgb_transform(flashPhoto)
        flashPhoto_wb = rgb_transform(flashPhoto_wb)
        ambient_wb = rgb_transform(ambient_wb)
        albedo_flshpht = rgb_transform(albedo_flshpht)
        albedo_amb = rgb_transform(albedo_amb)
        albedo_med = rgb_transform(albedo_med)
        flsh_impl_shd_med = grayscale_transform(flsh_impl_shd_med)
        flsh_impl_shd_amb = grayscale_transform(flsh_impl_shd_amb)
        ambi_impl_shd = grayscale_transform(ambi_impl_shd)
        normals_amb = rgb_transform(normals_amb)
        normals_flshpht = rgb_transform(normals_flshpht)

        return {'flashPhoto': flashPhoto, 'flashPhoto_wb': flashPhoto_wb, 'ambient': ambient,
                'ambient_wb': ambient_wb, 'ambi_impl_shd': ambi_impl_shd,
                'flsh_impl_shd_med': flsh_impl_shd_med, 'flsh_impl_shd_amb': flsh_impl_shd_amb,
                'albedo_med': albedo_med, 'albedo_flshpht': albedo_flshpht, 'albedo_amb': albedo_amb,
                'normals_amb': normals_amb, 'normals_flshpht': normals_flshpht,
                'depth_flashPhoto': flashphoto_depth,
                'depth_ambient': ambient_depth, "ambient_shading_temp": ambient_shading_temp,
                'ambient_shading_color': ambient_shading_color, 'image_path': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images_dir_all)
