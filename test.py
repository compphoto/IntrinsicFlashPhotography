import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np


def getColor(color):
    numpy_image = np.zeros((255, 255, 3))
    numpy_color = []
    for c in color:
        numpy_color.append(c)
    numpy_image[:,:,0] = numpy_color[0] * 255
    numpy_image[:,:,1] = numpy_color[1] * 255
    numpy_image[:,:,2] = numpy_color[2] * 255

    return numpy_image


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print(dataset)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_full_path = model.get_image_paths()     # get image paths
        if "generation" not in opt.model:
            colors = model.get_current_colors()
            pred_ambi_shd = visuals['pred_ambi_shd_dec'].cpu()
            color_output = colors['fake_ambient_color'].cpu()
            pred_ambi_shd[:,:,:,0] = pred_ambi_shd[:,:,:,0] * color_output[0]
            pred_ambi_shd[:,:,:,1] = pred_ambi_shd[:,:,:,1] * color_output[1]
            pred_ambi_shd[:,:,:,2] = pred_ambi_shd[:,:,:,2] * color_output[2]
            visuals['pred_ambi_shd_dec'] = pred_ambi_shd


        img_path = data['image_path'][0]
        img_path = img_path.replace(opt.dataroot,'')
        seprator = '_'
        img_full_path = img_path.split('/')
        img_path = [seprator.join(img_path.split('/'))]
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_images(opt.results_dir, opt.phase, webpage, visuals, img_full_path, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        print(img_path)
        print(img_full_path)
    webpage.save()  # save the HTML
