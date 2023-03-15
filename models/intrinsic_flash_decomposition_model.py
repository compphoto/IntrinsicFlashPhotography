import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import numpy
import itertools
from util.image_pool import ImagePool
from torchvision import transforms
import cv2
from models.altered_midas.flash_nets import DecomposeNet
from PIL import Image, ImageDraw


class IntrinsicFlashDecompositionModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--lambda_gradient', type=float, default=0)
        parser.add_argument('--vgg_loss', action='store_true')
        parser.add_argument('--gan_loss', action='store_true')
        parser.add_argument('--no_geometry', action='store_true')
        parser.add_argument('--no_gradient_loss', action='store_true')

        if is_train:
            parser.add_argument('--lambda_feat', type=float, default=40)
            parser.set_defaults(pool_size=0, gan_mode='wgangp')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_color', type=float, default=1)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_L1_dec', 'color']
        if self.opt.vgg_loss:
            self.loss_names += ['VGG_dec']

        if self.opt.gan_loss:
            self.loss_names += ['D_dec', 'G_GAN_dec']

        visual_names_B = ['real_flashPhoto']
        visual_names_A = ['real_ambient', 'fake_ambient_dec']
        visual_names_Sh = ['pred_flsh_shd_dec', 'grnd_flsh_shd_gen']
        self.visual_names = visual_names_B + visual_names_A + visual_names_Sh

        self.colors = ['ambient_shading_color', "fake_ambient_color"]

        self.model_names = ['G_Decomposition']

        if self.isTrain and self.opt.gan_loss:
            self.model_names += ['D_Decomposition']
        if self.opt.no_geometry:
            input_channels = 6
        else:
            input_channels = 10

        self.netG_Decomposition = DecomposeNet(input_channels= input_channels, activation='sigmoid')

        self.netG_Decomposition = networks.init_net(self.netG_Decomposition, gpu_ids=self.gpu_ids)

        if self.isTrain and self.opt.gan_loss:
            self.netD_Decomposition = networks.define_D(6, opt.ndf, opt.netD,
                                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                                                        self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, opt.netD).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.L1Loss()

            if opt.vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_Decomposition.parameters()), lr=opt.lr,
                betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            if self.opt.gan_loss:
                self.optimizer_D = torch.optim.Adam(
                    itertools.chain(self.netD_Decomposition.parameters()), lr=opt.lr2,
                    betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_flashPhoto = input['flashPhoto'].to(self.device)
        self.real_ambient = input['ambient'].to(self.device)
        self.image_paths = input['image_path']
        self.albedo_flshpht = input['albedo_flshpht'].to(self.device)
        self.normals_flshpht = input['normals_flshpht'].to(self.device)
        self.flsh_impl_shd_med = input['flsh_impl_shd_med'].to(self.device)
        self.ambi_impl_shd = input['ambi_impl_shd'].to(self.device)
        self.albedo_med = input['albedo_med'].to(self.device)
        self.ambient_shading_temp = input['ambient_shading_temp'].to(self.device)
        self.depth_flashPhoto = input['depth_flashPhoto'].to(self.device)
        self.ambient_shading_color = input['ambient_shading_color']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.no_geometry:
            decomposition_input = torch.cat(
                (self.real_flashPhoto, self.albedo_flshpht), 1)
        else:
            decomposition_input = torch.cat(
                (self.real_flashPhoto, self.depth_flashPhoto, self.albedo_flshpht, self.normals_flshpht), 1)

        # forward into networks
        self.fake_flash_shading_dec, self.fake_ambient_shading_dec, self.fake_ambient_temp_dec = self.netG_Decomposition(
            decomposition_input)

        # convert the shading predictions to the actual shading space
        dec_flsh_shd = (1.0 / self.fake_flash_shading_dec) - 1.0
        dec_ambi_shd = (1.0 / self.fake_ambient_shading_dec) - 1.0

        # color the ambient shading using the predicted ambient color temps
        self.fake_ambient_color, dec_ambi_shd_clr = networks.color_image(dec_ambi_shd, self.fake_ambient_temp_dec, self.device)
        # compute the new implied albedo from the input image and the predicted shading
        self.dec_impl_alb = self.real_flashPhoto / (dec_flsh_shd + dec_ambi_shd_clr)

        # compute the predicted ambient image from the implied alb and colored ambient shading
        self.fake_ambient_dec = dec_ambi_shd_clr * self.dec_impl_alb

        self.grnd_flsh_shd_gen = (1 - self.flsh_impl_shd_med) * 255
        self.pred_flsh_shd_dec = (1 - self.fake_flash_shading_dec) * 255

    def backward_D_basic(self, netD, real_input, real_output, fake_output):
        # Fake
        fake_input_output = torch.cat((real_input, fake_output), 1)
        pred_fake = netD(fake_input_output.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_input_output = torch.cat((real_input, real_output), 1)
        pred_real = netD(real_input_output)
        loss_D_real = self.criterionGAN(pred_real, True)

        if self.opt.gan_mode == 'wgangp':
            self.loss_gradient_penalty, gradients = networks.cal_gradient_penalty(
                netD, real_input_output, fake_input_output, self.device, lambda_gp=20.0
            )
            self.loss_gradient_penalty.backward(retain_graph=True)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_dec(self):
        self.loss_D_dec = self.backward_D_basic(self.netD_Decomposition, self.real_flashPhoto, self.real_ambient,
                                                self.fake_ambient_dec)

    def backward_G(self):
        self.loss_G_GAN_dec = 0
        if self.opt.gan_loss:
            fake_ambient = torch.cat((self.real_flashPhoto, self.fake_ambient_dec), 1)
            pred_fake = self.netD_Decomposition(fake_ambient)
            self.loss_G_GAN_dec = self.criterionGAN(pred_fake, True)

        self.loss_VGG_dec = 0
        if self.opt.vgg_loss:
            self.loss_VGG_dec = self.criterionVGG(
                self.fake_ambient_dec,
                self.real_ambient
            ) * self.opt.lambda_feat

        self.loss_G_L1_dec = networks.l1_grad_loss(self.fake_flash_shading_dec, self.flsh_impl_shd_med,
                                          self.opt.no_gradient_loss) * self.opt.lambda_L1 + \
                             networks.l1_grad_loss(self.fake_ambient_shading_dec, self.ambi_impl_shd,
                                          self.opt.no_gradient_loss) * self.opt.lambda_L1 + \
                             networks.l1_grad_loss(self.dec_impl_alb, self.albedo_med,
                                          self.opt.no_gradient_loss) * self.opt.lambda_L1

        self.loss_color = self.criterionL1(self.ambient_shading_temp,
                                           self.fake_ambient_temp_dec.squeeze()) * self.opt.lambda_L1 * self.opt.lambda_color
        self.loss_G = (0.5 * self.loss_G_L1_dec) + \
                      self.loss_VGG_dec + \
                      self.loss_G_GAN_dec + \
                      self.loss_color

        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        self.forward()  # compute fake images: G(A)
        if self.opt.gan_loss:
            self.set_requires_grad([self.netD_Decomposition], True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D_dec()
            self.optimizer_D.step()  # update D's weights
            # update G
            self.set_requires_grad([self.netD_Decomposition],
                                   False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

