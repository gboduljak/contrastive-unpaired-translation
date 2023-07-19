import itertools

import numpy as np
import torch
import torch.nn as nn

import util.util as util

from . import networks
from .base_model import BaseModel
from .patchnce import PatchNCELoss
from .ugatit_networks import Discriminator, ResnetGenerator, RhoClipper


class UGATITCUTModel(BaseModel):
  """ This class implements CUT and FastCUT model, described in the paper
  Contrastive Learning for Unpaired Image-to-Image Translation
  Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
  ECCV, 2020

  The code borrows heavily from the PyTorch implementation of CycleGAN
  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
  """
  @staticmethod
  def modify_commandline_options(parser, is_train=True):
    """  Configures options specific for CUT model
    """
    parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

    parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
    parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
    parser.add_argument('--lambda_CAM', type=float, default=1000.0, help='weight for CAM loss')

    parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
    parser.add_argument('--nce_layers', type=str, default='0,5,9,12,16', help='compute NCE loss on which layers')
    parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                        type=util.str2bool, nargs='?', const=True, default=False,
                        help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
    parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
    parser.add_argument('--netF_nc', type=int, default=256)
    parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
    parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')

    parser.add_argument('--flip_equivariance',
                        type=util.str2bool, nargs='?', const=True, default=False,
                        help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

    parser.set_defaults(pool_size=0)  # no image pooling

    opt, _ = parser.parse_known_args()

    parser.set_defaults(checkpoints_dir='./ckpts')
    # Set default parameters for CUT and FastCUT
    if opt.CUT_mode.lower() == "cut":
      parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
    elif opt.CUT_mode.lower() == "fastcut":
      parser.set_defaults(
          nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
          n_epochs=150, n_epochs_decay=50
      )
    else:
      raise ValueError(opt.CUT_mode)

    return parser

  def __init__(self, opt):
    BaseModel.__init__(self, opt)

    # specify the training losses you want to print out.
    # The training/test scripts will call <BaseModel.get_current_losses>
    self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
    self.visual_names = ['real_A', 'fake_B', 'real_B']
    self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

    if opt.nce_idt and self.isTrain:
      self.loss_names += ['NCE_Y']
      self.visual_names += ['idt_B']

    if self.isTrain:
      self.model_names = ['G', 'F', 'D_local', 'D_global']
    else:  # during test time, only load G
      self.model_names = ['G']

    # define networks (both generator and discriminator)

    n_blocks = {
        'resnet_9blocks': 9,
        'resnet_6blocks': 6
    }
    self.netG = ResnetGenerator(
        input_nc=opt.input_nc,
        output_nc=opt.output_nc,
        ngf=opt.ngf,
        n_blocks=n_blocks[opt.netG],
        nce_layers=[int(l) for l in opt.nce_layers.split(',')],
        light=True
    ).to(self.device)

    self.netF = networks.PatchSampleF(
        use_mlp=True,
        init_type=opt.init_type,
        init_gain=opt.init_gain,
        gpu_ids=self.gpu_ids,
        nc=opt.netF_nc,
        device=self.device
    ).to(self.device)

    if self.isTrain:
      # self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
      self.netD_global = Discriminator(
          input_nc=3,
          ndf=opt.ndf,
          n_layers=7
      ).to(self.device)
      self.netD_local = Discriminator(
          input_nc=3,
          ndf=opt.ndf,
          n_layers=5
      ).to(self.device)
      # define loss functions
      self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
      self.criterionNCE = []
      self.criterionMSE = nn.MSELoss().to(self.device)
      self.criterionBCE = nn.BCELoss().to(self.device)

      for _ in self.nce_layers:
        self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

      self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
      self.optimizer_D = torch.optim.Adam(
          itertools.chain(self.netD_global.parameters(),
                          self.netD_local.parameters()),
          lr=opt.lr,
          betas=(opt.beta1, opt.beta2)
      )
      self.optimizers.append(self.optimizer_G)
      self.optimizers.append(self.optimizer_D)

    """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
    self.Rho_clipper = RhoClipper(0, 1)

  def data_dependent_initialize(self, data):
    """
    The feature network netF is defined in terms of the shape of the intermediate, extracted
    features of the encoder portion of netG. Because of this, the weights of netF are
    initialized at the first feedforward pass with some input images.
    Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
    """
    bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
    self.set_input(data)
    self.real_A = self.real_A[:bs_per_gpu]
    self.real_B = self.real_B[:bs_per_gpu]
    self.forward()                     # compute fake images: G(A)
    if self.opt.isTrain:
      self.compute_D_loss().backward()                  # calculate gradients for D
      self.compute_G_loss().backward()                   # calculate graidents for G
      if self.opt.lambda_NCE > 0.0:
        self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
        self.optimizers.append(self.optimizer_F)

  def optimize_parameters(self):
    # forward
    self.forward()

    # update D
    self.set_requires_grad(self.netD_global, True)
    self.set_requires_grad(self.netD_local, True)

    self.optimizer_D.zero_grad()
    self.loss_D = self.compute_D_loss()
    self.loss_D.backward()
    self.optimizer_D.step()

    # forward (as in U-GAT-IT, we forward twice)
    self.forward()

    # update G
    self.set_requires_grad(self.netD_global, False)
    self.set_requires_grad(self.netD_local, False)
    self.optimizer_G.zero_grad()
    if self.opt.netF == 'mlp_sample':
      self.optimizer_F.zero_grad()
    self.loss_G = self.compute_G_loss()
    self.loss_G.backward()
    self.optimizer_G.step()
    if self.opt.netF == 'mlp_sample':
      self.optimizer_F.step()

    self.netG.apply(self.Rho_clipper)

  def set_input(self, input):
    """Unpack input data from the dataloader and perform necessary pre-processing steps.
    Parameters:
        input (dict): include the data itself and its metadata information.
    The option 'direction' can be used to swap domain A and domain B.
    """
    AtoB = self.opt.direction == 'AtoB'
    self.real_A = input['A' if AtoB else 'B'].to(self.device)
    self.real_B = input['B' if AtoB else 'A'].to(self.device)
    self.image_paths = input['A_paths' if AtoB else 'B_paths']

  def forward(self):
    """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
    if self.opt.flip_equivariance:
      self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
      if self.flipped_for_equivariance:
        self.real = torch.flip(self.real, [3])

    self.fake, fake_cam_logit, fake_heatmap = self.netG(self.real)

    self.fake_B = self.fake[:self.real_A.size(0)]
    self.g_fake_B_cam_logit = fake_cam_logit[:self.real_A.size(0)]
    self.g_fake_B_heatmap = fake_heatmap[:self.real_A.size(0)]

    if self.opt.nce_idt:
      self.idt_B = self.fake[self.real_A.size(0):]
      self.g_fake_idt_B_cam_logit = fake_cam_logit[self.real_A.size(0):]
      self.g_fake_idt_B_heatmap = fake_heatmap[self.real_A.size(0):]


  def compute_D_loss(self):
    """Calculate GAN loss for the discriminator"""
    fake = self.fake_B.detach()

    # Fake; stop backprop to the generator by detaching fake_B
    fake_GB_pred, fake_GB_cam_logit, _ = self.netD_global(fake)
    fake_LB_pred, fake_LB_cam_logit, _ = self.netD_local(fake)

    self.loss_D_GB_fake = self.criterionGAN(fake_GB_pred, False).mean()
    self.loss_D_LB_fake = self.criterionGAN(fake_LB_pred, False).mean()
    self.loss_D_fake = self.loss_D_GB_fake + self.loss_D_LB_fake

    # Real
    real_GB_pred, real_GB_cam_logit, _ = self.netD_global(self.real_B)
    real_LB_pred, real_LB_cam_logit, _ = self.netD_local(self.real_B)

    self.loss_D_GB_real = self.criterionGAN(real_GB_pred, True).mean()
    self.loss_D_LB_real = self.criterionGAN(real_LB_pred, True).mean()
    self.loss_D_real = self.loss_D_GB_real + self.loss_D_LB_real

    # D_ad_cam_loss_GB = self.criterionMSE(
    #     real_GB_cam_logit,
    #     torch.ones_like(real_GB_cam_logit).to(self.device)
    # ) + self.criterionMSE(
    #     fake_GB_cam_logit,
    #     torch.zeros_like(fake_GB_cam_logit).to(self.device)
    # )
    # D_ad_cam_loss_LB = self.criterionMSE(
    #     real_LB_cam_logit,
    #     torch.ones_like(real_LB_cam_logit).to(self.device)
    # ) + self.criterionMSE(
    #     fake_LB_cam_logit,
    #     torch.zeros_like(fake_LB_cam_logit).to(self.device)
    # )
    D_ad_cam_loss_GB = (
        self.criterionGAN(
            real_GB_cam_logit,
            True
        ).mean() +
        self.criterionGAN(
            fake_GB_cam_logit,
            False
        ).mean()
    )
    D_ad_cam_loss_LB = (
        self.criterionGAN(
            real_LB_cam_logit,
            True
        ).mean() +
        self.criterionGAN(
            fake_LB_cam_logit,
            False
        ).mean()
    )
    self.loss_D_cam = D_ad_cam_loss_GB + D_ad_cam_loss_LB

    # combine loss and calculate gradients
    self.loss_D = (self.loss_D_fake + self.loss_D_real + self.loss_D_cam) * 0.5
    return self.loss_D

  def compute_G_loss(self):
    """Calculate GAN and NCE loss for the generator"""
    fake = self.fake_B
    # First, G(A) should fake the discriminator
    if self.opt.lambda_GAN > 0.0:
      fake_GB_pred, fake_GB_cam_logit, _ = self.netD_global(fake)
      fake_LB_pred, fake_LB_cam_logit, _ = self.netD_local(fake)
      # G_ad_cam_loss_GB = self.criterionMSE(
      #     fake_GB_cam_logit,
      #     torch.ones_like(fake_GB_cam_logit).to(self.device)
      # )
      # G_ad_cam_loss_LB = self.criterionMSE(
      #     fake_LB_cam_logit,
      #     torch.ones_like(fake_LB_cam_logit).to(self.device)
      # )
      self.loss_G_GAN = (
          self.criterionGAN(fake_GB_pred, True).mean() +
          self.criterionGAN(fake_LB_pred, True).mean() +
          self.criterionGAN(fake_GB_cam_logit, True).mean() +
          self.criterionGAN(fake_LB_cam_logit, True).mean()
      ) * self.opt.lambda_GAN
    else:
      self.loss_G_GAN = 0.0

    # Second, G(A) should update CAM
    self.loss_G_CAM = self.criterionBCE(
        self.g_fake_B_cam_logit,
        torch.ones_like(self.g_fake_B_cam_logit).to(self.device)
    ) + self.criterionBCE(
        self.g_fake_idt_B_cam_logit,
        torch.zeros_like(self.g_fake_idt_B_cam_logit).to(self.device)
    )


    if self.opt.lambda_NCE > 0.0:
      self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
    else:
      self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

    if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
      self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
      loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
    else:
      loss_NCE_both = self.loss_NCE

    self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_G_CAM * self.opt.lambda_CAM
    return self.loss_G

  def calculate_NCE_loss(self, src, tgt):
    n_layers = len(self.nce_layers)
    feat_q = self.netG(tgt, encode_only=True)

    if self.opt.flip_equivariance and self.flipped_for_equivariance:
      feat_q = [torch.flip(fq, [3]) for fq in feat_q]

    feat_k = self.netG(src, encode_only=True)
    feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
    feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

    total_nce_loss = 0.0
    for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
      loss = crit(f_q, f_k) * self.opt.lambda_NCE
      total_nce_loss += loss.mean()

    return total_nce_loss / n_layers
