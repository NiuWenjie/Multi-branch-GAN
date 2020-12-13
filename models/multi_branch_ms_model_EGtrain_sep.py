import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class MultiBranchMSModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'encoded_A', 'D_B', 'G_B', 'cycle_B', 'encoded_B', 'ms', 'content', 'style']# , 'perceptual']
        visual_names_A = ['real_A', 'fake_B', 'fake_B2', 'encoded_A', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'fake_A2', 'encoded_B', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            pass

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        if self.isTrain:
            self.model_names = ['E', 'S', 'G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['E', 'G_A', 'G_B']


        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netE = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netE, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netS = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netS, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.vgg = networks.VGG19().to(self.device)
        self.loss_per = networks.PerceptualLoss()

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_E = torch.optim.Adam(itertools.chain(self.netE.parameters(), self.netS.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_E)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = torch.randn(batchSize, nz, 1, 1).to(self.device)
        return z*0.01

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.nz = 8
        self.z_random = self.get_z_random(self.real_A.size(0), self.nz)
        self.z_random2 = self.get_z_random(self.real_A.size(0), self.nz)

    def forward(self):
        self.content_A, self.content_B = self.netE(self.real_A), self.netE(self.real_B)
        self.style_A, self.style_B = self.netS(self.real_A), self.netS(self.real_B)
        # self.style_A = networks.norm(self.style_A)

        # real_A->fake_B->rec_A
        self.fake_B = self.netG_B(self.content_A, self.z_random)
        self.fake_B2 = self.netG_B(self.content_A, self.z_random2)
        self.encoded_A = self.netG_A(self.content_A, self.style_A)
        self.content_A_ = self.netE(self.fake_B)
        self.rec_A = self.netG_A(self.content_A_, self.style_A)

        # real_B->fake_A->rec_B
        self.fake_A = self.netG_A(self.content_B, self.z_random)
        self.fake_A2 = self.netG_A(self.content_B, self.z_random2)
        self.encoded_B = self.netG_B(self.content_B, self.style_B)
        self.content_B_ = self.netE(self.fake_A)
        self.rec_B = self.netG_B(self.content_B_, self.z_random)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B2 = self.fake_B_pool.query(self.fake_B2)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A += self.backward_D_basic(self.netD_A, self.real_B, fake_B2)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        fake_A2 = self.fake_A_pool.query(self.fake_A2)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B += self.backward_D_basic(self.netD_B, self.real_A, fake_A2)

    def backward_EG(self):
        self.loss_encoded_A = self.criterionIdt(self.encoded_A, self.real_A) * 5
        self.loss_encoded_B = self.criterionIdt(self.encoded_B, self.real_B) * 5

        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * 10
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * 10

        # Content loss
        self.loss_content_A = self.criterionIdt(self.content_A, self.content_A_)
        self.loss_content_B = self.criterionIdt(self.content_B, self.content_B_)
        self.loss_content = self.loss_content_A + self.loss_content_B

        self.loss_E = self.loss_encoded_A + self.loss_encoded_B +\
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_content

        self.loss_E.backward(retain_graph=True)

    # def backward_E(self):
    #     # Content loss
    #     self.loss_content_A = self.criterionIdt(self.content_A, self.content_A_)
    #     self.loss_content_B = self.criterionIdt(self.content_B, self.content_B_)
    #     self.loss_content = self.loss_content_A + self.loss_content_B
    #
    #     # L = loss + style_loss 权重问题
    #     self.loss_percptual_A = self.loss_per(self.real_A, self.fake_B)
    #     self.loss_percptual_B = self.loss_per(self.real_B, self.fake_A)
    #     self.loss_perceptual = (self.loss_percptual_A + self.loss_percptual_B) * 10
    #
    #     self.loss_E = self.loss_content + \
    #                   self.loss_perceptual #  + \
    #                   # self.loss_ms
    #
    #     self.loss_E.backward(retain_graph=True)

    def backward_G(self):
        # adv loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        if True:
            self.loss_G_A += self.criterionGAN(self.netD_A(self.fake_B2), True)
            self.loss_G_B += self.criterionGAN(self.netD_B(self.fake_A2), True)

        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * 10
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * 10
        #
        # self.loss_encoded_A = self.criterionIdt(self.encoded_A, self.real_A) * 5
        # self.loss_encoded_B = self.criterionIdt(self.encoded_B, self.real_B) * 5

        # Mode seeking loss
        self.lz_AB = torch.mean(torch.abs(self.fake_B2 - self.fake_B)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
        self.lz_BA = torch.mean(torch.abs(self.fake_A2 - self.fake_A)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
        self.eps = 1 * 1e-5
        self.loss_lz_AB = 1 / (self.lz_AB + self.eps)
        self.loss_lz_BA = 1 / (self.lz_BA + self.eps)
        self.loss_ms = self.loss_lz_AB + self.loss_lz_BA

        # print('==========')
        # print(self.style_A)
        # print('----------')
        # print(self.z_random)
        # print('++++++++++')
        # print(self.content_A)
        self.loss_style_A = torch.mean(torch.abs(self.style_A - self.z_random))
        self.loss_style_B = torch.mean(torch.abs(self.style_B - self.z_random))
        self.loss_style = (self.loss_style_A + self.loss_style_B)*0.1

        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_ms + self.loss_style

        # self.loss_cycle_A + self.loss_cycle_B + \
        #               self.loss_encoded_A + self.loss_encoded_B + \

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        # E and G
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.backward_G()
        self.optimizer_E.step()
        self.optimizer_G.step()