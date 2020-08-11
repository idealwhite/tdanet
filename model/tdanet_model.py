import torch
from .base_model import BaseModel
from . import network, base_function, external_function
from util import task, util
import itertools
from options.global_config import TextConfig
import pickle

class TDAnet(BaseModel):
    """This class implements the text-guided image completion, for 256*256 resolution"""
    def name(self):
        return "TDAnet Image Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--prior_alpha', type=float, default=0.8,
                            help='factor to contorl prior variation: 1/(1+e^((x-0.8)*8))')
        parser.add_argument('--prior_beta', type=float, default=8,
                            help='factor to contorl prior variation: 1/(1+e^((x-0.8)*8))')
        parser.add_argument('--no_maxpooling', action='store_true', help='rm maxpooling in DMA for ablation')
        parser.add_argument('--update_language', action='store_true', help='update language encoder while training')
        parser.add_argument('--detach_embedding', action='store_true',
                            help='do not pass grad to embedding in DAMSM-text end')

        if is_train:
            parser.add_argument('--train_paths', type=str, default='two', help='training strategies with one path or two paths')
            parser.add_argument('--dynamic_sigma', action='store_true', help='change sigma base on mask area')
            parser.add_argument('--lambda_rec_l1', type=float, default=20.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_gen_l1', type=float, default=20.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_kl', type=float, default=20.0, help='weight for kl divergence loss')
            parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for generation loss')
            parser.add_argument('--lambda_match', type=float, default=0.1, help='weight for image-text match loss')

        return parser

    def __init__(self, opt):
        """Initial the pluralistic model"""
        BaseModel.__init__(self, opt)

        self.loss_names = ['kl_rec', 'kl_g', 'l1_rec', 'l1_g', 'gan_g', 'word_g', 'sentence_g', 'ad_l2_g',
                           'gan_rec', 'ad_l2_rec', 'word_rec', 'sentence_rec',  'dis_img', 'dis_img_rec']
        self.log_names = []
        self.visual_names = ['img_m', 'img_truth', 'img_c', 'img_out', 'img_g', 'img_rec']
        self.text_names = ['text_positive']
        self.value_names = ['u_m', 'sigma_m', 'u_post', 'sigma_post', 'u_prior', 'sigma_prior']
        self.model_names = ['E', 'G', 'D', 'D_rec']
        self.distribution = []
        self.prior_alpha = opt.prior_alpha
        self.prior_beta = opt.prior_beta
        self.max_pool = None if opt.no_maxpooling else 'max'

        # define the inpainting model
        self.net_E = network.define_att_textual_e(ngf=32, z_nc=256, img_f=256, layers=5, norm='none', activation='LeakyReLU',
                          init_type='orthogonal', gpu_ids=opt.gpu_ids, image_dim=256, text_dim=256, multi_peak=False, pool_attention=self.max_pool)
        self.net_G = network.define_hidden_textual_g(f_text_dim=768, ngf=32, z_nc=256, img_f=256, L=0, layers=5, output_scale=opt.output_scale,
                                      norm='instance', activation='LeakyReLU', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        # define the discriminator model
        self.net_D = network.define_d(ndf=32, img_f=128, layers=5, model_type='ResDis', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_D_rec = network.define_d(ndf=32, img_f=128, layers=5, model_type='ResDis', init_type='orthogonal', gpu_ids=opt.gpu_ids)

        text_config = TextConfig(opt.text_config)
        self._init_language_model(text_config)

        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()

            self.image_encoder = network.CNN_ENCODER(text_config.EMBEDDING_DIM)
            state_dict = torch.load(
                text_config.IMAGE_ENCODER, map_location=lambda storage, loc: storage)
            self.image_encoder.load_state_dict(state_dict)
            self.image_encoder.eval()
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                self.image_encoder.cuda()
            base_function._freeze(self.image_encoder)

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters()),
                        filter(lambda p: p.requires_grad, self.net_E.parameters())), lr=opt.lr, betas=(0.0, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                                filter(lambda p: p.requires_grad, self.net_D_rec.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.setup(opt)

    def _init_language_model(self, text_config):
        x = pickle.load(open(text_config.VOCAB, 'rb'))
        self.ixtoword = x[2]
        self.wordtoix = x[3]

        word_len = len(self.wordtoix)
        self.text_encoder = network.RNN_ENCODER(word_len, nhidden=256)

        state_dict = torch.load(text_config.LANGUAGE_ENCODER, map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        self.text_encoder.eval()
        if not self.opt.update_language:
            self.text_encoder.requires_grad_(False)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.text_encoder.cuda()

    def set_input(self, input, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.mask = input['mask']
        self.caption_idx = input['caption_idx']
        self.caption_length = input['caption_len']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0], True)
            self.mask = self.mask.cuda(self.gpu_ids[0], True)

        # get I_m and I_c for image with mask and complement regions for training
        self.img_truth = self.img * 2 - 1
        self.img_m = self.mask * self.img_truth
        self.img_c =  (1 - self.mask) * self.img_truth

        # get multiple scales image ground truth and mask for training
        self.scale_img = task.scale_pyramid(self.img_truth, self.opt.output_scale)
        self.scale_mask = task.scale_pyramid(self.mask, self.opt.output_scale)

        # About text stuff
        self.text_positive = util.idx_to_caption(
                                    self.ixtoword, self.caption_idx[0].tolist(), self.caption_length[0].item())
        self.word_embeddings, self.sentence_embedding = util.vectorize_captions_idx_batch(
                                                    self.caption_idx, self.caption_length, self.text_encoder)
        self.text_mask = util.lengths_to_mask(self.caption_length, max_length=self.word_embeddings.size(-1))
        self.match_labels = torch.LongTensor(range(len(self.img_m)))
        if len(self.gpu_ids) > 0:
            self.word_embeddings = self.word_embeddings.cuda(self.gpu_ids[0], True)
            self.sentence_embedding = self.sentence_embedding.cuda(self.gpu_ids[0], True)
            self.text_mask = self.text_mask.cuda(self.gpu_ids[0], True)
            self.match_labels = self.match_labels.cuda(self.gpu_ids[0], True)

    def test(self, mark=None):
        """Forward function used in test time"""
        # save the groundtruth and masked image
        self.save_results(self.img_truth, data_name='truth')
        self.save_results(self.img_m, data_name='mask')

        # encoder process
        distribution, f, f_text = self.net_E(
            self.img_m, self.sentence_embedding, self.word_embeddings, self.text_mask, self.mask)
        variation_factor = 0. if self.opt.no_variance else 1.
        q_distribution = torch.distributions.Normal(distribution[-1][0], distribution[-1][1] * variation_factor)
        scale_mask = task.scale_img(self.mask, size=[f[2].size(2), f[2].size(3)])

        # decoder process
        for i in range(self.opt.nsampling):
            z = q_distribution.sample()

            self.img_g, attn = self.net_G(z, f_text, f_e=f[2], mask=scale_mask.chunk(3, dim=1)[0])
            self.img_out = (1 - self.mask) * self.img_g[-1].detach() + self.mask * self.img_m
            self.score = self.net_D(self.img_out)
            self.save_results(self.img_out, i, data_name='out', mark=mark)

    def get_distribution(self, distribution_factors):
        """Calculate encoder distribution for img_m, img_c only in train, all about distribution layer of VAE model"""
        # get distribution
        sum_valid = (torch.mean(self.mask.view(self.mask.size(0), -1), dim=1) - 1e-5).view(-1, 1, 1, 1)
        m_sigma = 1 if not self.opt.dynamic_sigma else (1 / (1 + ((sum_valid - self.prior_alpha) * self.prior_beta).exp_()))
        p_distribution, q_distribution, kl_rec, kl_g = 0, 0, 0, 0
        self.distribution = []
        for distribution in distribution_factors:
            p_mu, p_sigma, q_mu, q_sigma = distribution
            # the assumption distribution for different mask regions
            std_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), m_sigma * torch.ones_like(p_sigma))
            # m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), torch.ones_like(p_sigma))
            # the post distribution from mask regions
            p_distribution = torch.distributions.Normal(p_mu, p_sigma)
            p_distribution_fix = torch.distributions.Normal(p_mu.detach(), p_sigma.detach())
            # the prior distribution from valid region
            q_distribution = torch.distributions.Normal(q_mu, q_sigma)

            # kl divergence
            kl_rec += torch.distributions.kl_divergence(std_distribution, p_distribution)
            if self.opt.train_paths == "one":
                kl_g += torch.distributions.kl_divergence(std_distribution, q_distribution)
            elif self.opt.train_paths == "two":
                kl_g += torch.distributions.kl_divergence(p_distribution_fix, q_distribution)
            self.distribution.append([torch.zeros_like(p_mu), m_sigma * torch.ones_like(p_sigma), p_mu, p_sigma, q_mu, q_sigma])

        return p_distribution, q_distribution, kl_rec, kl_g

    def get_G_inputs(self, p_distribution, q_distribution, f):
        """Process the encoder feature and distributions for generation network, combine two dataflow when implement."""
        f_m = torch.cat([f[-1].chunk(2)[0], f[-1].chunk(2)[0]], dim=0)
        f_e = torch.cat([f[2].chunk(2)[0], f[2].chunk(2)[0]], dim=0)
        scale_mask = task.scale_img(self.mask, size=[f_e.size(2), f_e.size(3)])
        mask = torch.cat([scale_mask.chunk(3, dim=1)[0], scale_mask.chunk(3, dim=1)[0]], dim=0)
        z_p = p_distribution.rsample()
        z_q = q_distribution.rsample()
        z = torch.cat([z_p, z_q], dim=0)
        return z, f_m, f_e, mask

    def forward(self):
        """Run forward processing to get the inputs"""
        # encoder process
        distribution_factors, f, f_text = self.net_E(
            self.img_m, self.sentence_embedding, self.word_embeddings, self.text_mask, self.mask, self.img_c)

        p_distribution, q_distribution, self.kl_rec, self.kl_g = self.get_distribution(distribution_factors)

        # decoder process
        z, f_m, f_e, mask = self.get_G_inputs(p_distribution, q_distribution, f) # prepare inputs: img, mask, distribute

        results, attn = self.net_G(z, f_text, f_e, mask)
        self.img_rec = []
        self.img_g = []
        for result in results:
            img_rec, img_g = result.chunk(2)
            self.img_rec.append(img_rec)
            self.img_g.append(img_g)
        self.img_out = (1-self.mask) * self.img_g[-1].detach() + self.mask * self.img_truth

        self.region_features_rec, self.cnn_code_rec = self.image_encoder(self.img_rec[-1])
        self.region_features_g, self.cnn_code_g = self.image_encoder(self.img_g[-1])


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss +=gradient_penalty

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D, self.net_D_rec)
        ## Note: changed gen path gan loss to rec path
        # self.loss_dis_img = self.backward_D_basic(self.net_D, self.img_truth, self.img_g[-1])
        self.loss_dis_img = self.backward_D_basic(self.net_D, self.img_truth, self.img_rec[-1])
        self.loss_dis_img_rec = self.backward_D_basic(self.net_D_rec, self.img_truth, self.img_rec[-1])

    def backward_G(self):
        """Calculate training loss for the generator"""

        # encoder kl loss
        self.loss_kl_rec = self.kl_rec.mean() * self.opt.lambda_kl * self.opt.output_scale
        self.loss_kl_g = self.kl_g.mean() * self.opt.lambda_kl * self.opt.output_scale

        # Adversarial loss
        base_function._freeze(self.net_D, self.net_D_rec)

        # D loss fake
        D_fake_g = self.net_D(self.img_g[-1])
        self.loss_gan_g = self.GANloss(D_fake_g, True, False) * self.opt.lambda_gan
        D_fake_rec = self.net_D(self.img_rec[-1])
        self.loss_gan_rec = self.GANloss(D_fake_rec, True, False) * self.opt.lambda_gan

        # LSGAN loss
        D_fake = self.net_D_rec(self.img_rec[-1])
        D_real = self.net_D_rec(self.img_truth)
        D_fake_g = self.net_D_rec(self.img_g[-1])
        self.loss_ad_l2_rec = self.L2loss(D_fake, D_real) * self.opt.lambda_gan
        self.loss_ad_l2_g = self.L2loss(D_fake_g, D_real) * self.opt.lambda_gan

        # Text-image consistent loss
        if not self.opt.detach_embedding:
            sentence_embedding = self.sentence_embedding
            word_embeddings = self.word_embeddings
        else:
            sentence_embedding = self.sentence_embedding.detach()
            word_embeddings = self.word_embeddings.detach()


        loss_sentence = base_function.sent_loss(self.cnn_code_rec, sentence_embedding, self.match_labels)
        loss_word, _ = base_function.words_loss(self.region_features_rec, word_embeddings, self.match_labels, \
                                 self.caption_length, len(word_embeddings))
        self.loss_word_rec = loss_word * self.opt.lambda_match
        self.loss_sentence_rec = loss_sentence * self.opt.lambda_match

        loss_sentence = base_function.sent_loss(self.cnn_code_g, sentence_embedding, self.match_labels)
        loss_word, _ = base_function.words_loss(self.region_features_g, word_embeddings, self.match_labels, \
                                 self.caption_length, len(word_embeddings))
        self.loss_word_g = loss_word * self.opt.lambda_match
        self.loss_sentence_g = loss_sentence * self.opt.lambda_match


        # calculate l1 loss ofr multi-scale, multi-depth-level outputs
        loss_l1_rec, loss_l1_g, log_PSNR_rec, log_PSNR_out = 0, 0, 0, 0
        for i, (img_rec_i, img_fake_i, img_out_i, img_real_i, mask_i) in enumerate(zip(self.img_rec, self.img_g, self.img_out, self.scale_img, self.scale_mask)):
            loss_l1_rec += self.L1loss(img_rec_i, img_real_i)
            if self.opt.train_paths == "one":
                loss_l1_g += self.L1loss(img_fake_i, img_real_i)
            elif self.opt.train_paths == "two":
                loss_l1_g += self.L1loss(img_fake_i, img_real_i)

        self.loss_l1_rec = loss_l1_rec * self.opt.lambda_rec_l1
        self.loss_l1_g = loss_l1_g * self.opt.lambda_gen_l1

        # if one path during the training, just calculate the loss for generation path
        if self.opt.train_paths == "one":
            self.loss_l1_rec = self.loss_l1_rec * 0
            self.loss_ad_l2_rec = self.loss_ad_l2_rec * 0
            self.loss_kl_rec = self.loss_kl_rec * 0

        total_loss = 0

        for name in self.loss_names:
            if name != 'dis_img' and name != 'dis_img_rec':
                total_loss += getattr(self, "loss_" + name)

        total_loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        # compute the image completion results
        self.forward()
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
