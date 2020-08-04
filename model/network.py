from .base_function import *
from .external_function import SpectralNorm
import torch.nn.functional as F
from util import task


##############################################################################################################
# Network function
##############################################################################################################
def define_e(input_nc=3, ngf=64, z_nc=512, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[]):

    net = ResEncoder(input_nc, ngf, z_nc, img_f, L, layers, norm, activation, use_spect, use_coord)

    return init_net(net, init_type, activation, gpu_ids)

def define_textual_e(input_nc=3, ngf=64, z_nc=512, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[], image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
    net = TextualResEncoder(input_nc, ngf, z_nc, img_f, L, layers, norm, activation, use_spect, use_coord, image_dim, text_dim, multi_peak, pool_attention)
    return init_net(net, init_type, activation, gpu_ids)

def define_att_textual_e(input_nc=3, ngf=64, z_nc=512, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[], image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
    net = AttTextualResEncoder(input_nc, ngf, z_nc, img_f, L, layers, norm, activation, use_spect, use_coord, image_dim, text_dim, multi_peak, pool_attention)
    return init_net(net, init_type, activation, gpu_ids)

def define_contract_e(input_nc=3, ngf=64, z_nc=512, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[], image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
    net = ContrastResEncoder(input_nc, ngf, z_nc, img_f, L, layers, norm, activation, use_spect, use_coord, image_dim, text_dim, multi_peak, pool_attention)
    return init_net(net, init_type, activation, gpu_ids)

def define_pos_textual_e(input_nc=3, ngf=64, z_nc=512, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[], image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
    net = PosAttTextualResEncoder(input_nc, ngf, z_nc, img_f, L, layers, norm, activation, use_spect, use_coord, image_dim, text_dim, multi_peak, pool_attention)
    return init_net(net, init_type, activation, gpu_ids)

def define_word_attn_e(input_nc=3, ngf=64, z_nc=512, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[], image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
    net = WordAttnEncoder(input_nc, ngf, z_nc, img_f, L, layers, norm, activation, use_spect, use_coord, image_dim, text_dim, multi_peak, pool_attention)
    return init_net(net, init_type, activation, gpu_ids)

def define_constraint_e(input_nc=3, ngf=64, z_nc=512, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[], image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
    net = ConstraintResEncoder(input_nc, ngf, z_nc, img_f, L, layers, norm, activation, use_spect, use_coord, image_dim, text_dim, multi_peak, pool_attention)
    return init_net(net, init_type, activation, gpu_ids)

def define_g(output_nc=3, ngf=64, z_nc=512, img_f=512, L=1, layers=5, norm='instance', activation='ReLU', output_scale=1,
             use_spect=True, use_coord=False, use_attn=True, init_type='orthogonal', gpu_ids=[]):

    net = ResGenerator(output_nc, ngf, z_nc, img_f, L, layers, norm, activation, output_scale, use_spect, use_coord, use_attn)

    return init_net(net, init_type, activation, gpu_ids)

def define_constrast_g(output_nc=3, ngf=64, z_nc=512, img_f=512, L=1, layers=5, norm='instance', activation='ReLU', output_scale=1,
             use_spect=True, use_coord=False, use_attn=True, init_type='orthogonal', gpu_ids=[]):

    net = ContrastResGenerator(output_nc, ngf, z_nc, img_f, L, layers, norm, activation, output_scale, use_spect, use_coord, use_attn)

    return init_net(net, init_type, activation, gpu_ids)


def define_textual_g(output_nc=3, f_text_dim=384, ngf=64, z_nc=512, img_f=512, L=1, layers=5, norm='instance', activation='ReLU', output_scale=1,
             use_spect=True, use_coord=False, use_attn=True, init_type='orthogonal', gpu_ids=[]):

    net = TextualResGenerator(output_nc, f_text_dim, ngf, z_nc, img_f, L, layers, norm, activation, output_scale, use_spect, use_coord, use_attn)

    return init_net(net, init_type, activation, gpu_ids)

def define_hidden_textual_g(output_nc=3, f_text_dim=384, ngf=64, z_nc=512, img_f=512, L=1, layers=5, norm='instance', activation='ReLU', output_scale=1,
             use_spect=True, use_coord=False, use_attn=True, init_type='orthogonal', gpu_ids=[]):

    net = HiddenResGenerator(output_nc, f_text_dim, ngf, z_nc, img_f, L, layers, norm, activation, output_scale, use_spect, use_coord, use_attn)

    return init_net(net, init_type, activation, gpu_ids)

def define_d(input_nc=3, ndf=64, img_f=512, layers=6, norm='none', activation='LeakyReLU', use_spect=True, use_coord=False,
             use_attn=True,  model_type='ResDis', init_type='orthogonal', gpu_ids=[]):

    if model_type == 'ResDis':
        net = ResDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn)
    elif model_type == 'PatchDis':
        net = SNPatchDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn)

    return init_net(net, init_type, activation, gpu_ids)

def define_textual_attention(image_dim, text_dim, multi_peak=True, init_type='orthogonal',  gpu_ids=[]):
    net = ImageTextAttention(idf=image_dim, cdf=text_dim, multi_peak=multi_peak)
    return init_net(net, init_type, gpu_ids=gpu_ids)


#############################################################################################################
# Network structure
#############################################################################################################
class ResEncoder(nn.Module):
    """
    ResNet Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ngf=64, z_nc=128, img_f=1024, L=6, layers=6, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False):
        super(ResEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # inference part
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf *mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'infer_prior' + str(i), block)

        self.posterior = ResBlock(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        self.prior = ResBlock(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)

    def forward(self, img_m, img_c=None):
        """
        :param img_m: image with mask regions I_m
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        """

        if type(img_c) != type(None):
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        # infer part
        # during the training, we have two paths, during the testing, we only have one paths
        if type(img_c) != type(None):
            distribution = self.two_paths(out)
            return distribution, feature
        else:
            distribution = self.one_path(out)
            return distribution, feature

    def one_path(self, f_in):
        """one path for baseline training or testing"""
        f_m = f_in
        distribution = []

        # infer state
        for i in range(self.L):
            infer_prior = getattr(self, 'infer_prior' + str(i))
            f_m = infer_prior(f_m)

        # get distribution
        o = self.prior(f_m)
        q_mu, q_std = torch.split(o, self.z_nc, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])

        return distribution

    def two_paths(self, f_in):
        """two paths for the training"""
        f_m, f_c = f_in.chunk(2)
        distributions = []

        # get distribution
        o = self.posterior(f_c)
        p_mu, p_std = torch.split(o, self.z_nc, dim=1)
        distribution = self.one_path(f_m)
        distributions.append([p_mu, F.softplus(p_std), distribution[0][0], distribution[0][1]])

        return distributions

class TextualResEncoder(nn.Module):
    """
    ResNet Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ngf=32, z_nc=256, img_f=256, L=6, layers=5, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False, image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
        super(TextualResEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)
        self.word_attention = ImageTextAttention(idf=image_dim, cdf=text_dim, multi_peak=multi_peak, pooling=pool_attention)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 2), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # inference part
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf *mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'infer_prior' + str(i), block)

        # For textual, only change input and hidden dimension, z_nc is set when called.
        self.posterior = ResBlock(2*text_dim, 2*z_nc, ngf * mult * 2, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        self.prior =     ResBlock(2*text_dim, 2*z_nc, ngf * mult * 2, norm_layer, nonlinearity, 'none', use_spect, use_coord)

    def forward(self, img_m, sentence_embedding, word_embeddings, text_mask, image_mask, img_c=None):
        """
        :param img_m: image with mask regions I_m
        :param sentence_embedding: the sentence embedding of I
        :param word_embeddings: word embedding of I
        :param text_mask: mask of word sequence of word_embeddings
        :param image_mask: mask of Im and Ic, need to scale if apply to fm
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        :return text_feature: word and sentence features
        """

        if type(img_c) != type(None):
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        # infer part
        # during the training, we have two paths, during the testing, we only have one paths
        image_mask = task.scale_img(image_mask, size=[feature[-1].size(2), feature[-1].size(3)])
        if image_mask.size(1) == 3:
            image_mask = image_mask.chunk(3, dim=1)[0]

        if type(img_c) != type(None):
            # adapt to word embedding, compute weighted word embedding with fm separately
            f_m_g, f_m_rec = feature[-1].chunk(2)
            img_mask_g = image_mask
            img_mask_rec = 1 - img_mask_g
            weighted_word_embedding_rec = self.word_attention(
                        f_m_rec, word_embeddings, mask=text_mask, image_mask=img_mask_rec, inverse_attention=False)
            weighted_word_embedding_g = self.word_attention(
                        f_m_g, word_embeddings, mask=text_mask, image_mask=img_mask_g, inverse_attention=True)

            weighted_word_embedding =  torch.cat([weighted_word_embedding_g, weighted_word_embedding_rec])
            distribution, f_text = self.two_paths(out, sentence_embedding, weighted_word_embedding)

            return distribution, feature, f_text
        else:
            # adapt to word embedding, compute weighted word embedding with fm of one path
            f_m = feature[-1]
            weighted_word_embedding = self.word_attention(
                f_m, word_embeddings, mask=text_mask, image_mask=image_mask, inverse_attention=True)

            distribution, f_text = self.one_path(out, sentence_embedding, weighted_word_embedding)

            return distribution, feature, f_text

    def one_path(self, f_in, sentence_embedding, weighted_word_embedding):
        """one path for baseline training or testing"""
        # TOTEST: adapt to word embedding, compute distribution with word embedding.
        f_m = f_in
        distribution = []

        # infer state
        for i in range(self.L):
            infer_prior = getattr(self, 'infer_prior' + str(i))
            f_m = infer_prior(f_m)

        # get distribution
        # use sentence embedding here
        ix, iw = f_m.size(2), f_m.size(3)
        sentence_dim = sentence_embedding.size(1)
        sentence_embedding_replication = sentence_embedding.view(-1, sentence_dim, 1, 1).repeat(1, 1, ix, iw)
        f_text = torch.cat([sentence_embedding_replication, weighted_word_embedding], dim=1)

        o = self.prior(f_text)
        q_mu, q_std = torch.split(o, self.z_nc, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])

        return distribution, f_text

    def two_paths(self, f_in, sentence_embedding, weighted_word_embedding):
        """two paths for the training"""
        f_m, f_c = f_in.chunk(2)
        weighted_word_embedding_m, weighted_word_embedding_c = weighted_word_embedding.chunk(2)
        distributions = []

        # get distribution
        # use text embedding here
        ix, iw = f_c.size(2), f_c.size(3)
        sentence_dim = sentence_embedding.size(1)
        sentence_embedding_replication = sentence_embedding.view(-1, sentence_dim, 1, 1).repeat(1, 1, ix, iw)

        f_text_c = torch.cat([sentence_embedding_replication, weighted_word_embedding_c], dim=1)
        o = self.posterior(f_text_c)
        p_mu, p_std = torch.split(o, self.z_nc, dim=1)

        distribution, f_text_m = self.one_path(f_m, sentence_embedding, weighted_word_embedding_m)
        distributions.append([p_mu, F.softplus(p_std), distribution[0][0], distribution[0][1]])

        return distributions, torch.cat([f_text_m, f_text_c], dim=0)

class AttTextualResEncoder(nn.Module):
    """
    Attentive Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param image_dim: num of image feature maps
    :param text_dim: num of text embedding dimension
    :param multi_peak: use sigmoid in text attention if set to True
    """
    def __init__(self, input_nc=3, ngf=32, z_nc=256, img_f=256, L=6, layers=5, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False, image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
        super(AttTextualResEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)
        self.word_attention = ImageTextAttention(idf=image_dim, cdf=text_dim, multi_peak=multi_peak, pooling=pool_attention)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 2), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # inference part
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf *mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'infer_prior' + str(i), block)

        # For textual, only change input and hidden dimension, z_nc is set when called.
        self.posterior = ResBlock(ngf * mult + 2*text_dim, 2*z_nc, ngf * mult * 2, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        self.prior =     ResBlock(ngf * mult + 2*text_dim, 2*z_nc, ngf * mult * 2, norm_layer, nonlinearity, 'none', use_spect, use_coord)

    def forward(self, img_m, sentence_embedding, word_embeddings, text_mask, image_mask, img_c=None):
        """
        :param img_m: image with mask regions I_m
        :param sentence_embedding: the sentence embedding of I
        :param word_embeddings: word embedding of I
        :param text_mask: mask of word sequence of word_embeddings
        :param image_mask: mask of Im and Ic, need to scale if apply to fm
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        :return text_feature: word and sentence features
        """

        if type(img_c) != type(None):
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        # infer part
        # during the training, we have two paths, during the testing, we only have one paths
        image_mask = task.scale_img(image_mask, size=[feature[-1].size(2), feature[-1].size(3)])
        if image_mask.size(1) == 3:
            image_mask = image_mask.chunk(3, dim=1)[0]

        if type(img_c) != type(None):
            # adapt to word embedding, compute weighted word embedding with fm separately
            f_m_g, f_m_rec = feature[-1].chunk(2)
            img_mask_g = image_mask
            img_mask_rec = 1 - img_mask_g
            weighted_word_embedding_rec = self.word_attention(
                        f_m_rec, word_embeddings, mask=text_mask, image_mask=img_mask_rec, inverse_attention=False)
            weighted_word_embedding_g = self.word_attention(
                        f_m_g, word_embeddings, mask=text_mask, image_mask=img_mask_g, inverse_attention=True)

            weighted_word_embedding =  torch.cat([weighted_word_embedding_g, weighted_word_embedding_rec])
            distribution, f_text = self.two_paths(out, sentence_embedding, weighted_word_embedding)

            return distribution, feature, f_text
        else:
            # adapt to word embedding, compute weighted word embedding with fm of one path
            f_m = feature[-1]
            weighted_word_embedding = self.word_attention(
                f_m, word_embeddings, mask=text_mask, image_mask=image_mask, inverse_attention=True)

            distribution, f_m_text = self.one_path(out, sentence_embedding, weighted_word_embedding)
            f_text = torch.cat([f_m_text, weighted_word_embedding], dim=1)
            return distribution, feature, f_text

    def one_path(self, f_in, sentence_embedding, weighted_word_embedding):
        """one path for baseline training or testing"""
        f_m = f_in
        distribution = []

        # infer state
        for i in range(self.L):
            infer_prior = getattr(self, 'infer_prior' + str(i))
            f_m = infer_prior(f_m)

        # get distribution
        # use sentence embedding here
        ix, iw = f_m.size(2), f_m.size(3)
        sentence_dim = sentence_embedding.size(1)
        sentence_embedding_replication = sentence_embedding.view(-1, sentence_dim, 1, 1).repeat(1, 1, ix, iw)
        f_m_sent = torch.cat([f_m, sentence_embedding_replication], dim=1)
        f_m_text = torch.cat([f_m_sent, weighted_word_embedding], dim=1)

        o = self.prior(f_m_text)
        q_mu, q_std = torch.split(o, self.z_nc, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])

        return distribution, f_m_sent

    def two_paths(self, f_in, sentence_embedding, weighted_word_embedding):
        """two paths for the training"""
        f_m, f_c = f_in.chunk(2)
        weighted_word_embedding_m, weighted_word_embedding_c = weighted_word_embedding.chunk(2)
        distributions = []

        # get distribution
        # use text embedding here
        ix, iw = f_c.size(2), f_c.size(3)
        sentence_dim = sentence_embedding.size(1)
        sentence_embedding_replication = sentence_embedding.view(-1, sentence_dim, 1, 1).repeat(1, 1, ix, iw)
        f_c_sent = torch.cat([f_c, sentence_embedding_replication], dim=1)
        f_c_text = torch.cat([f_c_sent, weighted_word_embedding_c], dim=1)
        o = self.posterior(f_c_text)
        p_mu, p_std = torch.split(o, self.z_nc, dim=1)

        distribution, f_m_sent = self.one_path(f_m, sentence_embedding, weighted_word_embedding_m)
        distributions.append([p_mu, F.softplus(p_std), distribution[0][0], distribution[0][1]])

        f_m_text = torch.cat([f_m_sent, weighted_word_embedding_m], dim=1)
        # TODO: rm weighted_word_emb_c for consis generation
        f_c_text = torch.cat([f_m_sent, weighted_word_embedding_c], dim=1)
        return distributions, torch.cat([f_m_text, f_c_text], dim=0)

class ContrastResEncoder(nn.Module):
    """
    Contrastive Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param image_dim: num of image feature maps
    :param text_dim: num of text embedding dimension
    :param multi_peak: use sigmoid in text attention if set to True
    """
    def __init__(self, input_nc=3, ngf=32, z_nc=256, img_f=256, L=6, layers=5, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False, image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
        super(ContrastResEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)
        self.word_attention = ImageTextAttention(idf=image_dim, cdf=text_dim, multi_peak=multi_peak, pooling=pool_attention)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 2), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # inference part
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf *mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'infer_prior' + str(i), block)

        # For textual, only change input and hidden dimension, z_nc is set when called.
        self.posterior = ResBlock(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        self.prior     = ResBlock(text_dim + ngf * mult, 2*z_nc, 2*ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)

    def forward(self, img_m, sentence_embedding, word_embeddings, text_mask, image_mask, img_c=None):
        """
        :param img_m: image with mask regions I_m
        :param sentence_embedding: the sentence embedding of I
        :param word_embeddings: word embedding of I
        :param text_mask: mask of word sequence of word_embeddings
        :param image_mask: mask of Im and Ic, need to scale if apply to fm
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        :return text_feature: word and sentence features
        """

        if type(img_c) != type(None):
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        # infer part
        # during the training, we have two paths, during the testing, we only have one paths
        image_mask = task.scale_img(image_mask, size=[feature[-1].size(2), feature[-1].size(3)])
        if image_mask.size(1) == 3:
            image_mask = image_mask.chunk(3, dim=1)[0]

        if type(img_c) != type(None):
            # adapt to word embedding, compute weighted word embedding with fm separately
            f_m_g, f_m_rec = feature[-1].chunk(2)
            img_mask_g = image_mask
            img_mask_rec = 1 - img_mask_g
            weighted_word_embedding_rec = self.word_attention(
                        f_m_rec, word_embeddings, mask=text_mask, image_mask=img_mask_rec, inverse_attention=False)
            weighted_word_embedding_g = self.word_attention(
                        f_m_g, word_embeddings, mask=text_mask, image_mask=img_mask_g, inverse_attention=True)

            weighted_word_embedding =  torch.cat([weighted_word_embedding_g, weighted_word_embedding_rec])
            distribution, h_word = self.two_paths(out, weighted_word_embedding)

            return distribution, feature, h_word
        else:
            # adapt to word embedding, compute weighted word embedding with fm of one path
            f_m = feature[-1]
            weighted_word_embedding = self.word_attention(
                f_m, word_embeddings, mask=text_mask, image_mask=image_mask, inverse_attention=True)

            distribution, h_word = self.one_path(weighted_word_embedding, f_m)
            return distribution, feature, h_word

    def one_path(self, weighted_word_embedding, v_h):
        """one path for baseline training or testing"""
        h_word = weighted_word_embedding
        distribution = []

        # infer state
        for i in range(self.L):
            infer_prior = getattr(self, 'infer_prior' + str(i))
            h_word = infer_prior(h_word)

        # get distribution
        o = self.prior(torch.cat([h_word,v_h], dim=1))
        q_mu, q_std = torch.split(o, self.z_nc, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])

        return distribution, h_word

    def two_paths(self, f_in, weighted_word_embedding):
        """two paths for the training"""
        # use text embedding here
        f_m, f_c = f_in.chunk(2)
        weighted_word_embedding_m, weighted_word_embedding_c = weighted_word_embedding.chunk(2)

        h_word_c = weighted_word_embedding_c
        # infer state
        for i in range(self.L):
            infer_prior = getattr(self, 'infer_prior' + str(i))
            h_word_c = infer_prior(h_word_c)

        # get distribution
        distributions = []
        o = self.posterior(f_c)
        p_mu, p_std = torch.split(o, self.z_nc, dim=1)
        distribution, h_word = self.one_path(weighted_word_embedding_m, f_m)
        distributions.append([p_mu, F.softplus(p_std), distribution[0][0], distribution[0][1]])

        return distributions, torch.cat([h_word, h_word_c], dim=0)


class PosAttTextualResEncoder(nn.Module):
    """
    Positive Attentive Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param image_dim: num of image feature maps
    :param text_dim: num of text embedding dimension
    :param multi_peak: use sigmoid in text attention if set to True
    """
    def __init__(self, input_nc=3, ngf=32, z_nc=256, img_f=256, L=6, layers=5, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False, image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
        super(PosAttTextualResEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)
        self.word_attention = ImageTextAttention(idf=image_dim, cdf=text_dim, multi_peak=multi_peak, pooling=pool_attention)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 2), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # inference part
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf *mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'infer_prior' + str(i), block)

        # For textual, only change input and hidden dimension, z_nc is set when called.
        self.posterior = ResBlock(ngf * mult + 2*text_dim, 2*z_nc, ngf * mult * 2, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        self.prior =     ResBlock(ngf * mult + 2*text_dim, 2*z_nc, ngf * mult * 2, norm_layer, nonlinearity, 'none', use_spect, use_coord)

    def forward(self, img_m, sentence_embedding, word_embeddings, text_mask, image_mask, img_c=None):
        """
        :param img_m: image with mask regions I_m
        :param sentence_embedding: the sentence embedding of I
        :param word_embeddings: word embedding of I
        :param text_mask: mask of word sequence of word_embeddings
        :param image_mask: mask of Im and Ic, need to scale if apply to fm
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        :return text_feature: word and sentence features
        """

        if type(img_c) != type(None):
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        # infer part
        # during the training, we have two paths, during the testing, we only have one paths
        image_mask = task.scale_img(image_mask, size=[feature[-1].size(2), feature[-1].size(3)])
        if image_mask.size(1) == 3:
            image_mask = image_mask.chunk(3, dim=1)[0]

        if type(img_c) != type(None):
            # adapt to word embedding, compute weighted word embedding with fm separately
            f_m_g, f_m_rec = feature[-1].chunk(2)
            img_mask_g = image_mask
            img_mask_rec = 1 - img_mask_g
            weighted_word_embedding_rec = self.word_attention(
                        f_m_rec, word_embeddings, mask=text_mask, image_mask=img_mask_rec, inverse_attention=True)
            weighted_word_embedding_g = self.word_attention(
                        f_m_g, word_embeddings, mask=text_mask, image_mask=img_mask_g, inverse_attention=False)

            weighted_word_embedding =  torch.cat([weighted_word_embedding_g, weighted_word_embedding_rec])
            distribution, f_text = self.two_paths(out, sentence_embedding, weighted_word_embedding)

            return distribution, feature, f_text
        else:
            # adapt to word embedding, compute weighted word embedding with fm of one path
            f_m = feature[-1]
            weighted_word_embedding = self.word_attention(
                f_m, word_embeddings, mask=text_mask, image_mask=image_mask, inverse_attention=False)

            distribution, f_m_text = self.one_path(out, sentence_embedding, weighted_word_embedding)
            f_text = torch.cat([f_m_text, weighted_word_embedding], dim=1)
            return distribution, feature, f_text

    def one_path(self, f_in, sentence_embedding, weighted_word_embedding):
        """one path for baseline training or testing"""
        # TOTEST: adapt to word embedding, compute distribution with word embedding.
        f_m = f_in
        distribution = []

        # infer state
        for i in range(self.L):
            infer_prior = getattr(self, 'infer_prior' + str(i))
            f_m = infer_prior(f_m)

        # get distribution
        # use sentence embedding here
        ix, iw = f_m.size(2), f_m.size(3)
        sentence_dim = sentence_embedding.size(1)
        sentence_embedding_replication = sentence_embedding.view(-1, sentence_dim, 1, 1).repeat(1, 1, ix, iw)
        f_m_sent = torch.cat([f_m, sentence_embedding_replication], dim=1)
        f_m_text = torch.cat([f_m_sent, weighted_word_embedding], dim=1)

        o = self.prior(f_m_text)
        q_mu, q_std = torch.split(o, self.z_nc, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])

        return distribution, f_m_sent

    def two_paths(self, f_in, sentence_embedding, weighted_word_embedding):
        """two paths for the training"""
        f_m, f_c = f_in.chunk(2)
        weighted_word_embedding_m, weighted_word_embedding_c = weighted_word_embedding.chunk(2)
        distributions = []

        # get distribution
        # use text embedding here
        ix, iw = f_c.size(2), f_c.size(3)
        sentence_dim = sentence_embedding.size(1)
        sentence_embedding_replication = sentence_embedding.view(-1, sentence_dim, 1, 1).repeat(1, 1, ix, iw)
        f_c_sent = torch.cat([f_c, sentence_embedding_replication], dim=1)
        f_c_text = torch.cat([f_c_sent, weighted_word_embedding_c], dim=1)
        o = self.posterior(f_c_text)
        p_mu, p_std = torch.split(o, self.z_nc, dim=1)

        distribution, f_m_sent = self.one_path(f_m, sentence_embedding, weighted_word_embedding_m)
        distributions.append([p_mu, F.softplus(p_std), distribution[0][0], distribution[0][1]])

        f_m_text = torch.cat([f_m_sent, weighted_word_embedding_m], dim=1)
        # TODO: rm weighted_word_emb_c for consis generation
        f_c_text = torch.cat([f_m_sent, weighted_word_embedding_c], dim=1)
        return distributions, torch.cat([f_m_text, f_c_text], dim=0)


class WordAttnEncoder(nn.Module):
    """
    WordAttn Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param image_dim: num of image feature maps
    :param text_dim: num of text embedding dimension
    :param multi_peak: use sigmoid in text attention if set to True
    """
    def __init__(self, input_nc=3, ngf=32, z_nc=256, img_f=256, L=6, layers=5, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False, image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
        super(WordAttnEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)
        self.word_attention = ImageTextAttention(idf=image_dim, cdf=text_dim, multi_peak=multi_peak, pooling=pool_attention)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 2), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # inference part
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf *mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'infer_prior' + str(i), block)

        # For textual, only change input and hidden dimension, z_nc is set when called.
        self.posterior = ResBlock(ngf * mult + text_dim, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        self.prior =     ResBlock(ngf * mult + text_dim, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)

    def forward(self, img_m, word_embeddings, text_mask, image_mask, img_c=None):
        """
        :param img_m: image with mask regions I_m
        :param word_embeddings: word embedding of I
        :param text_mask: mask of word sequence of word_embeddings
        :param image_mask: mask of Im and Ic, need to scale if apply to fm
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        :return text_feature: word and sentence features
        """

        if type(img_c) != type(None):
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        # infer part
        # during the training, we have two paths, during the testing, we only have one paths
        image_mask = task.scale_img(image_mask, size=[feature[-1].size(2), feature[-1].size(3)])
        if image_mask.size(1) == 3:
            image_mask = image_mask.chunk(3, dim=1)[0]

        if type(img_c) != type(None):
            # adapt to word embedding, compute weighted word embedding with fm separately
            f_m_g, f_m_rec = feature[-1].chunk(2)
            img_mask_g = image_mask
            img_mask_rec = 1 - img_mask_g
            weighted_word_embedding_rec = self.word_attention(
                        f_m_rec, word_embeddings, mask=text_mask, image_mask=img_mask_rec, inverse_attention=False)
            weighted_word_embedding_g = self.word_attention(
                        f_m_g, word_embeddings, mask=text_mask, image_mask=img_mask_g, inverse_attention=True)

            weighted_word_embedding =  torch.cat([weighted_word_embedding_g, weighted_word_embedding_rec])
            distribution, f_text = self.two_paths(out, weighted_word_embedding)

            return distribution, feature, f_text
        else:
            # adapt to word embedding, compute weighted word embedding with fm of one path
            f_m = feature[-1]
            weighted_word_embedding = self.word_attention(
                f_m, word_embeddings, mask=text_mask, image_mask=image_mask, inverse_attention=True)

            distribution = self.one_path(out, weighted_word_embedding)
            f_text = weighted_word_embedding
            return distribution, feature, f_text

    def one_path(self, f_in, weighted_word_embedding):
        """one path for baseline training or testing"""
        # TOTEST: adapt to word embedding, compute distribution with word embedding.
        f_m = f_in
        distribution = []

        # infer state
        for i in range(self.L):
            infer_prior = getattr(self, 'infer_prior' + str(i))
            f_m = infer_prior(f_m)

        # get distribution
        # use sentence embedding here
        f_m_text = torch.cat([f_m, weighted_word_embedding], dim=1)

        o = self.prior(f_m_text)
        q_mu, q_std = torch.split(o, self.z_nc, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])

        return distribution

    def two_paths(self, f_in, weighted_word_embedding):
        """two paths for the training"""
        f_m, f_c = f_in.chunk(2)
        weighted_word_embedding_m, weighted_word_embedding_c = weighted_word_embedding.chunk(2)
        distributions = []

        # get distribution
        # use text embedding here
        f_c_text = torch.cat([f_c, weighted_word_embedding_c], dim=1)
        o = self.posterior(f_c_text)
        p_mu, p_std = torch.split(o, self.z_nc, dim=1)

        distribution = self.one_path(f_m, weighted_word_embedding_m)
        distributions.append([p_mu, F.softplus(p_std), distribution[0][0], distribution[0][1]])

        f_m_text = weighted_word_embedding_m
        f_c_text = weighted_word_embedding_c
        return distributions, torch.cat([f_m_text, f_c_text], dim=0)


class ConstraintResEncoder(nn.Module):
    """
    Constraint Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param image_dim: num of image feature maps
    :param text_dim: num of text embedding dimension
    :param multi_peak: use sigmoid in text attention if set to True
    """
    def __init__(self, input_nc=3, ngf=32, z_nc=256, img_f=256, L=6, layers=5, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False, image_dim=256, text_dim=256, multi_peak=True, pool_attention='max'):
        super(ConstraintResEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)
        self.word_attention = ImageTextAttention(idf=image_dim, cdf=text_dim, multi_peak=multi_peak, pooling=pool_attention)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 2), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # inference part
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf *mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'infer_prior' + str(i), block)

        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf *mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'infer_prior_word' + str(i), block)

        # For textual, only change input and hidden dimension, z_nc is set when called.
        self.posterior = ResBlock(ngf * mult + 2*text_dim, 2*z_nc, ngf * mult * 2, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        self.prior =     ResBlock(ngf * mult + 2*text_dim, 2*z_nc, ngf * mult * 2, norm_layer, nonlinearity, 'none', use_spect, use_coord)

    def forward(self, img_m, sentence_embedding, word_embeddings, text_mask, image_mask, img_c=None):
        """
        :param img_m: image with mask regions I_m
        :param sentence_embedding: the sentence embedding of I
        :param word_embeddings: word embedding of I
        :param text_mask: mask of word sequence of word_embeddings
        :param image_mask: mask of Im and Ic, need to scale if apply to fm
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        :return text_feature: word and sentence features
        """

        if type(img_c) != type(None):
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        # infer part
        # during the training, we have two paths, during the testing, we only have one paths
        image_mask = task.scale_img(image_mask, size=[feature[-1].size(2), feature[-1].size(3)])
        if image_mask.size(1) == 3:
            image_mask = image_mask.chunk(3, dim=1)[0]

        if type(img_c) != type(None):
            # During training
            f_m_g, f_m_rec = feature[-1].chunk(2)
            img_mask_g = image_mask
            img_mask_rec = 1 - img_mask_g
            weighted_word_embedding_rec = self.word_attention(
                        f_m_rec, word_embeddings, mask=text_mask, image_mask=img_mask_rec, inverse_attention=False)
            weighted_word_embedding_g = self.word_attention(
                        f_m_g, word_embeddings, mask=text_mask, image_mask=img_mask_g, inverse_attention=True)

            weighted_word_embedding =  torch.cat([weighted_word_embedding_g, weighted_word_embedding_rec])
            distribution, f_text, dual_word_embedding = self.two_paths(out, sentence_embedding, weighted_word_embedding)

            return distribution, feature, f_text, dual_word_embedding
        else:
            # During test
            f_m = feature[-1]
            weighted_word_embedding = self.word_attention(
                f_m, word_embeddings, mask=text_mask, image_mask=image_mask, inverse_attention=True)

            distribution, f_m_text, infered_word_embedding = self.one_path(out, sentence_embedding, weighted_word_embedding)
            f_text = torch.cat([f_m_text, weighted_word_embedding], dim=1)
            return distribution, feature, f_text

    def one_path(self, f_in, sentence_embedding, weighted_word_embedding):
        """one path for baseline training or testing"""
        # TOTEST: adapt to word embedding, compute distribution with word embedding.
        f_m = f_in
        distribution = []

        # infer state
        for i in range(self.L):
            infer_prior = getattr(self, 'infer_prior' + str(i))
            f_m = infer_prior(f_m)

        # infer state
        for i in range(self.L):
            infer_prior_word = getattr(self, 'infer_prior_word' + str(i))
            infered_word_embedding = infer_prior_word(weighted_word_embedding)

        # get distribution
        # use sentence embedding here
        ix, iw = f_m.size(2), f_m.size(3)
        sentence_dim = sentence_embedding.size(1)
        sentence_embedding_replication = sentence_embedding.view(-1, sentence_dim, 1, 1).repeat(1, 1, ix, iw)
        f_m_sent = torch.cat([f_m, sentence_embedding_replication], dim=1)
        f_m_text = torch.cat([f_m_sent, infered_word_embedding], dim=1)

        o = self.prior(f_m_text)
        q_mu, q_std = torch.split(o, self.z_nc, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])

        return distribution, f_m_sent, infered_word_embedding

    def two_paths(self, f_in, sentence_embedding, weighted_word_embedding):
        """two paths for the training"""
        f_m, f_c = f_in.chunk(2)
        weighted_word_embedding_m, weighted_word_embedding_c = weighted_word_embedding.chunk(2)
        distributions = []

        # get distribution
        # use text embedding here
        ix, iw = f_c.size(2), f_c.size(3)
        sentence_dim = sentence_embedding.size(1)
        sentence_embedding_replication = sentence_embedding.view(-1, sentence_dim, 1, 1).repeat(1, 1, ix, iw)
        f_c_sent = torch.cat([f_c, sentence_embedding_replication], dim=1)
        f_c_text = torch.cat([f_c_sent, weighted_word_embedding_c], dim=1)
        o = self.posterior(f_c_text)
        p_mu, p_std = torch.split(o, self.z_nc, dim=1)

        distribution, f_m_sent, infered_word_embedding = self.one_path(f_m, sentence_embedding, weighted_word_embedding_m)
        distributions.append([p_mu, F.softplus(p_std), distribution[0][0], distribution[0][1]])
        dual_word_embedding = torch.cat([infered_word_embedding, weighted_word_embedding_c], dim=0)

        f_m_text = torch.cat([f_m_sent, infered_word_embedding], dim=1)
        # TODO: evaluate wether to replace infered to weighted_c
        f_c_text = torch.cat([f_m_sent, infered_word_embedding], dim=1)

        return distributions, torch.cat([f_m_text, f_c_text], dim=0), dual_word_embedding

class ResGenerator(nn.Module):
    """
    ResNet Generator Network
    :param output_nc: number of channels in output
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param output_scale: Different output scales
    """
    def __init__(self, output_nc=3, ngf=64, z_nc=128, img_f=1024, L=1, layers=6, norm='batch', activation='ReLU',
                 output_scale=1, use_spect=True, use_coord=False, use_attn=True):
        super(ResGenerator, self).__init__()

        self.layers = layers
        self.L = L
        self.output_scale = output_scale
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # latent z to feature
        mult = min(2 ** (layers-1), img_f // ngf)
        # input -> hidden
        self.generator = ResBlock(z_nc, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)

        # transform
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'generator' + str(i), block)

        # decoder part
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers - i - 1), img_f // ngf)
            if i > layers - output_scale:
                # upconv = ResBlock(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                upconv = ResBlockDecoder(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            else:
                # upconv = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                upconv = ResBlockDecoder(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), upconv)
            # output part
            if i > layers - output_scale - 1:
                outconv = Output(ngf * mult, output_nc, 3, None, nonlinearity, use_spect, use_coord)
                setattr(self, 'out' + str(i), outconv)
            # short+long term attention part
            if i == 1 and use_attn:
                attn = Auto_Attn(ngf*mult, None)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, z, f_m=None, f_e=None, mask=None):
        """
        ResNet Generator Network
        :param z: latent vector
        :param f_m: feature of valid regions for conditional VAG-GAN
        :param f_e: previous encoder feature for short+long term attention layer
        :return results: different scale generation outputs
        """

        f = self.generator(z)
        for i in range(self.L):
             generator = getattr(self, 'generator' + str(i)) # dimension not change
             f = generator(f)

        # the features come from mask regions and valid regions, we directly add them together
        out = f_m + f
        results= []
        attn = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            if i == 1 and self.use_attn:
                # auto attention
                model = getattr(self, 'attn' + str(i))
                out, attn = model(out, f_e, mask)
            if i > self.layers - self.output_scale - 1:
                model = getattr(self, 'out' + str(i))
                output = model(out)
                results.append(output)
                out = torch.cat([out, output], dim=1)

        return results, attn

class ContrastResGenerator(nn.Module):
    """
    Contrast ResNet Generator Network
    :param output_nc: number of channels in output
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param output_scale: Different output scales
    """
    def __init__(self, output_nc=3, ngf=64, z_nc=128, img_f=1024, L=1, layers=6, norm='batch', activation='ReLU',
                 output_scale=1, use_spect=True, use_coord=False, use_attn=True):
        super(ContrastResGenerator, self).__init__()

        self.layers = layers
        self.L = L
        self.output_scale = output_scale
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # latent z to feature
        mult = min(2 ** (layers-1), img_f // ngf)
        # input -> hidden
        self.generator = ResBlock(z_nc, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)

        # transform
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'generator' + str(i), block)

        # decoder part
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers - i - 1), img_f // ngf)
            if i > layers - output_scale:
                # upconv = ResBlock(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                upconv = ResBlockDecoder(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            else:
                # upconv = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                upconv = ResBlockDecoder(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), upconv)
            # output part
            if i > layers - output_scale - 1:
                outconv = Output(ngf * mult, output_nc, 3, None, nonlinearity, use_spect, use_coord)
                setattr(self, 'out' + str(i), outconv)
            # short+long term attention part
            if i == 1 and use_attn:
                attn = Auto_Attn(ngf*mult, None)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, z, f_m=None, f_e=None, mask=None):
        """
        ResNet Generator Network
        :param z: latent vector
        :param f_m: feature of valid regions for conditional VAG-GAN
        :param f_e: previous encoder feature for short+long term attention layer
        :return results: different scale generation outputs
        """

        f = self.generator(z)
        for i in range(self.L):
             generator = getattr(self, 'generator' + str(i)) # dimension not change
             f = generator(f)

        # the features come from mask regions and valid regions, we directly add them together
        # out = f_m + f
        out = f
        results= []
        attn = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            if i == 1 and self.use_attn:
                # auto attention
                model = getattr(self, 'attn' + str(i))
                out, attn = model(out, f_e, mask)
            if i > self.layers - self.output_scale - 1:
                model = getattr(self, 'out' + str(i))
                output = model(out)
                results.append(output)
                out = torch.cat([out, output], dim=1)

        return results, attn

class HiddenResGenerator(nn.Module):
    """
    ResNet Generator Network
    :param output_nc: number of channels in output
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param output_scale: Different output scales
    """
    def __init__(self, output_nc=3, f_text_dim=384, ngf=64, z_nc=128, img_f=256, L=1, layers=6, norm='batch', activation='ReLU',
                 output_scale=1, use_spect=True, use_coord=False, use_attn=True):
        super(HiddenResGenerator, self).__init__()

        self.layers = layers
        self.L = L
        self.output_scale = output_scale
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # latent z to feature
        mult = min(2 ** (layers-1), img_f // ngf)
        # input -> hidden
        self.generator = ResBlock(z_nc, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
        self.f_transformer = ResBlock(f_text_dim, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)

        # transform
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'generator' + str(i), block)

        # decoder part
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers - i - 1), img_f // ngf)
            if i > layers - output_scale:
                # upconv = ResBlock(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                upconv = ResBlockDecoder(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            else:
                # upconv = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                upconv = ResBlockDecoder(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), upconv)
            # output part
            if i > layers - output_scale - 1:
                outconv = Output(ngf * mult, output_nc, 3, None, nonlinearity, use_spect, use_coord)
                setattr(self, 'out' + str(i), outconv)
            # short+long term attention part
            if i == 1 and use_attn:
                attn = Auto_Attn(ngf*mult, None)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, z, f_text=None, f_e=None, mask=None):
        """
        ResNet Generator Network
        :param z: latent vector
        :param f_m: feature of valid regions for conditional VAG-GAN
        :param f_e: previous encoder feature for short+long term attention layer
        :return results: different scale generation outputs
        """
        f = self.generator(z)
        for i in range(self.L):
             generator = getattr(self, 'generator' + str(i)) # dimension not change
             f = generator(f)
        f_text_trans = self.f_transformer(f_text)
        # the features come from mask regions and valid regions, we directly add them together
        out = f_text_trans + f
        results= []
        attn = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            if i == 1 and self.use_attn:
                # auto attention
                model = getattr(self, 'attn' + str(i))
                out, attn = model(out, f_e, mask)
            if i > self.layers - self.output_scale - 1:
                model = getattr(self, 'out' + str(i))
                output = model(out)
                results.append(output)
                out = torch.cat([out, output], dim=1)

        return results, attn

class TextualResGenerator(nn.Module):
    """
    Textual ResNet Generator Network.
    This fucking code is hard to maintenance, just list a trip of shit.
    :param output_nc: number of channels in output
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param output_scale: Different output scales
    """
    def __init__(self, output_nc=3, f_text_dim=384, ngf=64, z_nc=128, img_f=256, L=1, layers=6, norm='batch', activation='ReLU',
                 output_scale=1, use_spect=True, use_coord=False, use_attn=True):
        super(TextualResGenerator, self).__init__()

        self.layers = layers
        self.L = L
        self.output_scale = output_scale
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # latent z to feature
        mult = min(2 ** (layers-1), img_f // ngf)
        # input -> hidden
        self.generator = ResBlock(z_nc, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
        self.f_transformer = ResBlock(f_text_dim, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)

        # transform
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'generator' + str(i), block)

        # decoder part
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers - i - 1), img_f // ngf)
            if i > layers - output_scale:
                # upconv = ResBlock(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                upconv = ResBlockDecoder(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            else:
                # upconv = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                upconv = ResBlockDecoder(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), upconv)
            # output part
            if i > layers - output_scale - 1:
                outconv = Output(ngf * mult, output_nc, 3, None, nonlinearity, use_spect, use_coord)
                setattr(self, 'out' + str(i), outconv)
            # short+long term attention part
            if i == 1 and use_attn:
                attn = Auto_Attn(ngf*mult, None)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, z, f_text=None, f_e=None, mask=None):
        """
        ResNet Generator Network
        :param z: latent vector
        :param f_m: feature of valid regions for conditional VAG-GAN
        :param f_e: previous encoder feature for short+long term attention layer
        :return results: different scale generation outputs
        """
        f = self.generator(z)
        for i in range(self.L):
             generator = getattr(self, 'generator' + str(i)) # dimension not change
             f = generator(f)
        f_text_trans = self.f_transformer(f_text)
        # the features come from mask regions and valid regions, we directly add them together
        out = f_text_trans + f
        results= []
        attn = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            if i == 1 and self.use_attn:
                # auto attention
                model = getattr(self, 'attn' + str(i))
                out, attn = model(out, f_e, mask)
            if i > self.layers - self.output_scale - 1:
                model = getattr(self, 'out' + str(i))
                output = model(out)
                results.append(output)
                out = torch.cat([out, output], dim=1)

        return results, attn

class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=True):
        super(ResDiscriminator, self).__init__()

        self.layers = layers
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf,norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn(ndf * mult_prev, norm_layer)
                setattr(self, 'attn' + str(i), attn)
            block = ResBlock(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        self.block1 = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        self.conv = SpectralNorm(nn.Conv2d(ndf * mult, 1, 3))

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.block1(out)
        out = self.conv(self.nonlinearity(out))
        return out



class SNPatchDiscriminator(nn.Module):
    """
    SN Patch Discriminator Network for Local 70*70 fake/real
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param img_f: the largest channel for the model
    :param layers: down sample layers
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectral normalization or not
    :param use_coord: use CoordConv or nor
    """
    def __init__(self, input_nc=4, ndf=64, img_f=256, layers=6, activation='LeakyReLU',
                 use_spect=True, use_coord=False):
        super(SNPatchDiscriminator, self).__init__()

        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        kwargs = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}
        sequence = [
            coord_conv(input_nc, ndf, use_spect, use_coord, **kwargs),
            nonlinearity,
        ]

        mult = 1
        for i in range(1, layers):
            mult_prev = mult
            mult = min(2 ** i, img_f // ndf)
            sequence +=[
                    coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
                    nonlinearity,
                ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()

        self.n_steps = 25
        self.rnn_type = 'LSTM'

        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True, enforce_sorted=False)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb

from torchvision import models
import torch.utils.model_zoo as model_zoo

class CNN_ENCODER(nn.Module):
    def __init__(self, nef, pre_train=False):
        super(CNN_ENCODER, self).__init__()
        self.nef = nef  # define a uniform ranker

        model = models.inception_v3()
        if pre_train:
            url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
            model.load_state_dict(model_zoo.load_url(url))
            for param in model.parameters():
                param.requires_grad = False
            print('Load pretrained model from ', url)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code

