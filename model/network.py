from .base_function import *
from .external_function import SpectralNorm
import torch.nn.functional as F


##############################################################################################################
# Network function
##############################################################################################################
def define_e(input_nc=3, ngf=64, z_nc=512, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[]):

    net = ResEncoder(input_nc, ngf, z_nc, img_f, L, layers, norm, activation, use_spect, use_coord)

    return init_net(net, init_type, activation, gpu_ids)


def define_g(output_nc=3, ngf=64, z_nc=512, img_f=512, L=1, layers=5, norm='instance', activation='ReLU', output_scale=1,
             use_spect=True, use_coord=False, use_attn=True, init_type='orthogonal', gpu_ids=[]):

    net = ResGenerator(output_nc, ngf, z_nc, img_f, L, layers, norm, activation, output_scale, use_spect, use_coord, use_attn)

    return init_net(net, init_type, activation, gpu_ids)


def define_d(input_nc=3, ndf=64, img_f=512, layers=6, norm='none', activation='LeakyReLU', use_spect=True, use_coord=False,
             use_attn=True,  model_type='ResDis', init_type='orthogonal', gpu_ids=[]):

    if model_type == 'ResDis':
        net = ResDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn)
    elif model_type == 'PatchDis':
        net = PatchDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn)

    return init_net(net, init_type, activation, gpu_ids)


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


class PatchDiscriminator(nn.Module):
    """
    Patch Discriminator Network for Local 70*70 fake/real
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param img_f: the largest channel for the model
    :param layers: down sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectral normalization or not
    :param use_coord: use CoordConv or nor
    :param use_attn: use short+long attention or not
    """
    def __init__(self, input_nc=3, ndf=64, img_f=512, layers=3, norm='batch', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=False):
        super(PatchDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
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

        mult_prev = mult
        mult = min(2 ** i, img_f // ndf)
        kwargs = {'kernel_size': 4, 'stride': 1, 'padding': 1, 'bias': False}
        sequence += [
            coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
            nonlinearity,
            coord_conv(ndf * mult, 1, use_spect, use_coord, **kwargs),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    """
    Global attention takes a matrix and a query metrix.
    Based on each query vector q, it computes a parameterized convex combination of the matrix
    based.
    H_1 H_2 H_3 ... H_n
      q   q   q       q
        |  |   |       |
          \ |   |      /
                  .....
              \   |  /
                      a
    Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.
    References:
    https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
    http://www.aclweb.org/anthology/D15-1166
    """
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn

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