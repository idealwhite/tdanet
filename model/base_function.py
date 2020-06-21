import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from .external_function import SpectralNorm

######################################################################################
# base function for network structure
######################################################################################


def init_weights(net, init_type='normal', gain=0.02):
    """Get different initial method for the network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_norm_layer(norm_type='batch'):
    """Get the normalization layer for the networks"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.iter_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f M' % (num_params/1e6))


def init_net(net, init_type='normal', activation='relu', gpu_ids=[]):
    """print the network structure and initial the network"""
    print_network(net)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module


def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, **kwargs):
    """use coord convolution layer to add position information"""
    if use_coord:
        return CoordConv(input_nc, output_nc, with_r, use_spect, **kwargs)
    else:
        return spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

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
    attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
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


######################################################################################
# Network basic function
######################################################################################
class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    CoordConv operation
    """
    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret


class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 sample_type='none', use_spect=False, use_coord=False):
        super(ResBlock, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc
        self.sample = True
        if sample_type == 'none':
            self.sample = False
        elif sample_type == 'up':
            output_nc = output_nc * 4
            self.pool = nn.PixelShuffle(upscale_factor=2)
        elif sample_type == 'down':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc), nonlinearity, self.conv2,)

        self.shortcut = nn.Sequential(self.bypass,)

    def forward(self, x):
        if self.sample:
            out = self.pool(self.model(x)) + self.pool(self.shortcut(x))
        else:
            out = self.model(x) + self.shortcut(x)

        return out


class ResBlockEncoderOptimized(nn.Module):
    """
    Define an Encoder block for the first layer of the discriminator and representation network
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(), use_spect=False, use_coord=False):
        super(ResBlockEncoderOptimized, self).__init__()

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(output_nc, output_nc, use_spect, use_coord, **kwargs)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(self.conv1, nonlinearity, self.conv2, nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.model = nn.Sequential(self.conv1, norm_layer(output_nc), nonlinearity, self.conv2, nn.AvgPool2d(kernel_size=2, stride=2))

        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), self.bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)

        return out


class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc

        self.conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        self.conv2 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        self.bypass = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc), nonlinearity, self.conv2,)

        self.shortcut = nn.Sequential(self.bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)

        return out


class Output(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, kernel_size = 3, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Output, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1, nn.Tanh())
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out


class Auto_Attn(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(Auto_Attn, self).__init__()
        self.input_nc = input_nc

        self.query_conv = nn.Conv2d(input_nc, input_nc // 4, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.model = ResBlock(int(input_nc*2), input_nc, input_nc, norm_layer=norm_layer, use_spect=True)

    def forward(self, x, pre=None, mask=None):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H)  # B X (N)X C
        proj_key = proj_query  # B X C x (N)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x

        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            context_flow = torch.bmm(pre.view(B, -1, W*H), attention.permute(0, 2, 1)).view(B, -1, W, H)
            context_flow = self.alpha * (1-mask) * context_flow + (mask) * pre
            out = self.model(torch.cat([out, context_flow], dim=1))

        return out, attention


class ImageTextAttention(nn.Module):
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

    :param idf: image dimension
    :param cdf: text dimension
    :param multi_peak: use sigmoid when computing word attention
    :param pooling: pooling layer type on weightedConext
    """
    def __init__(self, idf, cdf, multi_peak=False, pooling='max'):
        super(ImageTextAttention, self).__init__()
        self.conv_image = conv1x1(idf, cdf)
        self.sm = nn.Softmax()
        self.multi_peak = multi_peak
        self.sigmoid = nn.Sigmoid()
        self.pooling = pooling
        if self.pooling == 'max':
            self.pooling_layer = nn.AdaptiveMaxPool2d(1)
        elif self.pooling == 'avg':
            self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        else:
            self.pooling = False

    def forward_softmax(self, image, text, mask=None, image_mask=None, inverse_attention=False):
        """
            input: batch x idf x ih x iw (image_L=ihxiw)
            context: batch x cdf x text_L
        """
        ih, iw = image.size(2), image.size(3)
        image_L = ih * iw
        batch_size, text_L = text.size(0), text.size(2)

        # --> batch x image_L x idf
        image = self.conv_image(image)
        image_flat = image.view(batch_size, -1, image_L)
        image_flat_T = torch.transpose(image_flat, 1, 2).contiguous()

        # Get attention
        # (batch x image_L x idf)(batch x idf x text_L)
        # -->batch x image_L x text_L
        attn = torch.bmm(image_flat_T, text)
        if inverse_attention:
            attn *= -1

        if image_mask is not None:
            # in img_mask, 0 is masked, so here we inverse the mask value
            image_mask = (1-image_mask).bool()#torch.logical_not(image_mask.bool())
            image_mask = image_mask.view(-1, image_L, 1).repeat(1, 1, text_L)

            attn.data.masked_fill_(image_mask.data, -float('inf'))

        # --> batch*image_L x text_L
        attn = attn.view(batch_size*image_L, text_L)
        if mask is not None:
            # batch_size x text_L --> batch_size*image_L x text_L
            mask = mask.repeat(image_L, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))

        attn = self.sm(attn)  # Eq. (2)
        attn.data.masked_fill_(attn != attn, 0)
        # --> batch x image_L x text_L
        attn = attn.view(batch_size, image_L, text_L)
        # --> batch x text_L x image_L
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x text_L)(batch x text_L x image_L)
        # --> batch x idf x image_L
        weightedContext = torch.bmm(text, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        # torch.save(attn.detach(), 'attention_map.pt')

        return weightedContext

    def forward_sigmoid(self, image, text, mask=None, image_mask=None, inverse_attention=False):
        """
            input: batch x idf x ih x iw (image_L=ihxiw)
            context: batch x cdf x text_L
        """
        ih, iw = image.size(2), image.size(3)
        image_L = ih * iw
        batch_size, text_L = text.size(0), text.size(2)

        # --> batch x image_L x idf
        image = self.conv_image(image)
        image_flat = image.view(batch_size, -1, image_L)
        image_flat_T = torch.transpose(image_flat, 1, 2).contiguous()

        # Get attention
        # (batch x image_L x idf)(batch x idf x text_L)
        # -->batch x image_L x text_L
        attn = torch.bmm(image_flat_T, text)

        # Apply mask
        if image_mask is not None:
            # in img_mask, 0 is masked, so here we inverse the mask value
            image_mask = (1-image_mask).bool()#torch.logical_not(image_mask.bool())
            image_mask = image_mask.view(-1, image_L, 1).repeat(1, 1, text_L)

            attn.data.masked_fill_(image_mask.data, -float('inf'))

        # --> batch*image_L x text_L
        attn = attn.view(batch_size*image_L, text_L)
        if mask is not None:
            # batch_size x text_L --> batch_size*image_L x text_L
            mask = mask.repeat(image_L, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))

        attn = self.sigmoid(attn)  # Eq. (2)
        if inverse_attention:
            attn = 1 - attn

        attn.data.masked_fill_(attn != attn, 0)
        # --> batch x image_L x text_L
        attn = attn.view(batch_size, image_L, text_L)
        # --> batch x text_L x image_L
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x text_L)(batch x text_L x image_L)
        # --> batch x idf x image_L
        weightedContext = torch.bmm(text, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext

    def forward(self, image, text, mask=None, image_mask=None, inverse_attention=False):
        if self.multi_peak:
            weightedContext = self.forward_sigmoid(image, text, mask, image_mask, inverse_attention)
        else:
            weightedContext = self.forward_softmax(image, text, mask, image_mask, inverse_attention)
        if self.pooling is not False:
            ih, iw = weightedContext.size(2), weightedContext.size(3)
            weightedContext = self.pooling_layer(weightedContext)
            weightedContext = weightedContext.repeat(1, 1, ih, iw)

        return weightedContext

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

    def forward(self, input, context, mask=None, inverse_attention=False):
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
        if inverse_attention:
            attn *= -1
        attn = attn.view(batch_size * queryL, sourceL)
        if mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = mask.repeat(queryL, 1)
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

        return weightedContext

# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps))


def sent_loss(cnn_code, rnn_code, labels, eps=1e-8, smooth_gama3=10.0):

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * smooth_gama3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze(0)

    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0 + loss1


def words_loss(img_features, words_emb, labels, cap_lens, batch_size,
                            smooth_gamma1=5.0, smooth_gamma2=5.0, smooth_gamma3=10.0):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """

    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):

        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, smooth_gamma1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(smooth_gamma2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)

    similarities = similarities * smooth_gamma3

    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0 + loss1, att_maps

class GANHingeLoss(nn.Module):
    def __init__(self):
        super(GANHingeLoss, self).__init__()
        self.activation = nn.ReLU(inplace=True)

    def __call__(self, pos, neg):
        hinge_pos = torch.mean(self.activation(1-pos))
        hinge_neg = torch.mean(self.activation(1+neg))
        d_loss = .5 * hinge_pos + .5 * hinge_neg
        g_loss = -torch.mean(neg)

        return d_loss, g_loss