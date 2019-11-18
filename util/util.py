import numpy as np
import os
import imageio
import torch

from nltk.tokenize import word_tokenize

def idx_to_caption(ixtoword, caption, length):
    """ Turn idx to word"""
    return ' '.join([ixtoword[caption[i]] for i in range(length)])

def _caption_to_idx(wordtoix, caption, max_length):
    '''Transform single text caption to idx and length tensor'''
    caption_token = word_tokenize(caption.lower())

    caption_idx = []
    for token in caption_token:
        t = token.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            caption_idx.append(wordtoix[t])

    length = len(caption_idx)
    if length <= max_length:
        caption_idx = caption_idx + [0] * (max_length - len(caption_idx))
    else:
        caption_idx = caption_idx[:max_length]

    return caption_idx, length

def vectorize_captions_idx_batch(batch_padded_captions_idx, batch_length, language_encoder):
    '''Transform batch_padded_captions_idx to sentence embedding'''
    batch_size = len(batch_length)

    with torch.no_grad():
        hidden = language_encoder.init_hidden(batch_size)
        device = hidden[0].device
        word_embs, sent_emb = language_encoder(batch_padded_captions_idx.to(device), \
                                       batch_length.to(device), hidden)
    return word_embs, sent_emb

def lengths_to_mask(lengths, max_length, device=None):
    '''transform digital lengths to tensor mask.'''
    masks = torch.ones(len(lengths), max_length)
    for i, length in enumerate(lengths):
        masks[i,:length] = 0
    masks = masks.bool()
    return masks if device is None else masks.to(device)


def PSNR(a, b):
    '''compute PSNR for a and b image'''
    mse = np.mean((a - b) ** 2) + 1e-8

    return 20 * np.log10(255.0 / np.sqrt(mse))

def tensor_image_scale(tensor):
    '''scale the value in tensor as image'''
    return (tensor + 1) / 2.0 * 255.0

# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)


# conver a tensor into a numpy array
def tensor2array(value_tensor):
    if value_tensor.dim() == 3:
        numpy = value_tensor.view(-1).cpu().float().numpy()
    else:
        numpy = value_tensor[0].view(-1).cpu().float().numpy()
    return numpy


def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
