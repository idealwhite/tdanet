import torch
import random
import json
import pickle
import os
import numpy as np
import imageio
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
from util import task, util
from options.global_config import TextConfig

class CreateDataset(data.Dataset):
    def __init__(self, opt, debug=False):
        self.opt = opt
        self.debug = debug
        self.img_paths, self.img_size = make_dataset(opt.img_file)
        # provides random file for training and testing
        if opt.mask_file != 'none':
            if not opt.mask_file.endswith('.json'):
                self.mask_paths, self.mask_size = make_dataset(opt.mask_file)
            else:
                with open(opt.mask_file, 'r') as f:
                    self.image_bbox = json.load(f)

        self.transform = get_transform(opt)

        ## ========Abnout text stuff===============
        text_config = TextConfig(opt.text_config)
        self.max_length = text_config.MAX_TEXT_LENGTH
        if 'coco' in text_config.CAPTION.lower():
            self.num_captions = 5
        elif 'place' in text_config.CAPTION.lower():
            self.num_captions = 1
        else:
            self.num_captions = 10

        # load caption file
        with open(text_config.CAPTION, 'r') as f:
            self.captions = json.load(f)

        x = pickle.load(open(text_config.VOCAB, 'rb'))
        self.ixtoword = x[2]
        self.wordtoix = x[3]

        self.epoch = 0 # Used for iter on captions.

    def __getitem__(self, index):
        # load image
        index = self.epoch*self.img_size+index

        img, img_path = self.load_img(index)
        # load mask
        mask = self.load_mask(img, index, img_path)
        assert sum(img.shape) == sum(mask.shape), (img.shape, mask.shape)
        caption_idx, caption_len, caption, img_name= self._load_text_idx(index)
        return {'img': img, 'img_path': img_path, 'mask': mask, \
                'caption_idx' : torch.Tensor(caption_idx).long(), 'caption_len':caption_len,\
                'caption_text': caption, 'image_path': img_name}

    def __len__(self):
        return self.img_size

    def name(self):
        return "inpainting dataset"

    def load_img(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.img_paths[index % self.img_size]
        img_pil = Image.open(img_path).convert('RGB')
        img = self.transform(img_pil)
        img_pil.close()
        return img, img_path

    def _load_text_idx(self, image_index):
        img_name = self.img_paths[image_index % self.img_size]
        caption_index_of_image = image_index // self.img_size  % self.num_captions
        img_name = os.path.basename(img_name)
        captions = self.captions[img_name]
        caption = captions[caption_index_of_image] if type(captions) == list else captions
        caption_idx, caption_len = util._caption_to_idx(self.wordtoix, caption, self.max_length)

        return caption_idx, caption_len, caption, img_name

    def load_mask(self, img, index, img_path):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.opt.mask_type) - 1)
        mask_type = self.opt.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return task.center_mask(img)

        # random regular mask
        if mask_type == 1:
            return task.random_regular_mask(img)

        # random irregular mask
        if mask_type == 2:
            return task.random_irregular_mask(img)

        if mask_type == 3:
            # file masks, e.g. CUB object mask
            mask_index = index
            mask_pil = Image.open(self.mask_paths[mask_index]).convert('RGB')

            mask_transform = get_transform_mask(self.opt)

            mask = (mask_transform(mask_pil) == 0).float()
            mask_pil.close()
            return mask

        if mask_type == 4:
            # coco json file object mask
            if os.path.basename(img_path) not in self.image_bbox:
                return task.random_regular_mask(img)

            img_original = np.asarray(Image.open(img_path).convert('RGB'))

            # create a mask matrix same as img_original
            mask = np.zeros_like(img_original)
            bboxes = self.image_bbox[os.path.basename(img_path)]

            # choose max area box
            choosen_box = 0,0,0,0
            max_area = 0
            for x1,x2,y1,y2 in bboxes:
                area = (x2-x1) * (y2-y1)
                if area > max_area:
                    max_area = area
                    choosen_box = x1,x2,y1,y2
            x1, x2, y1, y2 = choosen_box
            mask[x1:x2, y1:y2] = 1

            # apply same transform as img to the mask
            mask_pil = Image.fromarray(mask)

            mask_transform = get_transform_mask(self.opt)

            mask = (mask_transform(mask_pil) == 0).float()

            mask_pil.close()

            return mask


def dataloader(opt):
    datasets = CreateDataset(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=not opt.no_shuffle, num_workers=int(opt.nThreads), pin_memory=True)

    return dataset

def get_transform_mask(opt):
    """Basic process to transform PIL image to torch tensor"""
    transform_list = []
    osize = [opt.loadSize[0], opt.loadSize[1]]
    fsize = [opt.fineSize[0], opt.fineSize[1]]
    if opt.isTrain:
        if opt.resize_or_crop == 'resize_and_crop':
            transform_list.append(transforms.Resize(osize))
            transform_list.append(transforms.RandomCrop(fsize))
        elif opt.resize_or_crop == 'crop':
            transform_list.append(transforms.RandomCrop(fsize))
        if not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if not opt.no_rotation:
            transform_list.append(transforms.RandomRotation(3))
    else:
        transform_list.append(transforms.Resize(fsize))

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

def get_transform(opt):
    """Basic process to transform PIL image to torch tensor"""
    transform_list = []
    osize = [opt.loadSize[0], opt.loadSize[1]]
    fsize = [opt.fineSize[0], opt.fineSize[1]]
    if opt.isTrain:
        if opt.resize_or_crop == 'resize_and_crop':
            transform_list.append(transforms.Resize(osize))
            transform_list.append(transforms.RandomCrop(fsize))
        elif opt.resize_or_crop == 'crop':
            transform_list.append(transforms.RandomCrop(fsize))
        if not opt.no_augment:
            transform_list.append(transforms.ColorJitter(0.0, 0.0, 0.0, 0.0))
        if not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if not opt.no_rotation:
            transform_list.append(transforms.RandomRotation(3))
    else:
        transform_list.append(transforms.Resize(fsize))

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)
