#  python evaluation.py --ground_truth_path eval_hidden_ad-l2-g_maxpool_reproduce.flist --save_path results/hidden_ad-l2-g_maxpool_reproduce_centermask/ --batch_test 30 
import numpy as np
import argparse
from PIL import Image
import math
from dataloader.image_folder import make_dataset
import os
import glob
import shutil
from tqdm import tqdm
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.color import rgb2gray

parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
parser.add_argument('--ground_truth_path', type = str, default='dataset/image_painting/imagenet_test.txt',
                    help = 'path to original particular solutions')
parser.add_argument('--save_path', type = str, default='imagenet/center',
                    help='path to save the test dataset')
parser.add_argument('--batch_test', type=int, default=32,
                    help='how many images to load for each test, just like batch')
args = parser.parse_args()

def compare_mae(img_true, img_test):
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

def compute_errors(ground_truth, pre):
    pre = np.array(pre).astype(np.float32) /255.
    ground_truth = np.array(ground_truth).astype(np.float32)/255.

    pre = rgb2gray(pre)
    ground_truth = rgb2gray(ground_truth)

    PSNR = compare_psnr(ground_truth, pre,  data_range=1)
    SSIM = compare_ssim(ground_truth, pre, multichannel=True, data_range=pre.max()-pre.min(), sigma=1.5)
    l1 = compare_mae(ground_truth, pre)
    # TV
    gx = pre - np.roll(pre, -1, axis=1)
    gy = pre - np.roll(pre, -1, axis=0)
    grad_norm2 = gx ** 2 + gy ** 2
    TV = np.mean(np.sqrt(grad_norm2))

    return l1, PSNR, TV, SSIM


if __name__ == "__main__":

    ground_truth_paths, ground_truth_size = make_dataset(args.ground_truth_path)

    iters = int(ground_truth_size/args.batch_test)

    l1_loss = np.zeros(iters, np.float32)
    PSNR = np.zeros(iters, np.float32)
    TV = np.zeros(iters, np.float32)
    SSIM = np.zeros(iters, np.float32)

    for i in tqdm(range(0, iters)):
        # calculate one batch of test data
        l1_batch = np.zeros(args.batch_test, np.float32)
        PSNR_batch = np.zeros(args.batch_test, np.float32)
        TV_batch = np.zeros(args.batch_test, np.float32)
        SSIM_batch = np.zeros(args.batch_test, np.float32)

        num = i*args.batch_test

        for j in range(args.batch_test):
            # calculate one data in a batch
            index = num+j
            ground_truth_image = Image.open(ground_truth_paths[index]).resize([256,256]).convert('RGB')
            l1_sample = 1000
            PSNR_sample = 0
            SSIM_sample = 0
            TV_sample = 1000
            name = ground_truth_paths[index].split('/')[-1].split("truth")[0]+"*"
            pre_paths = sorted(glob.glob(os.path.join(args.save_path, name)))
            num_image_files = len(pre_paths)

            for k in range(num_image_files-1):
                # calculate each image of random results
                index2 = k

                pre_image = Image.open(pre_paths[index2]).resize([256,256]).convert('RGB')
                l1_temp, PSNR_temp, TV_temp, SSIM_temp = compute_errors(ground_truth_image, pre_image)
                # select the best results for the errors estimation
                if l1_temp - PSNR_temp + TV_temp - SSIM_temp < \
                          l1_sample - PSNR_sample + TV_sample - SSIM_temp:
                    l1_sample, PSNR_sample, TV_sample, SSIM_sample = \
                        l1_temp, PSNR_temp, TV_temp, SSIM_temp
                    best_index = index2

            # shutil.copy(pre_paths[best_index], '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/image_painting/results/ours/imagenet/center_copy/')
            # print(pre_paths[best_index])
            # print(l1_sample, PSNR_sample, TV_sample)

            l1_batch[j], PSNR_batch[j], TV_batch[j], SSIM_batch[j] = \
                l1_sample, PSNR_sample, TV_sample, SSIM_sample

        l1_loss[i] = np.mean(l1_batch)
        PSNR[i] = np.mean(PSNR_batch)
        TV[i] = np.mean(TV_batch)
        SSIM[i] = np.mean(SSIM_batch)

    print('{:>10},{:>10},{:>10},{:>10}'.format('L1_LOSS', 'PSNR', 'TV','SSIM'))
    print('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(l1_loss.mean(), PSNR.mean(), TV.mean(), SSIM.mean()))
