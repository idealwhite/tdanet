import numpy as np
import argparse
from PIL import Image
import math
from dataloader.image_folder import make_dataset
import os
import glob
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
parser.add_argument('--gt_path', type = str, default='dataset/image_painting/imagenet_test.txt',
                    help = 'path to original particular solutions')
parser.add_argument('--save_path', type = str, default='imagenet/center',
                    help='path to save the test dataset')
parser.add_argument('--batch_test', type=int, default=1000,
                    help='how many images to load for each test, just like batch')
args = parser.parse_args()


def compute_errors(gt, pre):

    # l1 loss
    l1 = np.mean(np.abs(gt-pre))

    # PSNR
    mse = np.mean((gt - pre) ** 2)
    if mse == 0:
        PSNR = 100
    else:
        PSNR = 20 * math.log10(255.0 / math.sqrt(mse))

    # TV
    gx = pre - np.roll(pre, -1, axis=1)
    gy = pre - np.roll(pre, -1, axis=0)
    grad_norm2 = gx ** 2 + gy ** 2
    TV = np.mean(np.sqrt(grad_norm2))

    return l1, PSNR, TV


if __name__ == "__main__":

    gt_paths, gt_size = make_dataset(args.gt_path)

    iters = int(gt_size/args.batch_test)

    l1_loss = np.zeros(iters, np.float32)
    PSNR = np.zeros(iters, np.float32)
    TV = np.zeros(iters, np.float32)

    for i in tqdm(range(0, iters)):
        l1_batch = np.zeros(args.batch_test, np.float32)
        PSNR_batch = np.zeros(args.batch_test, np.float32)
        TV_batch = np.zeros(args.batch_test, np.float32)

        num = i*args.batch_test

        for j in range(args.batch_test):
            index = num+j
            gt_image = Image.open(gt_paths[index]).resize([256,256]).convert('RGB')
            gt_numpy = np.array(gt_image).astype(np.float32)
            l1_sample = 1000
            PSNR_sample = 0
            TV_sample = 1000
            name = gt_paths[index].split('/')[-1].split("truth")[0]+"*"
            pre_paths = sorted(glob.glob(os.path.join(args.save_path, name)))
            num_image_files = len(pre_paths)

            for k in range(num_image_files-1):
                index2 = k
                try:
                    pre_image = Image.open(pre_paths[index2]).resize([256,256]).convert('RGB')
                    pre_numpy = np.array(pre_image).astype(np.float32)
                    l1_temp, PSNR_temp, TV_temp = compute_errors(gt_numpy, pre_numpy)
                    # select the best results for the errors estimation
                    if l1_temp - PSNR_temp + TV_temp < l1_sample - PSNR_sample + TV_sample:
                        l1_sample, PSNR_sample, TV_sample = l1_temp, PSNR_temp, TV_temp
                        best_index = index2
                except:
                    print(pre_paths[index2])
            # shutil.copy(pre_paths[best_index], '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/image_painting/results/ours/imagenet/center_copy/')
            # print(pre_paths[best_index])
            # print(l1_sample, PSNR_sample, TV_sample)

            l1_batch[j], PSNR_batch[j], TV_batch[j] = l1_sample, PSNR_sample, TV_sample

        l1_loss[i] = np.mean(l1_batch)
        PSNR[i] = np.mean(PSNR_batch)
        TV[i] = np.mean(TV_batch)

    print('{:>10},{:>10},{:>10}'.format('L1_LOSS', 'PSNR', 'TV'))
    print('{:10.4f},{:10.4f},{:10.4f}'.format(l1_loss.mean(), PSNR.mean(), TV.mean()))
