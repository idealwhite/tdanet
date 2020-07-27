# python test.py --name wordattninpainting  --img_file datasets/CUB_200_2011/valid.flist --results_dir results/wordattninpainting  --how_many 200 --mask_file datasets/CUB_200_2011/test_mask.flist --mask_type 3 --no_shuffle --gpu_ids 0 --nsampling 1
from options import test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
import torch
import os

if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    model.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)

    for i, data in enumerate(dataset):
        with torch.no_grad():
            model.set_input(data)
            model.test()
    os.system('ls '+opt.results_dir+'/*_truth.png > eval_'+opt.results_dir.split('/')[-1]+'.flist')
