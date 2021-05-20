from gui.ui_window import Ui_Form
from gui.ui_draw import *
from PIL import Image, ImageQt
import random, io, os, math
import numpy as np
import torch
import torchvision.transforms as transforms
from util import task, util
from dataloader.image_folder import make_dataset
from model import create_model
from util.visualizer import Visualizer
from options.global_config import TextConfig
import json
import pickle

def compute_errors(ground_truth, pre):

    # l1 loss
    l1 = np.mean(np.abs(ground_truth-pre))

    # PSNR
    mse = np.mean((ground_truth - pre) ** 2)
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

class ui_model(QtWidgets.QWidget, Ui_Form):
    shape = 'line'
    CurrentWidth = 1

    def __init__(self, opt):
        super(ui_model, self).__init__()
        self.setupUi(self)
        self.opt = opt
        # self.show_image = None
        self.show_result_flag = False
        self.opt.loadSize = [256, 256]
        self.visualizer = Visualizer(opt)
        self.model_name = ['bird', 'coco']
        self.config_name = ['config.bird.yml', 'config.coco.yml']
        self.img_root = './datasets/'
        self.img_files = ['CUB_200_2011/test.flist', 'coco/valid.flist']
        self.graphicsView_2.setMaximumSize(self.opt.loadSize[0]+30, self.opt.loadSize[1]+30)

        # show logo
        # self.show_logo()

        # original mask
        self.new_painter()

        # selcet model
        self.comboBox.activated.connect(self.load_model)

        # load image
        self.pushButton.clicked.connect(self.load_image)

        # random image
        self.pushButton_2.clicked.connect(self.random_image)

        # save result
        self.pushButton_4.clicked.connect(self.save_result)

        # draw/erasure the mask
        self.radioButton.toggled.connect(lambda: self.draw_mask('line'))
        self.radioButton_2.toggled.connect(lambda: self.draw_mask('rectangle'))
        self.spinBox.valueChanged.connect(self.change_thickness)
        # erase
        self.pushButton_5.clicked.connect(self.clear_mask)

        # fill image, image process
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.pushButton_3.clicked.connect(self.fill_mask)

        # show the result
        self.pushButton_6.clicked.connect(self.show_result)

        self.firstly=True

    def showImage(self, fname):
        """Show the masked images"""
        value = self.comboBox.currentIndex()
        img = Image.open(fname).convert('RGB')
        self.img_original = img.resize(self.opt.loadSize)
        if value < 4:
            self.img = self.img_original
        else:
            self.img = self.img_original
            sub_img = Image.fromarray(np.uint8(255*np.ones((128, 128, 3))))
            mask = Image.fromarray(np.uint8(255*np.ones((128, 128))))
            self.img.paste(sub_img, box=(64, 64), mask=mask)
        self.show_image = ImageQt.ImageQt(self.img)
        self.new_painter(self.show_image)

    def show_result(self):
        """Show the results and original image"""
        if self.show_result_flag:
            self.show_result_flag = False
            new_pil_image = Image.fromarray(util.tensor2im(self.img_out.detach()))
            new_qt_image = ImageQt.ImageQt(new_pil_image)
        else:
            self.show_result_flag = True
            new_qt_image = ImageQt.ImageQt(self.img_original)
        self.graphicsView_2.scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(new_qt_image))
        self.graphicsView_2.scene.addItem(item)
        self.graphicsView_2.setScene(self.graphicsView_2.scene)

    def show_logo(self):
        """Show the logo of NTU and BTC"""
        return

    def load_model(self):
        """Load different kind models for different datasets and mask types"""
        value = self.comboBox.currentIndex()
        if value == 0:
            raise NotImplementedError("Please choose a model")
        else:
            # define the model type and dataset type
            index = value-1
            self.opt.name = self.model_name[index]
            self.opt.text_config = self.config_name[index]
            self.opt.img_file = self.img_root + self.img_files[index % len(self.img_files)]

            text_config = TextConfig(self.opt.text_config)
            self.max_length = text_config.MAX_TEXT_LENGTH
            x = pickle.load(open(text_config.VOCAB, 'rb'))
            self.ixtoword = x[2]
            self.wordtoix = x[3]

            if index == 0:
                self.image_paths, self.image_size = make_dataset(self.opt.img_file)

                # load caption file
                with open(text_config.CAPTION, 'r') as f:
                    self.captions = json.load(f)

            self.model = create_model(self.opt)


    def load_image(self):
        """Load the image"""
        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'select the image', self.opt.img_file, 'Image files(*.jpg *.png)')
        self.showImage(self.fname)
        if self.opt.name == 'bird':
            self.random_caption()

    def random_image(self):
        """Random load the test image"""
        value = self.comboBox.currentIndex()
        if value == 0:
            return
        # read random mask
        if self.opt.mask_file != "none":
            mask_paths, mask_size = make_dataset(self.opt.mask_file)
            item = random.randint(0, mask_size - 1)
            self.mname = mask_paths[item]

        item = random.randint(0, self.image_size-1)
        self.fname = self.image_paths[item]
        if self.firstly:
            #self.fname = './datasets/CUB_200_2011\\images/042.Vermilion_Flycatcher/Vermilion_Flycatcher_0045_42219.jpg'
            self.fname = './datasets/CUB_200_2011\\images/169.Magnolia_Warbler/Magnolia_Warbler_0063_166121.jpg'

            self.firstly = False
        self.showImage(self.fname)
        print(self.fname)

        img_name = os.path.basename(self.fname)
        caption = sorted(self.captions[img_name], key=lambda x:len(x))[-1]
        self.textEdit.setText(caption)

    def random_caption(self):
        img_name = os.path.basename(self.fname)
        # caption = sorted(self.captions[img_name], key=lambda x:len(x))[0]
        # self.textEdit.setText(caption)

    def save_result(self):
        """Save the results to the disk"""
        util.mkdir(self.opt.results_dir)
        img_name = self.fname.split('/')[-1]
        data_name = self.opt.img_file.split('/')[-1].split('.')[0]

        # save the original image
        original_name = '%s_%s_%s' % ('original', data_name, img_name)
        original_path = os.path.join(self.opt.results_dir, original_name)
        img_original = util.tensor2im(self.img_truth)
        util.save_image(img_original, original_path)

        # save the mask
        mask_name = '%s_%s_%d_%s' % ('mask', data_name, self.PaintPanel.iteration, img_name)
        mask_path = os.path.join(self.opt.results_dir, mask_name)
        img_mask = util.tensor2im(self.img_m)
        util.save_image(img_mask, mask_path)

        # save the results
        result_name = '%s_%s_%d_%s' % ('result', data_name, self.PaintPanel.iteration, img_name)
        result_path = os.path.join(self.opt.results_dir, result_name)
        img_result = util.tensor2im(self.img_out)
        util.save_image(img_result, result_path)


    def new_painter(self, image=None):
        """Build a painter to load and process the image"""
        # painter
        self.PaintPanel = painter(self, image)
        self.PaintPanel.close()
        self.stackedWidget.insertWidget(0, self.PaintPanel)
        self.stackedWidget.setCurrentWidget(self.PaintPanel)

    def change_thickness(self, num):
        """Change the width of the painter"""
        self.CurrentWidth = num
        self.PaintPanel.CurrentWidth = num

    def draw_mask(self, maskStype):
        """Draw the mask"""
        self.shape = maskStype
        self.PaintPanel.shape = maskStype

    def clear_mask(self):
        """Clear the mask"""
        self.showImage(self.fname)
        if self.PaintPanel.Brush:
            self.PaintPanel.Brush = False
        else:
            self.PaintPanel.Brush = True

    def set_input(self):
        """Set the input for the network"""
        # get the test mask from painter
        text = self.textEdit.toPlainText()
        text_idx, text_len = util._caption_to_idx(self.model.wordtoix, text,  len(text))
        self.text_idx = torch.Tensor([text_idx]).long()
        self.text_len = torch.Tensor([text_len]).long()

        self.PaintPanel.saveDraw()
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QBuffer.ReadWrite)
        self.PaintPanel.map.save(buffer, 'PNG')
        pil_im = Image.open(io.BytesIO(buffer.data()))

        # transform the image to the tensor
        img = self.transform(self.img)
        value = self.comboBox.currentIndex()

        if value < 4:
            mask = torch.autograd.Variable(self.transform(pil_im)).unsqueeze(0)
            # mask from the random mask
            # mask = Image.open(self.mname)
            # mask = torch.autograd.Variable(self.transform(mask)).unsqueeze(0)
            mask = (mask < 1).float()
        else:
            mask = task.center_mask(img).unsqueeze(0)

        if len(self.opt.gpu_ids) > 0:
            img = img.unsqueeze(0).cuda(self.opt.gpu_ids[0])
            mask = mask.cuda(self.opt.gpu_ids[0])

        # get I_m and I_c for image with mask and complement regions for training
        mask = mask
        self.img_truth = img * 2 - 1
        self.img_m = mask * self.img_truth
        self.img_c = (1 - mask) * self.img_truth

        return self.img_m, self.img_c, self.img_truth, mask, self.text_idx, self.text_len

    def fill_mask(self):
        """Forward to get the generation results"""
        img_m, img_c, img_truth, mask, text_idx, text_len = self.set_input()
        if self.comboBox.currentIndex() == 0:
            return
        if text_len < 1:
            self.textEdit.setText('Input some words about this bird or the bird you want.')
            return
        print(self.textEdit.toPlainText())
        if self.PaintPanel.iteration < 100:
            print(self.PaintPanel.iteration)
            with torch.no_grad():
                # encoder process
                word_embeddings, sentence_embedding = util.vectorize_captions_idx_batch(
                                                text_idx, text_len, self.model.text_encoder)
                img_mask = torch.ones_like(img_m)
                img_mask[img_m == 0.] = 0.
                distributions, f, f_text = self.model.net_E(img_m, sentence_embedding, word_embeddings, None, img_mask, img_c)

                variation_factor = 1. if self.checkBox.isChecked() else 0.
                q_distribution = torch.distributions.Normal(distributions[-1][0], distributions[-1][1] * variation_factor)

                z = q_distribution.sample()

                # decoder process
                scale_mask = task.scale_pyramid(mask, 4)
                self.img_g, _ = self.model.net_G(z, f_text, f_e=f[2], mask=scale_mask[0].chunk(3, dim=1)[0])

                self.img_out = (1 - mask) * self.img_g[-1].detach() + mask * img_m

                # get score
                l1, PSNR, TV = compute_errors(util.tensor2im(self.img_truth), util.tensor2im(self.img_out.detach()))

                self.label_6.setText(str(PSNR))

                self.PaintPanel.iteration += 1

        self.show_result_flag = True
        # import ipdb; ipdb.set_trace()
        self.show_result()
