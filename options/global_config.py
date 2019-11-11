import os
import yaml

class DataConfig(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml, Loader=yaml.FullLoader)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


DEFAULT_CONFIG = {
    'MAX_TEXT_LENGTH' : 128,

    'VOCAB' : "./datasets/captions_vocab_bird.pickle",
    'LANGUAGE_ENCODER' : "./datasets/text_encoder_bird.pth",

    'CATE_IMAGE_TRAIN' : "./datasets/CUB_200_2011/cate_image_train.json",
    'IMAGE_CATE_TRAIN' : "./datasets/CUB_200_2011/image_cate.json",

    'TRAIN_CAPTION' : "./datasets/CUB_200_2011/caption.json",
    'VAL_CAPTION' : "./datasets/CUB_200_2011/caption.json",
    'TEST_CAPTION' : "./datasets/CUB_200_2011/caption.json",
}