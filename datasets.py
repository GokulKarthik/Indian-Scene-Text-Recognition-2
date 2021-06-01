import os
import random

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

random.seed(0)


class IndianSceneTextDataset(Dataset):

    def __init__(self, data_dir, tokenize_method, max_tokens, glyphs=None, num_train_samples=None):
        self.data_dir = data_dir
        self.tokenize_method = tokenize_method
        self.max_tokens = max_tokens
        self.image_fns = os.listdir(data_dir)
        print('Dataset Size:', len(self.image_fns))
        self.glyphs = glyphs
        self.image_fns = [fn for fn in self.image_fns if self.contains_glyph(fn)]
        print('Dataset Size (with glyphs):', len(self.image_fns))
        if num_train_samples:
            random.shuffle(self.image_fns)
            self.image_fns = self.image_fns[:num_train_samples*1000]
            print('Dataset Size (used):', len(self.image_fns))

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.data_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = self.image_transform(image)
        text = image_fn.split("_")[0]
        if self.tokenize_method == 'unicode':
            text_bounded = text[:self.max_tokens]
        elif self.tokenize_method == 'original':
            original_tokens = []
            unicode_tokens = list(text)
            for unicode_token in unicode_tokens:
                if unicode_token in self.glyphs:
                    original_tokens[-1] += unicode_token
                else:
                    original_tokens.append(unicode_token)
            if len(original_tokens) > self.max_tokens:
                text_bounded = ''.join(original_tokens[:self.max_tokens])
            else:
                text_bounded = ''.join(original_tokens)
        return image, text_bounded, image_fn

    def contains_glyph(self, fn):
        text = fn.split(".")[0]
        token_cnt = 0
        for c in text:
            if c in self.glyphs and token_cnt <= self.max_tokens:
                return True
            token_cnt += 1
        return False

    @staticmethod
    def image_transform(image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)