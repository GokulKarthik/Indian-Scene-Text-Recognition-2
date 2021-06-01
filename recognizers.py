import random

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18

from utils import correct_prediction, levenshtein_distance

random.seed(0)


class CRNN1(pl.LightningModule):

    def __init__(self, model_config, characters, consonants, glyphs, character_split, training_config=None, purpose=None):

        super().__init__()
        self.num_chars = len(characters)
        self.mode = model_config['mode']
        self.is_bidirectional = model_config['is_bidirectional']
        self.rnn_hidden_size = model_config['rnn_hidden_size']
        self.use_out_scale = model_config['use_out_scale']
        self.use_softmax = model_config['use_softmax']
        if self.mode == 'multi-label-shuffle':
            print(characters[:10])
            characters_copy = characters.copy()[1:]
            random.shuffle(characters_copy)
            characters = ['-'] + characters_copy
            print(characters[:10])
        self.characters = characters
        self.consonants = consonants
        self.glyphs = glyphs
        self.character_split = character_split
        self.idx2char = {k: v for k, v in enumerate(self.characters)}
        self.char2idx = {v: k for k, v in self.idx2char.items()}
        if self.mode.startswith('multi-label'):
            self.idx2consonant = {k: v for k, v in enumerate(self.consonants)}
            self.consonant2idx = {v: k for k, v in self.idx2consonant.items()}
            self.idx2glyph = {k: v for k, v in enumerate(self.glyphs)}
            self.glyph2idx = {v: k for k, v in self.idx2glyph.items()}

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=False)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
        self.dp3 = nn.Dropout(p=0.5)

        if training_config:
            resnet = resnet18(pretrained=training_config['is_pretrained'])
            self.weight_decay = training_config['weight_decay']
            self.remove_consonant_blanks = training_config['remove_consonant_blanks']
            self.remove_glyph_blanks = training_config['remove_glyph_blanks']
            self.weight_character_loss = training_config['weight_character_loss']
            self.weight_consonant_loss = training_config['weight_consonant_loss']
            self.weight_glyph_loss = training_config['weight_glyph_loss']
            self.lr = training_config['lr']
            self.lr_step_size = training_config['lr_step_size']
            self.lr_gamma = training_config['lr_gamma']
            self.grad_norm_threshold = training_config['grad_norm_threshold']
        else:
            resnet = resnet18(pretrained=False)

        self.purpose = purpose

        # CNN Part 1
        resnet_modules = list(resnet.children())[:-3]
        self.cnn_p1 = nn.Sequential(*resnet_modules)

        # CNN Part 2
        self.cnn_p2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.linear1 = nn.Linear(1024, 256)  # 256*4 = 1024

        # RNN
        self.rnn1 = nn.GRU(input_size=self.rnn_hidden_size,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=2,
                           bidirectional=self.is_bidirectional,
                           batch_first=True)
        self.rnn2 = nn.GRU(input_size=self.rnn_hidden_size,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=1,
                           bidirectional=self.is_bidirectional,
                           batch_first=True)

        # Output
        rnn_op_nodes = self.rnn_hidden_size
        if self.is_bidirectional:
            rnn_op_nodes *= 2

        if self.mode in ['original', 'unicode']:
            self.out = nn.Linear(rnn_op_nodes, self.num_chars)
        elif self.mode.startswith('multi-label'):
            self.out_consonant = nn.Linear(rnn_op_nodes, len(self.consonants))
            self.out_glyph = nn.Linear(rnn_op_nodes, len(self.glyphs))
            self.out_blank = nn.Linear(rnn_op_nodes, 1)

            out_scale = torch.rand((len(self.consonants)*len(self.glyphs))+1) * 0.1
            #out_scale = torch.rand((1))
            self.out_scale = nn.Parameter(out_scale, requires_grad=True)

            out_glyph_scale = torch.rand(len(self.glyphs))
            self.out_glyph_scale = nn.Parameter(out_glyph_scale, requires_grad=True)

            out_consonant_scale = torch.rand((len(self.consonants)))
            self.out_consonant_scale = nn.Parameter(out_consonant_scale, requires_grad=True)

            out_blank_scale = torch.rand((1))
            self.out_blank_scale = nn.Parameter(out_blank_scale, requires_grad=True)
        else:
            raise ValueError('Invalid model')

    def forward_step(self, x):

        batch_size = x.size(0)

        x = self.cnn_p1(x)  # [-1, 256, 4, 13]

        x = self.cnn_p2(x)  # [-1, 256, 4, 13]
        #print(x.size())
        #input()

        T = x.size(3)
        x = self.dp1(x)
        x = x.permute(0, 3, 1, 2)  # [-1, 13, 256, 4]
        x = x.view(batch_size, T, -1)  # [-1, 13, 1024]

        x = self.linear1(x)  # [-1, 13, 256]
        x = self.dp2(x)

        x, h_n = self.rnn1(x)  # [-1, 13, 2*256(or)256])
        if self.is_bidirectional:
            x = x[:, :, :self.rnn_hidden_size] + x[:, :, self.rnn_hidden_size:]  # [-1, 13, 256])

        x, h_n = self.rnn2(x)  # [-1, 13, 512(or)256]
        x = self.dp3(x)

        if self.mode in ['original', 'unicode']:
            x = self.out(x)  # [-1, 13, num_chars]
            out = x.permute(1, 0, 2)  # [13, -1, num_chars]
            consonant_out = None
            glyph_out = None
        elif self.mode.startswith('multi-label'):
            num_consonants = len(self.consonants)
            num_glyphs = len(self.glyphs)

            blank_out = self.out_blank(x)  # [-1, 13, 1]
            # blank_out = blank_out * self.out_blank_scale  # [-1, 13, 1]

            consonant_out = self.out_consonant(x)  # [-1, 13, num_consonants]
            # consonant_out = consonant_out * self.out_consonant_scale  # [-1, 13, num_consonants]
            consonant_out_view = consonant_out.view(batch_size * T, num_consonants, 1)  # [-1*13, num_consonants, 1]
            if self.use_softmax:
                consonant_out_sm = F.softmax(consonant_out, dim=2)
                consonant_out_sm_view = consonant_out_sm.view(batch_size * T, num_consonants, 1)  # [-1*13, num_consonants, 1]

            glyph_out = self.out_glyph(x)  # [-1, 13, num_glyphs]
            # glyph_out = glyph_out * self.out_glyph_scale  # [-1, 13, num_glyphs]
            glyph_out_view = glyph_out.view(batch_size*T, 1, num_glyphs)   # [-1*13, 1, num_glyphs]
            if self.use_softmax:
                glyph_out_sm = F.softmax(glyph_out, dim=2)
                glyph_out_sm_view = glyph_out_sm.view(batch_size * T, 1, num_glyphs)  # [-1*13, 1, num_glyphs]

            if self.use_softmax:
                x = consonant_out_sm_view * glyph_out_sm_view  # [-1*13, num_consonants, num_glyphs]
            else:
                x = consonant_out_view * glyph_out_view  # [-1*13, num_consonants, num_glyphs]
            x = x.view(batch_size, T, num_consonants, num_glyphs)  # [-1, 13, num_consonants, num_glyphs]
            x = x.view(batch_size, T, num_consonants*num_glyphs)  # [-1, 13, num_consonants*num_glyphs]
            x = torch.cat((blank_out, x), dim=2)  # [-1, 13, num_chars=1+num_consonants*num_glyphs]

            out = x.permute(1, 0, 2)  # [13, -1, num_chars=1+num_consonants*num_glyphs]
            if self.use_out_scale:
                out = out * self.out_scale  # [13, -1, num_chars=1+num_consonants*num_glyphs]
            consonant_out = consonant_out.permute(1, 0, 2)  # [13, -1, num_consonants]
            glyph_out = glyph_out.permute(1, 0, 2)  # [13, -1, num_glyphs]

            # blank_out = blank_out * self.out_blank_scale  # [-1, 13, 1]
            # consonant_out = consonant_out * self.out_consonant_scale  # [-1, 13, num_consonants]
            # glyph_out = glyph_out * self.out_glyph_scale  # [-1, 13, num_glyphs]

        else:
            raise ValueError('Invalid mode')

        return out, consonant_out, glyph_out

    def forward(self, x):

        out, _, _ = self.forward_step(x)
        out = out.permute(1, 0, 2)  # [-1, 13, num_chars]
        y = F.softmax(out, 2).argmax(2)  # [-1, 13]

        text_list = []
        for y_eg in y:
            text = [self.idx2char[idx.item()] for idx in y_eg]
            text = "".join(text)
            text_list.append(text)

        return text_list

    def encode_char_batch(self, text_batch_raw):

        if self.mode == 'unicode':
            char_batch_tokenized = [list(text) for text in text_batch_raw]
        else:
            char_batch_tokenized = []
            for text in text_batch_raw:
                unicode_tokens = list(text)
                original_tokens = []
                for unicode_token in unicode_tokens:
                    if unicode_token in self.glyphs:
                        original_tokens[-1] += unicode_token
                    else:
                        original_tokens.append(unicode_token)
                char_batch_tokenized.append(original_tokens)

        char_batch_concat = []
        for tokens in char_batch_tokenized:
            for token in tokens:
                char_batch_concat.append(token)
        char_batch_targets = [self.char2idx[c] for c in char_batch_concat]
        char_batch_targets = torch.IntTensor(char_batch_targets)

        char_batch_targets_lens = [len(tokens) for tokens in char_batch_tokenized]
        char_batch_targets_lens = torch.IntTensor(char_batch_targets_lens)

        return char_batch_targets, char_batch_targets_lens

    def encode_consonant_glyph_batch(self, text_batch_raw):

        consonant_batch_tokenized = []
        glyph_batch_tokenized = []
        for text in text_batch_raw:
            unicode_tokens = list(text)
            original_tokens = []
            for unicode_token in unicode_tokens:
                if unicode_token in self.glyphs:
                    original_tokens[-1] += unicode_token
                else:
                    original_tokens.append(unicode_token)
            consonant_tokens = [self.character_split[original_token]['consonant'] for original_token in original_tokens]
            if self.remove_consonant_blanks:
                consonant_tokens = [t for t in consonant_tokens if t != '-']
            glyph_tokens = [self.character_split[original_token]['glyph'] for original_token in original_tokens]
            if self.remove_glyph_blanks:
                glyph_tokens = [t for t in glyph_tokens if t != '-']
            consonant_batch_tokenized.append(consonant_tokens)
            glyph_batch_tokenized.append(glyph_tokens)

        consonant_batch_concat = []
        for tokens in consonant_batch_tokenized:
            for token in tokens:
                consonant_batch_concat.append(token)
        consonant_batch_targets = [self.consonant2idx[c] for c in consonant_batch_concat]
        consonant_batch_targets = torch.IntTensor(consonant_batch_targets)

        consonant_batch_targets_lens = [len(tokens) for tokens in consonant_batch_tokenized]
        consonant_batch_targets_lens = torch.IntTensor(consonant_batch_targets_lens)

        glyph_batch_concat = []
        for tokens in glyph_batch_tokenized:
            for token in tokens:
                glyph_batch_concat.append(token)
        glyph_batch_targets = [self.glyph2idx[g] for g in glyph_batch_concat]
        glyph_batch_targets = torch.IntTensor(glyph_batch_targets)

        glyph_batch_targets_lens = [len(tokens) for tokens in glyph_batch_tokenized]
        glyph_batch_targets_lens = torch.IntTensor(glyph_batch_targets_lens)

        return consonant_batch_targets, consonant_batch_targets_lens, glyph_batch_targets, glyph_batch_targets_lens

    def compute_loss(self, text_batch_raw, text_batch_logits, consonant_batch_logits, glyph_batch_logits):
        """
        text_batch_raw: list of strings of length equal to batch size
        text_batch_logits: Tensor of size([T, batch_size, num_classes])
        """
        text_batch_targets, text_batch_targets_lens = self.encode_char_batch(text_batch_raw)

        text_batch_logps = F.log_softmax(text_batch_logits, 2)  # [T, batch_size, num_classes]
        text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),),
                                           fill_value=text_batch_logps.size(0),
                                           dtype=torch.int32)  # [batch_size]

        char_loss = self.criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)
        consonant_loss, glyph_loss = None, None

        if self.mode.startswith('multi-label'):

            consonant_batch_targets, consonant_batch_targets_lens, glyph_batch_targets, glyph_batch_targets_lens = \
                self.encode_consonant_glyph_batch(text_batch_raw)

            consonant_batch_logps = F.log_softmax(consonant_batch_logits, 2)  # [T, batch_size, num_classes]
            consonant_batch_logps_lens = torch.full(size=(consonant_batch_logps.size(1),),
                                                    fill_value=consonant_batch_logps.size(0),
                                                    dtype=torch.int32)  # [batch_size]

            consonant_loss = self.criterion(consonant_batch_logps, consonant_batch_targets, consonant_batch_logps_lens,
                                            consonant_batch_targets_lens)
            #consonant_loss = torch.max(consonant_loss, torch.zeros_like(consonant_loss))

            glyph_batch_logps = F.log_softmax(glyph_batch_logits, 2)  # [T, batch_size, num_classes]
            glyph_batch_logps_lens = torch.full(size=(glyph_batch_logps.size(1),),
                                                    fill_value=glyph_batch_logps.size(0),
                                                    dtype=torch.int32)  # [batch_size]

            glyph_loss = self.criterion(glyph_batch_logps, glyph_batch_targets, glyph_batch_logps_lens,
                                        glyph_batch_targets_lens)
            #glyph_loss = torch.max(glyph_loss, torch.zeros_like(glyph_loss))
            if torch.isinf(glyph_loss):
                glyph_loss = torch.ones_like(glyph_loss)

            #losses = [char_loss.item(), consonant_loss.item(), glyph_loss.item()]
            #print('\nLosses: ', losses)

            if torch.isnan(glyph_loss) or torch.isinf(glyph_loss):
                print(text_batch_raw)
                print(consonant_batch_logits)
                print(glyph_batch_logits)
                input()

        return char_loss, consonant_loss, glyph_loss

    def compute_accuracy_edit_distance(self, batch):
        image_batch, text_batch_raw, image_fn_batch = batch
        text_batch_pred = self.forward(image_batch)
        df = pd.DataFrame()
        df['image_fn'] = image_fn_batch
        df['actual'] = [fn.split("_")[0] for fn in image_fn_batch]
        df['prediction'] = text_batch_pred
        df['prediction_corrected'] = df['prediction'].apply(correct_prediction)
        df['edit_distance'] = df.apply(levenshtein_distance, axis=1)
        accuracy_edit_distance = {}
        for max_edit_distance in range(4 + 1):
            accuracy_edit_distance[max_edit_distance] = (df['edit_distance'] <= max_edit_distance).sum() / len(df)

        return accuracy_edit_distance

    def training_step(self, batch, batch_idx):
        image_batch, text_batch_raw, image_fn_batch = batch
        noise_batch = torch.randn_like(image_batch) * 0.001
        image_batch = image_batch + noise_batch
        text_batch_logits, consonant_batch_logits, glyph_batch_logits = self.forward_step(image_batch)
        char_loss, consonant_loss, glyph_loss = self.compute_loss(text_batch_raw, text_batch_logits,
                                                                  consonant_batch_logits, glyph_batch_logits)
        nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm_threshold)
        if self.mode.startswith('multi-label'):
            loss = self.weight_character_loss*char_loss + \
                   self.weight_consonant_loss*consonant_loss + \
                   self.weight_glyph_loss*glyph_loss
            self.log('training/loss', loss)
            self.log('training/loss_char', char_loss)
            self.log('training/loss_consonant', consonant_loss)
            self.log('training/loss_glyph', glyph_loss)
        else:
            loss = char_loss
            self.log('training/loss', loss)
            self.log('training/loss_char', char_loss)

        accuracy_edit_distance = self.compute_accuracy_edit_distance(batch)
        self.log('training/accuracy-e0', accuracy_edit_distance[0])
        self.log('training/accuracy-e1', accuracy_edit_distance[1])
        self.log('training/accuracy-e2', accuracy_edit_distance[2])
        self.log('training/accuracy-e3', accuracy_edit_distance[3])
        self.log('training/accuracy-e4', accuracy_edit_distance[4])

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        image_batch, text_batch_raw, image_fn_batch = batch
        text_batch_logits, consonant_batch_logits, glyph_batch_logits = self.forward_step(image_batch)
        char_loss, consonant_loss, glyph_loss = self.compute_loss(text_batch_raw, text_batch_logits,
                                                                  consonant_batch_logits, glyph_batch_logits)
        if self.mode.startswith('multi-label'):
            loss = self.weight_character_loss * char_loss + \
                   self.weight_consonant_loss * consonant_loss + \
                   self.weight_glyph_loss * glyph_loss
            self.log('val/loss', loss)
            self.log('val/loss_char', char_loss)
            self.log('val/loss_consonant', consonant_loss)
            self.log('val/loss_glyph', glyph_loss)
        else:
            loss = char_loss
            self.log('val/loss', loss)
            self.log('val/loss_char', char_loss)

        accuracy_edit_distance = self.compute_accuracy_edit_distance(batch)
        self.log('val/accuracy-e0', accuracy_edit_distance[0])
        self.log('val/accuracy-e1', accuracy_edit_distance[1])
        self.log('val/accuracy-e2', accuracy_edit_distance[2])
        self.log('val/accuracy-e3', accuracy_edit_distance[3])
        self.log('val/accuracy-e4', accuracy_edit_distance[4])

        return True

    def test_step(self, batch, batch_idx):

        image_batch, text_batch_raw, image_fn_batch = batch
        text_batch_logits, consonant_batch_logits, glyph_batch_logits = self.forward_step(image_batch)
        char_loss, consonant_loss, glyph_loss = self.compute_loss(text_batch_raw, text_batch_logits,
                                                                  consonant_batch_logits, glyph_batch_logits)
        if self.mode.startswith('multi-label'):
            loss = self.weight_character_loss * char_loss + \
                   self.weight_consonant_loss * consonant_loss + \
                   self.weight_glyph_loss * glyph_loss
            self.log('test/loss', loss)
            self.log('test/loss_char', char_loss)
            self.log('test/loss_consonant', consonant_loss)
            self.log('test/loss_glyph', glyph_loss)
        else:
            loss = char_loss
            self.log('test/loss', loss)
            self.log('test/loss_char', char_loss)

        accuracy_edit_distance = self.compute_accuracy_edit_distance(batch)
        self.log('test/accuracy-e0', accuracy_edit_distance[0])
        self.log('test/accuracy-e1', accuracy_edit_distance[1])
        self.log('test/accuracy-e2', accuracy_edit_distance[2])
        self.log('test/accuracy-e3', accuracy_edit_distance[3])
        self.log('test/accuracy-e4', accuracy_edit_distance[4])

        return True

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)

        return [optimizer], [scheduler]


class CRNN2(CRNN1):

    def __init__(self, model_config, characters, consonants, glyphs, character_split, training_config=None):

        super().__init__(model_config, characters, consonants, glyphs, character_split, training_config=training_config)
        self.cnn_p2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # [1, 6]
        self.linear1 = nn.Linear(256, 256)  # 256*1 = 256
