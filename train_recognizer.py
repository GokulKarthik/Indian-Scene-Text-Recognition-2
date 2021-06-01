import os
import yaml

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from datasets import IndianSceneTextDataset
from recognizers import CRNN1, CRNN2
from utils import transform_char_df

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['train_recognizer']

data_dir = config['data']['data_dir']
language_dir = config['data']['language_dir']
character_map_file = config['data']['character_map_file']
num_train_samples = config['data']['num_train_samples']
tokenize_method = config['data']['tokenize_method']
max_tokens = config['data']['max_tokens']

model_name = config['model']['name']
mode = config['model']['mode']
is_bidirectional = config['model']['is_bidirectional']
rnn_hidden_size = config['model']['rnn_hidden_size']

max_epochs = config['training']['max_epochs']
batch_size = config['training']['batch_size']

torch.manual_seed(0)
if mode == 'unicode':
    assert tokenize_method == 'unicode'
else:
    assert tokenize_method == 'original'
if mode in ['original', 'unicode']:
    assert config['training']['weight_character_loss'] == 1
    assert config['training']['weight_consonant_loss'] == 0
    assert config['training']['weight_glyph_loss'] == 0

train_dir = os.path.join(data_dir, language_dir, 'Train')
val_dir = os.path.join(data_dir, language_dir, 'Val')
test_dir = os.path.join(data_dir, language_dir, 'Test')
character_map_file_path = os.path.join(data_dir, character_map_file)


# character mapping
character_df = pd.read_csv(character_map_file_path)
character_df_transformed, character_split = transform_char_df(character_df)

if mode == 'unicode':
    characters = set()
    for character in character_df['Character'].values:
        characters.update(list(character))
    characters = sorted(list(characters))
else:
    characters = character_df_transformed.values.flatten().tolist()

characters = ['-'] + characters
num_chars = len(characters)
consonants = character_df_transformed.index.tolist()
glyphs = character_df_transformed.columns.tolist()
print("Characters:\n", characters)
print("Consonants:\n", consonants)
print("Glyphs:\n", glyphs)


# data
trainset = IndianSceneTextDataset(train_dir, tokenize_method=tokenize_method, max_tokens=max_tokens, glyphs=glyphs,
                                  num_train_samples=num_train_samples)
valset = IndianSceneTextDataset(val_dir, tokenize_method=tokenize_method, max_tokens=max_tokens, glyphs=glyphs)
testset = IndianSceneTextDataset(test_dir, tokenize_method=tokenize_method, max_tokens=max_tokens, glyphs=glyphs)
print('Datasets - Train, Val, Test:', len(trainset), len(valset), len(testset))

train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=False)
test_loader = DataLoader(testset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=False)
print('Loaders - Train, Val, Test:', len(train_loader), len(val_loader), len(test_loader))


# model
if model_name == 'crnn-1':
    model = CRNN1(model_config=config['model'], characters=characters, consonants=consonants, glyphs=glyphs,
                  character_split=character_split, training_config=config['training'])
elif model_name == 'crnn-2':
    model = CRNN2(model_config=config['model'], characters=characters, consonants=consonants, glyphs=glyphs,
                  character_split=character_split, training_config=config['training'])
else:
    raise ValueError('Invalid Model')


# training
wandb_logger = WandbLogger(project='indian-scene-text-recognition')
wandb_logger.watch(model, log='gradients', log_freq=100)
wandb_logger.log_hyperparams(config)

trainer = pl.Trainer(gpus=1, logger=wandb_logger, max_epochs=max_epochs, fast_dev_run=False)
trainer.fit(model, train_loader, val_loader)


# save
checkpoint_name = '_'.join([language_dir, model_name, mode]) + '.pth'
checkpoint_save_path = os.path.join('Models', checkpoint_name)
checkpoint = model.state_dict()
torch.save(checkpoint, checkpoint_save_path)
print('Model saved!')