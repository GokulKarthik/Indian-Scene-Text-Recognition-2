import logging
import os
import shutil
import yaml

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from datasets import IndianSceneTextDataset
from recognizers import CRNN1, CRNN2
from utils import transform_char_df


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['generate_adversarial_examples']

data_dir = config['data']['data_dir']
language_dir = config['data']['language_dir']
character_map_file = config['data']['character_map_file']
tokenize_method = config['data']['tokenize_method']
max_tokens = config['data']['max_tokens']

attack_model = config['model']['name']
attack_model_mode = config['model']['mode']
is_bidirectional = config['model']['is_bidirectional']
rnn_hidden_size = config['model']['rnn_hidden_size']
use_out_scale = config['model']['use_out_scale']
use_softmax = config['model']['use_softmax']

save_path_val = config['save_path_val']
save_path_test = config['save_path_test']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f"Device: {device}")

save_path_val = os.path.join(save_path_val, language_dir, attack_model, attack_model_mode)
if os.path.exists(save_path_val):
    shutil.rmtree(save_path_val)
os.makedirs(save_path_val)

save_path_test = os.path.join(save_path_test, language_dir, attack_model, attack_model_mode)
if os.path.exists(save_path_test):
    shutil.rmtree(save_path_test)
os.makedirs(save_path_test)


def get_model(config):
    logging.info("Entering the function 'get_model' in 'generate_adversarial_examples.py'")
    global characters, consonants, glyphs, character_split,device

    language_dir = config['data']['language_dir']
    attack_model = config['model']['name']
    attack_model_mode = config['model']['mode']

    if attack_model == 'crnn-1':
        model = CRNN1(model_config=config['model'], characters=characters, consonants=consonants, glyphs=glyphs,
                      character_split=character_split, purpose='attack')
    elif attack_model == 'crnn-2':
        model = CRNN2(model_config=config['model'], characters=characters, consonants=consonants, glyphs=glyphs,
                      character_split=character_split, purpose='attack')
    else:
        raise ValueError("Undefined classifier")
    model_state = torch.load(os.path.join('Models', f'{language_dir}_{attack_model}_{attack_model_mode}.pth'))
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    logging.info("Exiting the function 'get_model' in 'generate_adversarial_examples.py'")
    return model


def make_dirs(config, save_path):
    logging.info("Entering the function 'make_dirs' in 'generate_adversarial_examples.py'")

    attack_save_path = os.path.join(save_path, f"Attack-{config['attack_id']}")
    os.mkdir(attack_save_path)

    with open(os.path.join(attack_save_path, 'config.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    logging.info("Exiting the function 'make_dirs' in 'generate_adversarial_examples.py'")
    return True


def compute_loss(model, criterion, text_bounded_mb, text_batch_logits):
    text_batch_targets, text_batch_targets_lens = model.encode_char_batch(text_bounded_mb)

    text_batch_logps = F.log_softmax(text_batch_logits, 2)  # [T, batch_size, num_classes]
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),),
                                       fill_value=text_batch_logps.size(0),
                                       dtype=torch.int32)  # [batch_size]

    char_loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return char_loss


def fgsm(model, image_mb, text_bounded_mb, epsilons=0.01):

    trans = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    inv_trans = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])

    criterion = nn.CTCLoss(blank=0, zero_infinity=False)
    image_mb.requires_grad = True

    text_batch_logits, consonant_batch_logits, glyph_batch_logits = model.forward_step(image_mb)
    char_loss = compute_loss(model, criterion, text_bounded_mb, text_batch_logits)

    model.zero_grad()
    char_loss.backward()
    image_mb_grad = image_mb.grad.data
    image_mb_grad_sign = image_mb_grad.sign()
    attacks_mb = image_mb + epsilons * image_mb_grad_sign

    attacks_mb = inv_trans(attacks_mb)
    attacks_mb = torch.clamp(attacks_mb, 0, 1)

    model.eval()
    attacks_mb_cp = attacks_mb.clone()
    attacks_mb_cp = trans(attacks_mb_cp)
    out, _, _ = model.forward_step(attacks_mb_cp)
    out = out.permute(1, 0, 2)  # [-1, 13, num_chars]
    y = F.softmax(out, 2).argmax(2)  # [-1, 13]

    text_list = []
    for y_eg in y:
        text = [model.idx2char[idx.item()] for idx in y_eg]
        text = "".join(text)
        text_list.append(text)

    return attacks_mb, text_list


def make_adversarial_examples(config, dataset, save_path, batch_size=1024):
    logging.info("Entering the function 'make_adversarial_examples' in 'generate_adversarial_examples.py'")
    global model

    attack_save_path = os.path.join(save_path, f"Attack-{config['attack_id']}")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_idx, (image_mb, text_bounded_mb, image_fn_mb) in tqdm(enumerate(data_loader),
                                                                    leave=False,
                                                                    total=len(data_loader),
                                                                    desc=f"{save_path}; Attack: {config['attack_id']}"):
        image_mb = image_mb.to(device)
        attack_function = config['attack_function']
        if attack_function == 'fgsm':
            attacks_mb, text_list = fgsm(model, image_mb, text_bounded_mb, epsilons=config['epsilons'])
        elif attack_function == 'none':
            attacks_mb, text_list = fgsm(model, image_mb, text_bounded_mb, epsilons=0)
        else:
            raise ValueError('Undefined Attack')
        for i in tqdm(range(len(image_mb)), leave=False, desc=f"{save_path}; Attack: {config['attack_id']}; Batch: {batch_idx}"):
            adversarial_image = attacks_mb[i]
            text, image_id = image_fn_mb[i][:-4].split('_', 1)
            text_predicted = text_list[i]
            attack_image_save_path = os.path.join(attack_save_path, f'{text}_{image_id}_{text_predicted}.png')
            save_image(adversarial_image, attack_image_save_path)

    logging.info("Entering the function 'make_adversarial_examples' in 'generate_adversarial_examples.py'")
    return True


val_dir = os.path.join('Data', language_dir, 'Val')
test_dir = os.path.join('Data', language_dir, 'Test')
character_map_file_path = os.path.join('Data', character_map_file)

# character mapping
character_df = pd.read_csv(character_map_file_path)
character_df_transformed, character_split = transform_char_df(character_df)

if attack_model_mode == 'unicode':
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
print("Consonants:\n", consonants)
print("Glyphs:\n", glyphs)

valset = IndianSceneTextDataset(val_dir, tokenize_method=tokenize_method, max_tokens=max_tokens, glyphs=glyphs)
testset = IndianSceneTextDataset(test_dir, tokenize_method=tokenize_method, max_tokens=max_tokens, glyphs=glyphs)
print('Datasets - Val, Test:', len(valset), len(testset))

model = get_model(config)
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)


attack_configs = []

# Attack: 0
# No attack
######################
config['attack_id'] = 0
config['attack_function'] = 'none'
config['epsilons'] = 0
attack_configs.append(config.copy())

# Attack: 1
# Linf-FGSM
######################
config['attack_id'] = 1
config['attack_function'] = 'fgsm'
config['epsilons'] = 0.005
attack_configs.append(config.copy())


# Attack: 2
# Linf-FGSM
######################
config['attack_id'] = 2
config['attack_function'] = 'fgsm'
config['epsilons'] = 0.01
attack_configs.append(config.copy())


# Attack: 3
# Linf-FGSM
######################
config['attack_id'] = 3
config['attack_function'] = 'fgsm'
config['epsilons'] = 0.05
attack_configs.append(config.copy())


# Attack: 4
# Linf-FGSM
######################
config['attack_id'] = 4
config['attack_function'] = 'fgsm'
config['epsilons'] = 0.1
attack_configs.append(config.copy())


# Attack: 5
# Linf-FGSM
######################
config['attack_id'] = 5
config['attack_function'] = 'fgsm'
config['epsilons'] = 1
attack_configs.append(config.copy())

for attack_config in tqdm(attack_configs, desc='Attacks'):

    make_dirs(attack_config, save_path_val)
    make_adversarial_examples(attack_config, valset, save_path_val)

    make_dirs(attack_config, save_path_test)
    make_adversarial_examples(attack_config, testset, save_path_test)
