## Generate dataset with all ranks given
## And will return the given torch.utils.data.Dataset instance
from numpy import empty
import os
import torch
import copy
import torch.nn.functional as F
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
import numpy as np
np.random.seed(2024)
import random
random.seed(2024)
from torchvision import datasets, transforms
import pandas
import pickle
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser

import sys
sys.path.append('/data0/yzhu/FLamingo/')
sys.path.append('/data0/yzhu/FedMetaAndInfo/')
from model import CharLSTM, AlexNet, FedAvgCNN
from FLamingo.core.utils.data_utils import save_data_to_picture
from FLamingo.core.utils.data_utils import ClientDataset

## This file contains functions that will try to calculate the 
# V-information in datasets and samples.

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

argsparser = ArgumentParser()
argsparser.add_argument('-t', '--train', type=int, default=0)
argsparser.add_argument('-d','--dataset_type', type=str, default='cifar')
argsparser.add_argument('-s','--dataset_dir', type=str, default='../datasets/')
argsparser.add_argument('-e','--epoch', type=int, default=5)
args = argsparser.parse_args()

data_dir = args.dataset_dir
save_dir = args.dataset_dir

if args.dataset_type == 'cifar10':
    # # cifar10 
    dataset_type = 'cifar10'
    empty_mode = 'random'
    train_bs = 32
    test_bs = 1
    zero_epoch = args.epoch
    epochs = args.epoch
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    num_clients = 30
    num_classes = 10
    lr = 0.05
elif args.dataset_type == 'femnist':
    # femnist
    dataset_type = 'femnist'
    empty_mode = 'random'
    train_bs = 32
    test_bs = 1
    zero_epoch = args.epoch
    epochs = args.epoch
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    num_clients = 30
    num_classes = 62
    lr = 0.03
else:
    # Shakespeare
    dataset_type = 'shakespeare'
    empty_mode = 'black'
    train_bs = 16
    test_bs = 1
    zero_epoch = args.epoch
    epochs = args.epoch
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    num_clients = 30
    num_classes = 80
    lr = 0.5

pics_path = save_dir + '/pics/' 
model_path = save_dir + '/models/'
save_dir += '/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir, exist_ok=True)
if os.path.exists(pics_path) is False:
    os.makedirs(pics_path, exist_ok=True)
if os.path.exists(model_path) is False:
    os.makedirs(model_path, exist_ok=True)
    
# rewrite the stdout to save_dir/log.log
log_file = open(save_dir+'log.log', 'w')
sys.stdout = log_file

# This function will train two models, one with empty input, one with normal input
def train(model, mode='white', rank=0):
    # train_set = datasets.CIFAR10(data_dir, train=True, download=True,
    #                              transform=transforms.Compose([
    #                                     transforms.RandomHorizontalFlip(),
    #                                     transforms.RandomCrop(32, 4),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                                                         (0.2023, 0.1994, 0.2010))
    #                                 ])
    #                              )

    client_set = ClientDataset(dataset_type, data_dir, rank)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_bs, shuffle=True)
    train_loader = client_set.get_train_loader(train_bs)
    
    # # model_empty = AlexNet(class_num=num_classes)
    # if 'cifar' in dataset_type:
    #     model_empty = AlexNet(class_num=num_classes)
    # elif 'femnist' in dataset_type:
    #     model_empty = FedAvgCNN(in_features=1, num_classes=num_classes)
    # elif 'shakespeare' in dataset_type:
    #     model_empty = CharLSTM()

    model_normal = copy.deepcopy(model)
    model_empty = copy.deepcopy(model)
    model_empty, model_normal = model_empty.to(device), model_normal.to(device)
    zero_optim = torch.optim.SGD(model_empty.parameters(), lr = lr, nesterov=True, momentum=0.9, weight_decay=5e-4)
    # zero_scheduler = torch.optim.lr_scheduler.ExponentialLR(zero_optim, gamma=0.993)
    loss_func = torch.nn.CrossEntropyLoss()
    for e in range(zero_epoch):
        num_zero, loss_zero = 0, 0.0
        for idx, (data, target) in enumerate(train_loader):
            # train zero input
            target = target.to(device)
            # Actually, we can use similar settings for both Shakespeare and Image Classification
            if mode == 'black':
                empty_input = torch.zeros_like(data).to(device)
            elif mode == 'white':
                empty_input = torch.ones_like(data).to(device)
            elif mode == 'random':
                empty_input = torch.rand_like(data).to(device)
            zero_optim.zero_grad()
            empty_out = model_empty(empty_input)
            empty_loss = loss_func(empty_out, target)
            empty_loss.backward()
            zero_optim.step()
            num_zero += len(data)
            loss_zero += empty_loss.item() * len(data)
            # zero_scheduler.step()
        print(f"Zero input: {num_zero} samples, loss {loss_zero/num_zero}")
    optimizer = torch.optim.SGD(model_normal.parameters(), lr = lr, nesterov=True, momentum=0.9, weight_decay=5e-4)
    loss_func = torch.nn.CrossEntropyLoss()
    # normal_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.993)
    torch.manual_seed(2024)
    for e in range(epochs):
        num, loss = 0, 0.0
        for idx, (data, target) in enumerate(train_loader):
            # train normal input
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            normal_out = model_normal(data)
            normal_loss = loss_func(normal_out, target)
            normal_loss.backward()
            optimizer.step()
            num += len(data)
            loss += normal_loss.item() * len(data)
            # normal_scheduler.step()
        print(f"Normal input: {num} samples, loss {loss/num}")
    return model_empty, model_normal
    
    
    # calc v info using models
def calc_v_info(model_empty, model_normal, rank=0):
    # test_set = datasets.CIFAR10(data_dir, train=False, download=True,
    #                     transform=transforms.Compose([
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                                             (0.2023, 0.1994, 0.2010))
    #                         ]))
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bs, shuffle=False)
    client_set = ClientDataset(dataset_type, data_dir, rank)
    test_loader = client_set.get_train_loader(test_bs)
    model_empty.eval()
    model_normal.eval()
    pvi_values = []
    HvY, HvYX = [], []
    test_len = len(test_loader)
    num_correct = 0
    with torch.no_grad():
        # empty_out = model_empty(torch.zeros_like(next(iter(test_loader))[0]).to(device))
        if empty_mode == 'black':
            empty_out = model_empty(torch.zeros_like(next(iter(test_loader))[0]).to(device))
        elif empty_mode == 'white':
            empty_out = model_empty(torch.ones_like(next(iter(test_loader))[0]).to(device))
        elif empty_mode == 'random':
            empty_out = model_empty(torch.rand_like(next(iter(test_loader))[0]).to(device))
        # 转换为softmax
        empty_out = F.softmax(empty_out, dim=1)
        # print(empty_out[0])
        # return
        empty_out = empty_out.tolist()[0]
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # get the probability of target in empty_out
            empty_out_score = empty_out[int(target)]
            normal_out = F.softmax(model_normal(data), dim=1)
            normal_pred = torch.argmax(normal_out, dim=1)
            normal_out_score = normal_out.tolist()[0][int(target)]
            fvy = -np.log2(empty_out_score)
            fvyx = -np.log2(normal_out_score)
            HvY.append(fvy)
            HvYX.append(fvyx)
            # pvi: entropy with x minus entropy without x to test how much info x has
            pvi_values.append((data, int(target), normal_pred, fvy-fvyx))
            num_correct += int(int(normal_pred) == int(target))
    print(f"num_correct: {num_correct}, test_len: {test_len}, acc: {num_correct/test_len}")
    return HvY, HvYX, pvi_values

def generate_info(model_empty, model_normal, rank=1):
    model_empty.eval()
    model_normal.eval()
    pvi_values = []
    HvY, HvYX = [], []
    train_data = np.load(save_dir+f'train/{rank}.npz', allow_pickle=True)
    # 使用 tensor dataset 转换为 DataLoader
    if 'cifar' in dataset_type or 'femnist' in dataset_type:
        test_set = TensorDataset(torch.tensor(train_data['data'],dtype=torch.float32), torch.tensor(train_data['targets']))
    elif 'shakespeare' in dataset_type:
        test_set = TensorDataset(torch.tensor(train_data['data'],dtype=torch.long), torch.tensor(train_data['targets']))
    test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False)
    test_len = len(test_loader)
    num_correct = 0
    with torch.no_grad():
        # empty_out = model_empty(torch.zeros_like(next(iter(test_loader))[0]).to(device))
        if empty_mode == 'black':
            empty_out = model_empty(torch.zeros_like(next(iter(test_loader))[0]).to(device))
        elif empty_mode == 'white':
            empty_out = model_empty(torch.ones_like(next(iter(test_loader))[0]).to(device))
        elif empty_mode == 'random':
            empty_out = model_empty(torch.rand_like(next(iter(test_loader))[0]).to(device))
        # 转换为softmax
        empty_out = F.softmax(empty_out, dim=1)
        # print(empty_out[0])
        # return
        empty_out = empty_out.tolist()[0]
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # get the probability of target in empty_out
            empty_out_score = empty_out[int(target)]
            normal_out = F.softmax(model_normal(data), dim=1)
            normal_pred = torch.argmax(normal_out, dim=1)
            normal_out_score = normal_out.tolist()[0][int(target)]
            fvy = -np.log2(empty_out_score)
            fvyx = -np.log2(normal_out_score)
            HvY.append(fvy)
            HvYX.append(fvyx)
            # pvi: entropy with x minus entropy without x to test how much info x has
            pvi_values.append(fvy-fvyx)
            num_correct += int(int(normal_pred) == int(target))
    print(f"num_correct: {num_correct}, test_len: {test_len}, acc: {num_correct/test_len}")
    print(f"Avg HvY: {np.mean(HvY)}, Avg HvYX: {np.mean(HvYX)}, Avg Pvi: {np.mean(pvi_values)}")
    np.savez_compressed(save_dir+f'info/{rank}.npz', 
                        data=train_data['data'],
                        targets=train_data['targets'],
                        pvi=pvi_values)

def draw_big_picture(filepath, rank=1):
    # Pvi = pandas.read_csv(filepath, index_col=False)
    Pvi = pickle.load(open(filepath, 'rb'))
    Pvi = Pvi.sort_values('pvi', ascending=False).reset_index(drop=True)
    print(Pvi.head())

    # Pvi is: data: (label, pvi_val)
    # It is sorted by pvi_val in descending order
    ## We want to print Pvi val and its label un-
    ## der each picture show its label and pvi_val

    print(f"len(Pvi): {len(Pvi)}")
    # Draw CIFAR10 images
    total_imgs = len(Pvi)

    big_img_col = 20
    big_img_row = total_imgs // big_img_col
    small_img_height = 32
    small_img_width = 32

    shard_height = 72
    shard_width = 100

    big_img_num_height = 11
    ttf = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', big_img_num_height)

    if 'mnist' in dataset_type:
        name_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z'
                ]
    elif 'cifar' in dataset_type:
        name_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # the image_large is the final image list
    # big_img = Image.new('L', (shard_width * big_img_col, big_img_row * shard_height), color=255)
    big_img = Image.new('RGB', (shard_width * big_img_col, big_img_row * shard_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(big_img)
    for i, data_piece in Pvi.iterrows():
        if i == total_imgs:
            break
        data, label, pred_val, pvi_val = data_piece['data'], data_piece['target'], data_piece['normal_pred'], data_piece['pvi']
        # transform data back to 0-255 value
        # data = data * 255
        data = data.to(torch.uint8).cpu().numpy().squeeze()
        # print(data.shape, data)
        if 'mnist' in dataset_type:
            data = np.stack([data]*3, axis=0)
            # print(data.shape)
        data = data.transpose(1, 2, 0)
        shard = Image.new('RGB', (shard_width, shard_height), color=(255, 255, 255))
        shard_draw = ImageDraw.Draw(shard)
        shard.paste(Image.fromarray(data), (0, 0))
        # draw label and pvi_val
        # msg = f'{name_list[int(label)]}\n{float(pvi_val):.6f}\nPred {name_list[int(pred_val)]}'
        msg1 = f'{name_list[int(label)]}'
        msg2 = f'{float(pvi_val):.6f}'
        msg3 = f'Pred {name_list[int(pred_val)]}'
        # if true label differ from pred label, draw it in RED
        if int(label) == int(pred_val):
            fill = (0, 0, 0)
        else:
            fill = (255, 0, 0)
        shard_draw.text((0, small_img_height), msg1, font=ttf, fill=fill)
        shard_draw.text((0, small_img_height+big_img_num_height), msg2, font=ttf, fill=fill)
        shard_draw.text((0, small_img_height+big_img_num_height*2), msg3, font=ttf, fill=fill)
        pos_x = shard_width * (i % big_img_col)
        pos_y = shard_height * (i // big_img_col)
        big_img.paste(shard, (pos_x, pos_y))

    big_img.save(pics_path+f'Pvi-{empty_mode}-{rank}.png')

def show_lines(filepath, rank=1):
    """This function will print all data in shakespeare dataset to show the pvi value.
    The wrong prediction will be shown in red.
    """
    Pvi = pickle.load(open(filepath, 'rb'))
    Pvi = Pvi.sort_values('pvi', ascending=False).reset_index(drop=True)
    print(Pvi.head())
    # Pvi is: data: (label, pvi_val)
    # It is sorted by pvi_val in descending order
    ## We want to print Pvi val and its label un-
    ## der each picture show its label and pvi_val

    print(f"len(Pvi): {len(Pvi)}")
    # Draw Shakespeare images
    total_lines = len(Pvi)
    # big_img_col = 20
    # big_img_row = total_imgs // big_img_col
    line_height = 15
    line_width = 55    # 80 per sentence, then one row for label, one for pred, one for pvi
    font_size = 13
    
    ttf = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size=font_size)
    big_img = Image.new('RGB', (line_width*font_size, total_lines * line_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(big_img)
    for i, data_piece in Pvi.iterrows():
        if i == total_lines:
            break
        data, label, pred_val, pvi_val = data_piece['data'], data_piece['target'], data_piece['normal_pred'], data_piece['pvi']
        # print(data.shape, data)
        # transform data back to english sentence according to ALL_LETTERS
        senten = ''.join([ALL_LETTERS[int(x)] for x in data.squeeze()])
        # senten += f'  {ALL_LETTERS[int(label)]}  {ALL_LETTERS[int(pred_val)]}  {pvi_val:.6f}'
        # Draw the senten on the big_img
        # if true label differ from pred label, draw it in RED
        if int(label) == int(pred_val):
            fill = (0, 0, 0)
        else:
            fill = (255, 0, 0)
        draw.text((0, line_height*i), senten, font=ttf, fill=(0,0,0))
        draw.text((45*font_size, line_height*i), f"{ALL_LETTERS[int(label)]}" ,font=ttf, fill=(0,0,0)  )
        draw.text((48*font_size, line_height*i), f"{ALL_LETTERS[int(pred_val)]}" ,font=ttf, fill=fill  )
        draw.text((51*font_size, line_height*i), f"{pvi_val:.6f}" ,font=ttf, fill=fill  )
    big_img.save(pics_path+f'Pvi-{empty_mode}-{rank}.png')
    

# if __name__ == '__main__':
if 'cifar' in dataset_type:
    model = AlexNet(class_num=num_classes)
elif 'femnist' in dataset_type:
    model = FedAvgCNN(in_features=1, num_classes=num_classes)
elif 'shakespeare' in dataset_type:
    model = CharLSTM()
# TRAIN = 1
# for i in range(1, num_clients+1):
for i in range(1, 51):
    # ==============Training Procedure===============
    if args.train:
        model_empty, model_normal = train(model, mode=empty_mode, rank=i)
        
        torch.save(model_empty.state_dict(), model_path+f'model_empty_{i}.pth')
        torch.save(model_normal.state_dict(), model_path+f'model_normal_{i}.pth')
    
    # ==================Calculation==================
    # else:
    model_normal = copy.deepcopy(model)
    model_empty = copy.deepcopy(model)
    model_empty.load_state_dict(torch.load(model_path+f'model_empty_{i}.pth'))
    model_normal.load_state_dict(torch.load(model_path+f'model_normal_{i}.pth'))
    model_normal, model_empty = model_normal.to(device), model_empty.to(device)
    HvY, HvYX, pvi_values = calc_v_info(model_empty, model_normal, rank=i)
    # print(pvi_values[0])
    # ========Save=========
    df = pandas.DataFrame(pvi_values, columns=['data', 'target', 'normal_pred', 'pvi'])
    # 将 HvY 和 HvYX 列添加到 df 后面
    df['HvY'] = HvY
    df['HvYX'] = HvYX
    if 'cifar' in dataset_type:
        # 将data转换成RGB 0-255 之间的值，std和mean都是：
        #transform = transforms.Compose(
        # [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        # )
        d_mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
        d_std = torch.tensor([0.5, 0.5, 0.5]).to(device)
        df['data'] = df['data'].apply(lambda x: torch.reshape(torch.round( (x*d_std[:,None,None,] + d_mean[:,None,None])*255 ), (3,32,32) ) )
    elif 'femnist' in dataset_type:
        # femnist是0-1之间的灰色图像
        df['data'] = df['data'].apply(lambda x: torch.reshape(torch.round( x*255 ), (1,28,28) ) )
    df['normal_pred'] = df['normal_pred'].apply(lambda x: int(torch.squeeze(x[0])))
    print(df.head())
    # Select the last 5 columns and export to csv
    df.iloc[:, 1:].to_csv(save_dir+'pvi_data.csv', index=False)
    with open(save_dir+'pvi_with_data.pkl', 'wb') as f:
        pickle.dump(df, f)
    # save data to picture within a big picture
    if 'cifar' in dataset_type or 'femnist' in dataset_type:
        draw_big_picture(save_dir+'pvi_with_data.pkl', rank=i)
    elif 'shake' in dataset_type:
        show_lines(save_dir+'pvi_with_data.pkl', rank=i)
    # generate pvi for each samples corresponding to save_dir/train/{rank}.npz
    if not os.path.exists(save_dir+'info/'):
        os.makedirs(save_dir+'info/', exist_ok=True)
    generate_info(model_empty, model_normal, rank=i)
    
    
    
    
        
    