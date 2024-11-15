# @Time : 2024/1/29
# @Author : WangXuSheng
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from model import EHSnet
import numpy as np
from utils_HSI import sample_gt, metrics, get_device, seed_worker
from datasets import HyperX, get_dataset
import os
import torch.utils.data as data
import scipy.io as io
from sklearn.metrics import classification_report
import clip
import time

parser = argparse.ArgumentParser(description='PyTorch EHSnet')

parser.add_argument('--save_path', type=str, default="./results/",
                    help='the path to save the model')
parser.add_argument('--data_path', type=str, default='./datasets/',
                    help='the path to load the data')
parser.add_argument('--dataset', type=str, default='Houston',
                    help='the name of dataset')
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--num_epoch', type=int, default=200,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=5,
                    help='multiple of of data augmentation')
# Training option
group_train = parser.add_argument_group('Training')
group_train.add_argument('--alpha', type=float, default=0.1,
                         help="Regularization parameter, controlling the contribution of different type of text "
                              "features.")
group_train.add_argument('--beta', type=float, default=1,
                         help="Regularization parameter, controlling the contribution of loss_clip ")
group_train.add_argument('--patch_size', type=int, default=13,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-2,
                         help="Learning rate, set by the model if not specified.")
group_train.add_argument('--batch_size', type=int, default=256,
                         help="Batch size (optional, if absent will be set by the model")
parser.add_argument('--seed', type=int, default=3667, metavar='S',
                    help='random seed ')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=False,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")

parser.add_argument('--with_exploration', default=True, action='store_true',
                    help="See data exploration visualization")

args = parser.parse_args()
DEVICE = get_device(args.cuda)

datasets_list = {
    'XS': {'source_name': 'XS_0', 'target_name': 'XS_1', 'source_label_name': 'XS_gt_0',
           'target_label_name': 'XS_gt_1'},
    'Pavia': {'source_name': 'paviaU', 'target_name': 'paviaC', 'source_label_name': 'paviaU_7gt',
              'target_label_name': 'paviaC_7gt'},
    'Houston': {'source_name': 'Houston13', 'target_name': 'Houston18', 'source_label_name': 'Houston13_7gt',
                'target_label_name': 'Houston18_7gt'},
}

description_list = {
    "XS": 'A hyperspectral image of the suburb of Xiongan city with the surface covered with roads, buildings, '
          'trees, farmland, bare land, orchard and water',
    "Pavia": 'A hyperspectral image of Pavia University with the surface covered with trees, asphalt, brick, '
             'bitumen,shadow, meadow, and bare soil',
    "Houston": 'A hyperspectral image of the University of Houston with the surface covered with grass healthy, '
               'grass stressed, trees, water, residential buildings, non-residential buildings and road',
}


def train(epoch, model, num_epoch, label_name, description=''):
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / num_epoch), 0.75)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if (epoch - 1) % 10 == 0:
        print('learning rate{: .4f}'.format(LEARNING_RATE))

    correct = 0
    iter_source = iter(train_loader)
    num_iter = len_src_loader

    for i in range(1, num_iter):

        model.train()
        data_src, label_src = iter_source.__next__()
        data_src, label_src = data_src.to(DEVICE), label_src.to(DEVICE)
        label_src = label_src - 1

        optimizer.zero_grad()
        text_1 = torch.cat(
            [clip.tokenize(f'A hyperspectral image of {label_name[int(k.item())]}').to(k.device) for k in label_src])
        text_2 = torch.cat(
            [clip.tokenize(description).to(k.device) for k in label_src])

        loss_clip, label_src_pred = model(data_src, text_1, text_2, label_src, text_ratio=args.alpha)
        loss_cls = F.nll_loss(F.log_softmax(label_src_pred, dim=1), label_src.long())
        loss = loss_cls + args.beta * loss_clip

        loss.backward()
        optimizer.step()

        pred = label_src_pred.data.max(1)[1]
        correct += pred.eq(label_src.data.view_as(pred)).cpu().sum()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch, i * len(data_src), len_src_dataset,
                                                             100. * i / len_src_loader))
            print('loss: {:.6f},  loss_cls: {:.6f},  loss_clip: {:.6f}'.format(
                loss.item(), loss_cls.item(), loss_clip.item()))
    acc = correct.item() / len_src_dataset
    print('[epoch: {:4}]  Train Accuracy: {:.4f} | train sample number: {:6}'.format(epoch, acc, len_src_dataset))

    return model


def test(model, label_name, description=''):
    model.eval()
    loss = 0
    correct = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            label = label - 1
            text_1 = torch.cat(
                [clip.tokenize(f'A hyperspectral image of {label_name[int(k.item())]}').to(k.device) for k in label])
            text_2 = torch.cat(
                [clip.tokenize(description).to(k.device) for k in label])
            loss_temp, label_src_pred = model(data, text_1, text_2, label)
            pred = label_src_pred.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(label_src_pred, dim=1), label.long()).item()
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss /= len_tar_loader
        print(
            'Average test loss: {:.4f},test Accuracy: {}/{} ({:.2f}%), | test sample number: {:6}\n'.format(
                loss, correct, len_tar_dataset, 100. * correct / len_tar_dataset, len_tar_dataset))

    return correct, correct.item() / len_tar_dataset, pred_list, label_list


if __name__ == '__main__':
    args.save_path = args.save_path + args.dataset + f'/alpha{args.alpha}_beta{args.beta}_e{args.num_epoch}'
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    args.data_path = args.data_path + args.dataset + '/'
    source_name = datasets_list[args.dataset]['source_name']
    target_name = datasets_list[args.dataset]['target_name']
    seed_worker(args.seed)
    print('load source dataset:', end=' ')
    img_src, gt_src, label_names_src, ignored_labels = get_dataset(source_name, args.data_path)
    print('load target dataset:', end=' ')
    img_tar, gt_tar, label_names_tar, ignored_labels = get_dataset(target_name, args.data_path)

    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': ignored_labels,
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, _, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src

    for i in range(args.re_ratio - 1):
        img_src_con = np.concatenate((img_src_con, img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))

    hyperparams_train = hyperparams.copy()
    hyperparams_train.update({'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})

    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True)
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  batch_size=256)
    len_src_loader = len(train_loader)
    len_src_dataset = len(train_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)

    print(hyperparams)
    print("train samples :", len_src_dataset)
    print("test samples :", len_tar_dataset)

    correct, acc = 0, 0
    pretrained_dict = torch.load('./ViT-B-32.pt', map_location="cpu").state_dict()
    embed_dim = pretrained_dict["text_projection"].shape[1]
    context_length = pretrained_dict["positional_embedding"].shape[0]
    vocab_size = pretrained_dict["token_embedding.weight"].shape[0]
    transformer_width = pretrained_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = 3
    model = EHSnet(embed_dim,
                   img_src.shape[-1], hyperparams['patch_size'], int(gt_src.max()),
                   context_length, vocab_size, transformer_width, transformer_heads,
                   transformer_layers).to(DEVICE)
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in pretrained_dict:
            del pretrained_dict[key]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'visual' not in k.split('.')}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')

    description = description_list[args.dataset]
    print(f'global description of the {args.dataset} : \'{description}\'')

    for epoch in range(1, args.num_epoch + 1):
        t1 = time.time()
        model = train(epoch, model, args.num_epoch, label_names_src, description)
        t2 = time.time()
        print('epoch time:', t2 - t1)

        test_correct, test_acc, pred, label = test(model, label_names_src, description)
        t3 = time.time()
        print('test time:', t3 - t2)
        if test_correct > correct:
            correct = test_correct
            acc = test_acc
            results = metrics(np.concatenate(pred), np.concatenate(label), ignored_labels=hyperparams['ignored_labels'],
                              n_classes=gt_src.max())
            print(classification_report(np.concatenate(pred), np.concatenate(label), target_names=label_names_tar))
            io.savemat(os.path.join(args.save_path, 'epoch_' + str(
                args.num_epoch) + '_results_' + source_name + '_' + f'{test_acc * 100 :.2f}' + '.mat'),
                       {'results': results})
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
            source_name, target_name, correct, 100. * correct / len_tar_dataset))
