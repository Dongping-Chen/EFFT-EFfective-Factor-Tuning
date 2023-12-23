import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
import numpy as np
import random
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from Imagenet_loader import *
from methods.Set_SSF import *
from methods.Set_LoRA import *
from Set_PoFT import *
from methods.Set_SSF_LoRA import *
from methods.Set_EFFT_SSF import *
from methods.Set_EFFT import *
from torch.cuda.amp import autocast, GradScaler
import json

device = torch.device("cuda:1")

def train(args, model, dl, opt, scheduler, epoch):
    model.train()
    model = model.to(args.device)
    scaler = GradScaler()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.to(args.device)
        for i, batch in enumerate(dl):
            x, y = batch[0].to(args.device), batch[1].to(args.device)
            opt.zero_grad()
            
            with autocast():
                out = model(x)
                loss = F.cross_entropy(out, y)
                
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        if scheduler is not None:
            scheduler.step(ep)
            
        if args.dataset == 'Imagenet' or ep % args.test_epoch == args.test_epoch - 1:
            acc = test(model, test_dl)
            if acc > args.best_acc:
                args.best_acc = acc
                save_checkpoint(model, opt, scheduler, ep, args.best_acc,'./checkpoints/{}_{}_{}_{}_{}.pth'.format(args.model,args.finetune,args.dataset,args.lr,"CHECKPOINT" if args.checkpoint != None else ""))
            pbar.set_description(str(acc) + '|' + str(args.best_acc))

    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    # pbar = tqdm(dl)
    model = model.to(args.device)
    for batch in dl:  # pbar:
        x, y = batch[0].to(args.device), batch[1].to(args.device)
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y)

    return acc.result()

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def save_checkpoint(model, optimizer, scheduler, epoch, acc, filename):
    model.eval()
    model = model.cpu()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # 保存scheduler的状态
        'best_acc': acc
    }
    print("Save Success!")
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(filename)
    if 'Imagenet' in filename:
        model.reset_classifier(1000)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:  # 如果提供了scheduler，从checkpoint中加载它的状态
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, epoch

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='ViT')
    parser.add_argument('--size', type=str, default='B')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--finetune', type=str, default='SSF')
    parser.add_argument('--best_acc', type=float, default=0)
    parser.add_argument('--device', type=str, default= str(torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--t_initial', type=int, default=100)
    parser.add_argument('--warmup_t', type=int, default=10)
    parser.add_argument('--lr_min',type=float, default=1e-5)
    parser.add_argument('--warmup_lr_init',type=float,default=1e-6)
    parser.add_argument('--epoch', type=str, default=100)
    parser.add_argument('--test_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
        
    return parser
    
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args() 
    seed = args.seed
    set_seed(seed)
    
    name = args.dataset
    with open("./configs/EFFT2.json","r") as f:
        configs = json.load(f)
    rank = configs[name]['rank']
    scale = configs[name]['scale']
    init = configs[name]['init']
    
    if args.model == 'ViT':
        if args.size == 'B':
            model = create_model('vit_base_patch16_224_in21k', checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1).to(args.device)
        elif args.size == 'L':
            model = create_model('vit_large_patch16_224_in21k', checkpoint_path='./imagenet21k_ViT-L_16.npz', drop_path_rate=0.1).to(args.device)
        elif args.size == 'H':
            model = create_model('vit_huge_patch14_224_in21k', checkpoint_path='./imagenet21k_ViT-H_14.npz', drop_path_rate=0.1).to(args.device)
    elif args.model == 'Swin':
        if args.size == 'B':
            model = create_model("swin_base_patch4_window7_224_in22k", checkpoint_path='./swin_base_patch4_window7_224_22k.pth', drop_path_rate=0.1).to(args.device)
        elif args.size == 'S':
            model = create_model("swin_small_patch4_window7_224_in22k", checkpoint=None, drop_path_rate=0.1)
        elif args.size == 'L':
            model = create_model("swin_large_patch4_window7_224_in22k", checkpoint_path='./swin_large_patch4_window7_224_22k.pth', drop_path_rate=0.1).to(args.device)
        elif args.size == 'T':
            model = create_model("swin_tiny_patch4_window7_224", checkpoint_path='./swin_tiny_patch4_window7_224.pth', drop_path_rate=0.1).to(args.device)
    else:
        print("Wrong model!")
        exit(0)
    
    if args.size == 'B':
        if args.model == 'ViT':
            if args.finetune == 'SSF':
                set_ViT_SSF(model)
            elif args.finetune == 'LoRA_qv':
                set_ViT_LoRA_qv(model)
            elif args.finetune == 'LoRA_ffn':
                set_ViT_LoRA_ffn(model)
            elif args.finetune == 'LoRA_all':
                set_ViT_LoRA_all(model)
            elif args.finetune == 'PoFT':
                set_ViT_PoFT(model)
            elif args.finetune == 'SSF_PoFT':
                set_ViT_PoFT_SSF(model)
            elif args.finetune == 'SSF_LoRA_qv':
                set_ViT_SSF_LoRA_qv(model)
            elif args.finetune == 'SSF_LoRA_all':
                set_ViT_SSF_LoRA_all(model)
            elif args.finetune == 'SSF_LoRA_ffn':
                set_ViT_SSF_LoRA_ffn(model)
            elif args.finetune == 'EFFT':
                set_ViT_EFFT(model, dim = rank, s = scale, init = init)
        elif args.model == 'Swin':
            if args.finetune == 'SSF':
                set_Swin_SSF(model)
            elif args.finetune == 'LoRA_qv':
                set_Swin_LoRA_qv(model)
            elif args.finetune == 'PoFT':
                set_Swin_PoFT(model)
            elif args.finetune == 'EFFT':
                set_Swin_EFFT(model,s=1)
    else:
        if args.finetune == 'EFFT':
            if args.model == 'ViT':
                if args.size == 'L':
                    set_ViT_L_EFFT(model)
                elif args.size == 'H':
                    set_ViT_H_EFFT(model)
            elif args.model == 'Swin':
                if args.size == 'S':
                    set_Swin_S_EFFT(model, s=1)
                elif args.size == 'L':
                    set_Swin_L_EFFT(model, s=1)
        if args.finetune == 'LoRA_qv':
            if args.model == 'ViT':
                if args.size == 'L':
                    set_ViT_LoRA_qv(model)
                elif args.size == 'H':
                    set_ViT_LoRA_qv(model)
            elif args.model == 'Swin':
                if args.size == 'S':
                    set_Swin_LoRA_qv(model)
                elif args.size == 'L':
                    set_Swin_LoRA_qv(model)
                elif args.size == 'T':
                    set_Swin_LoRA_qv(model)
                

    if args.checkpoint != None:
        model,_,_,_ = load_checkpoint(args.checkpoint, model)
        
    if name == 'Imagenet':
        train_dl = Imagenet_train_loader(batch_size=args.batch_size)
        test_dl = Imagenet_val_loader(batch_size=args.batch_size)
    else:
        train_dl, test_dl = get_data(name,batch_size=args.batch_size)
    
    model.reset_classifier(get_classes_num(name))
    
    trainable = []
    total_param = 0
    if args.finetune == 'linear':
        for n, p in model.named_parameters():
            if 'head' in n :
                trainable.append(p)
                total_param += p.numel()
            else:
                p.requires_grad = False
    elif args.finetune == 'full':
        for n, p in model.named_parameters():
            trainable.append(p)
            total_param += p.numel()
    else:
        for n, p in model.named_parameters():
            if 'SSF' in n or 'head' in n or 'LoRA' in n or 'PoFT' in n or 'EFFT' in n:
                trainable.append(p)
                if 'head' not in n:
                    total_param += p.numel()
            else:
                p.requires_grad = False
    print('total_param', total_param)
    
    
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=args.t_initial,
                                warmup_t=args.warmup_t, lr_min=args.lr_min, warmup_lr_init=args.warmup_lr_init)
    model = train(args, model, train_dl, opt, scheduler, epoch=args.epoch)
    to_print = str(args)
    print(to_print)
    with open('./results/{}_{}.txt'.format(args.model,args.finetune), 'a') as f:
        f.write(to_print + '\n')
