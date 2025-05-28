import os
from tqdm import tqdm

import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed

import config as CFG
from dataset import PDDDataset
from models import PhenoProfiler_MSE, PhenoProfiler
from utils import AvgMeter, get_lr
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='Train PhenoProfiler')

parser.add_argument('--exp_name', type=str, default='result/PhenoProfiler_MSE', help='')
parser.add_argument('--batch_size', type=int, default=200, help='')  # change it if cuda out of memory
parser.add_argument('--max_epochs', type=int, default=200, help='')
parser.add_argument('--num_workers', type=int, default=10, help='')
parser.add_argument('--pretrained_model', type=str, default=None, help='')
parser.add_argument('--init_method', default='tcp://127.0.0.1:3453', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument('--model', type=str, default='PhenoProfiler', help='')


def build_loaders(args):
    print("Building loaders")
    train_dataset = PDDDataset(image_path = "/data/boom/",
               embedding_path = '/data/boom/',
               CSV_path = "/data/boom/bbbc_train.csv")
    test_dataset = PDDDataset(image_path = "/data/boom/",
               embedding_path = '/data/boom/',
               CSV_path = "/data/boom/bbbc_test.csv")
    
    # print("Using pre-split train/test datasets")
    print(len(train_dataset), len(test_dataset))  # Ensure datasets are pre-split

    train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print("Finished building loaders")
    return train_loader, test_loader

def cleanup():
    dist.destroy_process_group()

def train_epoch(model, train_loader, optimizer, args, lr_scheduler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k in ["image", "embedding", "class"]}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()

        for param in model.parameters():
          if param.grad is not None:
              param.grad.data /= args.world_size

        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter

def test_epoch(model, test_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k in ["image", "embedding", "class"]}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def main():
    print("Starting...")
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    rank = int(os.environ.get("SLURM_NODEID", 0))*ngpus_per_node + local_rank
    current_device = local_rank
    torch.cuda.set_device(current_device)

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    print("process group ready!")

    # load the model
    print('From Rank: {}, ==> Making model..'.format(rank))

    if args.model == 'PhenoProfiler':
        model = PhenoProfiler().cuda(current_device)
    else:
        model = PhenoProfiler_MSE().cuda(current_device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load pre-trained model if specified
    if args.pretrained_model:
        model_path = args.pretrained_model + "/best.pt"
        print(f'Loading model from pretrained: {model_path}')
        #model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        pretrained_dict = torch.load(model_path, weights_only=True)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    #load the data
    print('From Rank: {}, ==> Preparing data..'.format(rank))
    train_loader, test_loader = build_loaders(args)
    
    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    # Train the model
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1}")

        if epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-4

        if epoch == 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4

        if epoch == 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5
        
        # Train the model
        model.train()
        train_epoch(model, train_loader, optimizer, args)
        
        if not os.path.exists(args.exp_name):
            os.makedirs(args.exp_name)

        # Evaluate the model
        model.eval()
        torch.save(model.state_dict(), str(args.exp_name) + "/last.pt")
        with torch.no_grad():
            test_loss = test_epoch(model, test_loader)
        
        if test_loss.avg < best_loss and rank == 0:
            best_loss = test_loss.avg
            best_epoch = epoch

            torch.save(model.state_dict(), str(args.exp_name) + "/best.pt")
            print("Saved Best Model! Loss: {}".format(best_loss))
        # break

    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))
    cleanup()

if __name__ == "__main__":
    main()

