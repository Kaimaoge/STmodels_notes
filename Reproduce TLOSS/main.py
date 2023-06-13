# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:54:21 2023

@author: AA
"""
import os
import sys

sys.path.append("..")
import torch
from datetime import datetime
import logging
import numpy as np
import argparse
from models.causal_cnn import CausalCNNEncoder
#from configs.sleepEDF import Config
from dataloader.dataloader import *
from dataloader.augmentations import Data_Transform
from models.loss import TripletLoss
from utils import _logger, _calc_metrics, copy_Files, set_requires_grad
from trainer.trainer import *
start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp4', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run2', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='fine_tune', type=str,
                    help='Modes of choice: self_supervised, fine_tune')
parser.add_argument('--selected_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()

device = torch.device(args.device)


# Logging

experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'T-LOSS'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

exec(f'from configs.{data_type} import Config')
Configs = Config()
model = CausalCNNEncoder(Configs).to(Configs.device)
model_optimizer = torch.optim.Adam(model.parameters(), lr=Configs.lr, betas=(Configs.beta1, Configs.beta2), weight_decay=3e-4)

total_loss = []
total_acc = []

train_dl, valid_dl, test_dl = data_generator(Configs)

# for epoch in range(1, Configs.num_epoch + 1):
#     total_loss = []
#     total_acc = []
#     model.train()
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = data.float().to(Configs.device)
#         x_ref, x_pos, x_neg = Data_Transform(data, Configs)
#         model_optimizer.zero_grad()
#         loss = criteria(model, x_ref, x_pos, x_neg)
#         total_loss.append(loss.item())
#         loss.backward()
#         model_optimizer.step()
        
#     train_loss = torch.tensor(total_loss).mean()
    
#     logger.debug(f'\nEpoch : {epoch}\n'
#                  f'Train Loss     : {train_loss:.4f}\t')
    
#     os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
#     chkpoint = {'model_state_dict': model.state_dict()}
#     torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    
#     logger.debug("\n################## Training is Done! #########################")


if training_mode == "fine_tune":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=Configs.device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
# Trainer
Trainer(model, model_optimizer, train_dl, valid_dl, test_dl, device, logger, Configs, experiment_log_dir, training_mode)


if training_mode != "self_supervised":
    # Testing
    outs = model_evaluate(model, test_dl, device, training_mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now()-start_time}")

# for epoch in range(1, Configs.num_epoch + 1):
#     total_loss = []
#     total_acc = []
#     model.train()
#     for batch_idx, (data, labels) in enumerate(train_loader):
#         data, labels = data.float().to(Configs.device), labels.long().to(Configs.device)
#         model_optimizer.zero_grad()
#         output = model(data)
#         predictions, features = output
#         loss = criterion(predictions, labels)
#         total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
#         total_loss.append(loss.item())
#         loss.backward()
#         model_optimizer.step()
       
#     outs = np.array([])
#     trgs = np.array([])
    
#     with torch.no_grad():
#         for data, labels in valid_loader:
#             data, labels = data.float().to(Configs.device), labels.long().to(Configs.device)
#             output = model(data)
#             predictions, features = output
#             loss = criterion(predictions, labels)
#             total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
#             total_loss.append(loss.item())
#             pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
#             outs = np.append(outs, pred.cpu().numpy())
#             trgs = np.append(trgs, labels.data.cpu().numpy())
#     test_loss = torch.tensor(total_loss).mean()
#     test_acc = torch.tensor(total_acc).mean()
        
#     scheduler.step(test_loss)
#     logger.debug('\nEvaluate on the Test set:')
#     logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')
    
# logger.debug("\n################## Training is Done! #########################")


# outs = np.array([])
# trgs = np.array([])

# with torch.no_grad():
#     for data, labels in test_loader:
#         data, labels = data.float().to(Configs.device), labels.long().to(Configs.device)
#         output = model(data)
#         predictions, features = output
#         loss = criterion(predictions, labels)
#         pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         outs = np.append(outs, pred.cpu().numpy())
#         trgs = np.append(trgs, labels.data.cpu().numpy())


# _calc_metrics(outs, trgs, experiment_log_dir, args.home_path)
    
        