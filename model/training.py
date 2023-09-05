import os
import time
from typing import Any

import torch
from numpy.core.defchararray import upper
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.sgd import SGD
from torch.utils import data
from tqdm import tqdm

from configuration.configuration import Configuration
from model.evaluation import evaluate
from utils.custom_dataset import NerDataset
from utils.trainer_utils import EarlyStopping


def train(model: nn.Module, training_set: NerDataset, validation_set: NerDataset, conf: Configuration,
          grid_search_param: dict, output_dir='saved_model', data_processor=None):
    train_dataloader = data.DataLoader(dataset=training_set, batch_size=grid_search_param['batch_size'],
                                       shuffle=True, collate_fn=NerDataset.pad)
    dev_dataloader = data.DataLoader(dataset=validation_set, batch_size=grid_search_param['batch_size'],
                                     shuffle=False, collate_fn=NerDataset.pad)

    gradient_accumulation_steps = 1
    warmup_proportion = 0.1

    total_train_steps = int(
        len(training_set) / grid_search_param['batch_size'] / gradient_accumulation_steps * grid_search_param[
            'max_epoch'])

    print("***** Running training *****")
    print("  Num examples = %d" % len(training_set))
    print("  Batch size = %d" % grid_search_param['batch_size'])
    print("  Num steps = %d" % total_train_steps)
    print("  Parameters:", grid_search_param)

    if upper(conf.hyperparam_tuning) == 'Y' and os.path.exists(output_dir + '/ner_model_trained.pt'):
        checkpoint = torch.load(output_dir + '/ner_model_trained.pt', map_location=conf.gpu)
        start_epoch = checkpoint['epoch'] + 1
        valid_f1_prev = checkpoint['valid_f1']
        pretrained_dict = checkpoint['model_state']
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)
        print('Loaded the pretrain NER_BERT_CRF model, epoch:', checkpoint['epoch'], 'valid acc:',
              checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
    else:
        start_epoch = 0
        valid_f1_prev = 0

    model.to(conf.gpu)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
                    and not any(nd in n for nd in new_param)],
         'weight_decay': grid_search_param['weight_decay_finetune']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
                    and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if n in ('transitions', 'hidden2label.weight')],
         'lr': grid_search_param['lr_crf'], 'weight_decay': grid_search_param['weight_decay_crf_fc']},
        {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'], 'lr': grid_search_param['lr_crf'],
         'weight_decay': 0.0}
    ]
    optimizer = Any
    if grid_search_param['optimizer'] == 'Adam':
        optimizer = Adam(optimizer_grouped_parameters, lr=grid_search_param['learning_rate'])
    if grid_search_param['optimizer'] == 'SGD':
        optimizer = SGD(optimizer_grouped_parameters, lr=grid_search_param["learning_rate"],
                        momentum=grid_search_param["momentum"],
                        weight_decay=grid_search_param["weight_decay_finetune"], nesterov=True)

    # --------- Early stopping ---------
    es = EarlyStopping(
        grid_search_param["max_epoch"] if grid_search_param['early_stopping'] <= 0 else grid_search_param[
            'early_stopping'])

    # --------- Scheduling the learning rate to improve the convergence ---------
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=grid_search_param['early_stopping'])

    # train procedure
    global_step_th = int(
        len(train_dataloader) / grid_search_param['batch_size'] / gradient_accumulation_steps * start_epoch)

    tr_loss = 0
    epoch = 0
    accuracy, recall, precision, f1 = 0, 0, 0, 0
    while epoch < grid_search_param['max_epoch'] and not es.earlyStop:
        train_start = time.time()
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training", mininterval=conf.refresh_rate)):
            batch = tuple(t.to(conf.gpu) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

            neg_log_likelihood = model.neg_log_likelihood(input_ids, input_mask, label_ids)
            if gradient_accumulation_steps > 1:
                neg_log_likelihood = neg_log_likelihood / gradient_accumulation_steps

            neg_log_likelihood.backward()
            tr_loss += neg_log_likelihood.item()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=2.0)
            if grid_search_param["optimizer"] == 'Adam' and (step + 1) % gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = grid_search_param["learning_rate"] * warmup_linear(global_step_th / total_train_steps,
                                                                                  warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

            optimizer.zero_grad()
            optimizer.step()
            scheduler.step(tr_loss)
            global_step_th += 1
        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss / len(train_dataloader),
                                                                                 (time.time() - train_start) / 60.0))
        result_df, _ = evaluate(model, dev_dataloader, epoch, 'Valid_set', conf, data_processor=data_processor)

        # Save a checkpoint
        if result_df['F1'].max() > valid_f1_prev:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': result_df['Accuracy'].max(),
                        'valid_f1': result_df['F1'].max(), 'max_seq_length': 180, 'lower_case': False,
                        'batch_size': grid_search_param['batch_size']},
                       os.path.join('saved_models/', 'ner_model_trained.pt'))
            accuracy = result_df['Accuracy'].max()
            precision = result_df['Precision'].max()
            recall = result_df['Recall'].max()
            f1 = result_df['F1'].max()
            valid_f1_prev = result_df['F1'].max()
        # Update the scheduler
        scheduler.step(result_df['F1'].max())
        # Update the early stopping
        es.update(result_df['F1'].max())
        epoch += 1
    return {
        'config': grid_search_param,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x
