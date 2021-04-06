import re
import os
import time
import yaml
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from functools import reduce

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from model import NeoMeanMaxExcite_v2
from utils import parse_textgrid

import warnings
warnings.filterwarnings("ignore")


def evaluate_metrics(pred_label, true_label):
    pred_label = np.array(pred_label)
    true_label = np.array(true_label)
    wa = np.mean(pred_label.astype(int) == true_label.astype(int))
    pred_onehot = np.eye(4)[pred_label.astype(int)]
    true_onehot = np.eye(4)[true_label.astype(int)]
    ua = np.mean(np.sum((pred_onehot==true_onehot)*true_onehot,axis=0)/np.sum(true_onehot,axis=0))
    key_metric, report_metric = 0.9*wa+0.1*ua, {'wa':wa,'ua':ua}
    return key_metric, report_metric



class RAVDESSDataset(object):
    def __init__(self, config, data_list):
        self.data_list = data_list
        self.vocabulary_dict = pickle.load(open(config['semantic']['vocabulary_path'],'rb'))
        self.audio_length = config['acoustic']['audio_length']
        self.feature_name = config['acoustic']['feature_name']
        self.feature_dim = config['acoustic']['embedding_dim']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, asr_text, align_path, label = self.data_list[index]
        audio_name = os.path.basename(audio_path)
        #------------- extract the audio features -------------#
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.feature_name == 'fbank':
            audio_input = torchaudio.compliance.kaldi.fbank(
                waveform, sample_frequency=sample_rate, num_mel_bins=self.feature_dim, 
                frame_length=25, frame_shift=10, use_log_fbank=True
            )
        else:
            raise ValueError('Current feature type does not supported!')
        if self.audio_length is not None: audio_input=audio_input[:self.audio_length,:]
        audio_length = audio_input.size(0)
        #------------- extract the text contexts -------------#
        text_words = [x.lower() for x in re.split(' +',re.sub('[\.,\?\!]',' ', asr_text))]
        text_input = torch.LongTensor([int(self.vocabulary_dict.get(x,'-1')) for x in text_words if len(x)>0])
        # Here we use the 0 to represent the padding tokens
        text_input = text_input + 1
        text_length = text_input.size(0)
        #------------- generate the force alignment matrix -------------#
        align_info = [x for x in parse_textgrid(align_path) if x[0] != 'None']
        align_input = []
        for _, begin_time, end_time in align_info:
            begin_idx = int(begin_time / 0.01)
            end_idx = int(end_time / 0.01) + 1
            align_slice = torch.zeros(audio_input.size(0))
            align_slice[begin_idx:end_idx] = 1.0
            align_input.append(align_slice[None,:])
        align_input = torch.cat(align_input, dim=0)
        #------------- wrap up all the output info the dict format -------------#
        return {'audio_input':audio_input,'text_input':text_input,'audio_length':audio_length,
                'text_length':text_length,'align_input':align_input,'label':label,'audio_name':audio_name}


def collate(sample_list):
    batch_audio = [x['audio_input'] for x in sample_list]
    batch_text = [x['text_input'] for x in sample_list]
    batch_audio = pad_sequence(batch_audio, batch_first=True)
    batch_text = pad_sequence(batch_text, batch_first=True)

    audio_length = torch.LongTensor([x['audio_length'] for x in sample_list])
    text_length = torch.LongTensor([x['text_length'] for x in sample_list])

    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
    batch_name = [x['audio_name'] for x in sample_list]
    #------------- pad for the alignment result -------------#
    batch_align = [F.pad(
        x['align_input'],
        (0,int((torch.max(audio_length)-x['align_input'].size(1)).numpy()),
         0,int((torch.max(text_length)-x['align_input'].size(0)).numpy())),
        "constant", 0
    )[None,:,:] for x in sample_list]
    batch_align = torch.cat(batch_align, dim=0)
    return ((batch_audio,audio_length),(batch_text,text_length),batch_align),batch_label,batch_name


def run(args, config, train_data, valid_data):
    assert config['loss']['name'] in ['CTC','BCE']

    ############################ PARAMETER SETTING ##########################
    num_workers = 8
    batch_size = 32
    epochs = args.epochs
    learning_rate = 5e-4

    ############################## PREPARE DATASET ##########################
    train_dataset = RAVDESSDataset(config, train_data)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, batch_size = batch_size, collate_fn=lambda x: collate(x),
        shuffle = True, num_workers = num_workers
    )
    valid_dataset = RAVDESSDataset(config, valid_data)
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset, batch_size = batch_size, collate_fn=lambda x: collate(x),
        shuffle = False, num_workers = num_workers
    )
    ########################### CREATE MODEL #################################
    model = NeoMeanMaxExcite_v2(config)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if config['loss']['name'] == 'BCE':
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CTCLoss(blank=config['classifier']['class_num'])
    ########################### TRAINING #####################################
    count, best_metric, save_metric, best_epoch = 0, -np.inf, None, 0

    for epoch in range(epochs):
        epoch_train_loss = []
        model.train()
        start_time = time.time()
        time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
        progress = tqdm(train_loader, desc='Epoch {:0>3d}'.format(epoch))
        for batch_input, label_input, _ in progress:
            acoustic_input, acoustic_length = batch_input[0]
            acoustic_input = acoustic_input.cuda()
            acoustic_length = acoustic_length.cuda()
            semantic_input, semantic_length = batch_input[1]
            semantic_input = semantic_input.cuda()
            semantic_length = semantic_length.cuda()
            align_input = batch_input[2].cuda()
            
            if config['loss']['name'] == 'CTC':
                label_length = (semantic_length * config['loss']['ratio']).long()
                label_length = torch.where(label_length > 1, label_length, 1)
                label_input = label_input[:,None].repeat(1,semantic_input.size(1))
            label_input = label_input.cuda()
            
            model.zero_grad()
            logits = model(acoustic_input,acoustic_length,
                           semantic_input,semantic_length,
                           align_input,)

            if config['loss']['name'] == 'BCE':
                loss = loss_function(logits, label_input.long())
            else:
                loss = loss_function(logits.permute(1,0,2), label_input.long(), 
                                     semantic_length, label_length)
            
            epoch_train_loss.append(loss)
            
            loss.backward()
            optimizer.step()

            count += 1
            acc_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
            progress.set_description("Epoch {:0>3d} - Loss {:.4f}".format(epoch, acc_train_loss))


        model.eval()
        pred_y, true_y = [], []
        with torch.no_grad():
            time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
            for batch_input, label_input, _ in valid_loader:
                acoustic_input, acoustic_length = batch_input[0]
                acoustic_input = acoustic_input.cuda()
                acoustic_length = acoustic_length.cuda()
                semantic_input, semantic_length = batch_input[1]
                semantic_input = semantic_input.cuda()
                semantic_length = semantic_length.cuda()
                align_input = batch_input[2].cuda()

                true_y.extend(list(label_input.numpy()))

                logits = model(
                    acoustic_input,acoustic_length,
                    semantic_input,semantic_length,
                    align_input,
                )
                
                if config['loss']['name'] == 'CTC':
                    logits = torch.mean(logits[:,:,:4], dim=1)

                prediction = torch.argmax(logits, axis=1)
                label_outputs = prediction.cpu().detach().numpy().astype(int)
                
                pred_y.extend(list(label_outputs))

        key_metric, report_metric = evaluate_metrics(pred_y, true_y)

        epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print('Valid Metric: {} - Train Loss: {:.3f}'.format(
            ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in report_metric.items()]),
            epoch_train_loss))

        if key_metric > best_metric:
            best_metric, best_epoch = key_metric, epoch
            print('Better Metric found on dev, calculate performance on Test')
            pred_y, true_y = [], []
            with torch.no_grad():
                time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
                for batch_input, label_input, _ in valid_loader:
                    acoustic_input, acoustic_length = batch_input[0]
                    acoustic_input = acoustic_input.cuda()
                    acoustic_length = acoustic_length.cuda()
                    semantic_input, semantic_length = batch_input[1]
                    semantic_input = semantic_input.cuda()
                    semantic_length = semantic_length.cuda()
                    align_input = batch_input[2].cuda()

                    true_y.extend(list(label_input.numpy()))

                    logits = model(
                        acoustic_input,acoustic_length,
                        semantic_input,semantic_length,
                        align_input,
                    )
                    
                    if config['loss']['name'] == 'CTC':
                        logits = torch.mean(logits[:,:,:4], dim=1)
                    
                    prediction = torch.argmax(logits, axis=1)
                    label_outputs = prediction.cpu().detach().numpy().astype(int)

                    pred_y.extend(list(label_outputs))        
            
            _, save_metric = evaluate_metrics(pred_y, true_y)
            print("Test Metric: {}".format(
                ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])
            ))

    print("End. Best epoch {:03d}: {}".format(best_epoch, ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])))
    return save_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help='configuration file path')
    parser.add_argument("--epochs", type=int, default=20, help="training epoches")
    parser.add_argument("--save_path", type=str, default=None, help="report or ckpt save path")
    
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    report_result = []

    data_root = '/dataset/RAVDESS/base/'
    
    data_source = ['Fold0-wav.pkl','Fold1-wav.pkl','Fold2-wav.pkl','Fold3-wav.pkl','Fold4-wav.pkl']
    
    for i in range(5):
        valid_path = os.path.join(data_root, data_source[i])
        valid_data = pickle.load(open(valid_path,'rb'))
        valid_data = [(os.path.join(data_root,x[0]),x[1],os.path.join(data_root,x[2]),x[3]) for x in valid_data]

        train_path = [os.path.join(data_root,x) for x in data_source[:i]+data_source[i+1:]]
        train_data = list(reduce(lambda a,b: a+b, [pickle.load(open(x,'rb')) for x in train_path]))
        train_data = [(os.path.join(data_root,x[0]),x[1],os.path.join(data_root,x[2]),x[3]) for x in train_data]

        report_metric = run(args, config, train_data, valid_data)
        report_result.append(report_metric)

    
    os.makedirs(args.save_path, exist_ok=True)
    pickle.dump(report_result, open(os.path.join(args.save_path, 'metric_report.pkl'),'wb'))
