import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class ModelDataset(Dataset):
    def __init__(self, dataset, path_vid, datamode='title+ocr'):
        self.vid = []
        self.dataset = dataset

        if self.dataset == 'fakesv':
            self.data_complete = pd.read_json('data/fakesv/data_complete.json', orient='records', dtype=False,
                                              lines=True)
            self.data_complete = self.data_complete[self.data_complete['annotation'] != '辟谣']
            with open('data/fakesv/vids/' + path_vid, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
        else:
            self.data_complete = pd.read_json('data/fakett/data.json', orient='records', dtype=False, lines=True)
            with open('data/fakett/vids/' + path_vid, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())

        self.clip_text_fea_file = 'data/' + dataset + '/text_clip\\'
        self.clip_img_fea_file = 'data/' + dataset + '/video_clip\\'
        self.clap_text_fea_file = 'data/' + dataset + '/text_clap\\'
        self.clap_audio_fea_file = 'data/' + dataset + '/audio_clap\\'

        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid)
        self.data.sort_values('video_id', ascending=True, inplace=True)
        self.data.reset_index(inplace=True)

        self.datamode = datamode
        
    def __len__(self):
        return self.data.shape[0]
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # label
        if self.dataset == 'fakesv':
            label = 0 if item['annotation'] == '真' else 1
        else:
            label = 0 if item['annotation'] == 'real' else 1
        label = torch.tensor(label)

        # text
        clip_text_fea_path = os.path.join(self.clip_text_fea_file, vid + '.pkl')
        clap_text_fea_path = os.path.join(self.clap_text_fea_file, vid + '.pkl')
        clip_text_fea = torch.load(clip_text_fea_path)
        clap_text_fea = torch.load(clap_text_fea_path)

        # audio
        clap_audio_fea_path = os.path.join(self.clap_audio_fea_file, vid + '.pkl')
        clap_audio_fea = torch.load(clap_audio_fea_path)

        # video
        clip_video_fea_path = os.path.join(self.clip_img_fea_file, vid + '.pkl')
        clip_video_fea = torch.load(clip_video_fea_path)

        return {
            'label': label,
            'clip_text_fea': clip_text_fea,
            'clap_text_fea': clap_text_fea,
            'clap_audio_fea': clap_audio_fea,
            'clip_video_fea': clip_video_fea,
        }

def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def collate_fn(batch):
    clip_text_fea = [torch.squeeze(item['clip_text_fea']) for item in batch]
    clap_text_fea = [torch.squeeze(item['clap_text_fea']) for item in batch]
    clap_audio_fea = [item['clap_audio_fea'] for item in batch]
    clip_video_fea = [torch.mean(item['clip_video_fea'],dim=0) for item in batch]

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'clip_text_fea': torch.stack(clip_text_fea).squeeze(dim=1).float(),
        'clap_text_fea': torch.stack(clap_text_fea).squeeze(dim=1),
        'clap_audio_fea': torch.stack(clap_audio_fea).squeeze(dim=1),
        'clip_video_fea': torch.stack(clip_video_fea).squeeze(dim=1).float(),
    }


class AlignmentDataset(Dataset):
    def __init__(self, dataset, path_vid):
        self.vid = []
        self.dataset = dataset

        if self.dataset == 'fakesv':
            self.data_complete = pd.read_json('data/fakesv/data_complete.json', orient='records', dtype=False,
                                              lines=True)
            self.data_complete = self.data_complete[self.data_complete['annotation'] != '辟谣']
        else:
            self.data_complete = pd.read_json('data/fakett/data.json', orient='records', dtype=False, lines=True)

        self.clip_text_fea_file = 'data/' + dataset + '/text_clip\\'
        self.clap_text_fea_file = 'data/' + dataset + '/text_clap\\'

    def __len__(self):
        return self.data_complete.shape[0]

    def __getitem__(self, idx):
        item = self.data_complete.iloc[idx]
        vid = item['video_id']

        # text
        clip_text_fea_path = os.path.join(self.clip_text_fea_file, vid + '.pkl')
        clap_text_fea_path = os.path.join(self.clap_text_fea_file, vid + '.pkl')
        clip_text_fea = torch.load(clip_text_fea_path)
        clap_text_fea = torch.load(clap_text_fea_path)

        return {
            'clip_text_fea': clip_text_fea,
            'clap_text_fea': clap_text_fea,
        }


def AlignmentDataset_collate_fn(batch):
    clip_text_fea = [item['clip_text_fea'] for item in batch]
    clap_text_fea = [item['clap_text_fea'] for item in batch]

    return {
        'clip_text_fea': torch.stack(clip_text_fea).squeeze(dim=1).float(),
        'clap_text_fea': torch.stack(clap_text_fea).squeeze(dim=1),
    }