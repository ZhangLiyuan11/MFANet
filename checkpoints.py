import torch
from tqdm import tqdm

from network import *
from dataloader import ModelDataset
from run import collate_fn
from torch.utils.data import DataLoader
from tools import *

path = ''

def load_chechpoint(path):
    model = MulModel(fea_dim=128, dropout=0.1)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model.cuda()


def get_dataloader():
    dataset_test = ModelDataset(dataset='fakesv', path_vid='vid_time3_test.txt')
    test_dataloader = DataLoader(dataset_test, batch_size=16,
                                 num_workers=0,
                                 pin_memory=True,
                                 shuffle=False,
                                 collate_fn=collate_fn)
    return test_dataloader


def test():
    model = load_chechpoint(path)
    test_dataloader = get_dataloader()
    tpred = []
    tlabel = []
    for batch in tqdm(test_dataloader):
        batch_data = batch
        for k, v in batch_data.items():
            batch_data[k] = v.cuda()
        label = batch_data['label']
        with torch.set_grad_enabled(False):
            outputs,clip_text_out,clap_text_out,fea = model(**batch_data)
            _,preds = torch.max(outputs, 1)
        tlabel.extend(label.detach().cpu().numpy().tolist())
        tpred.extend(preds.detach().cpu().numpy().tolist())
    get_confusionmatrix_fnd(tpred,tlabel)
    results = metrics(tlabel, tpred)
    print(results)


if __name__ == '__main__':
    test()
