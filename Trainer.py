import copy
import os
import time
from network import *
import tqdm
from tqdm import tqdm
from tools import *
from zmq import device
from torch import nn

import torch
import torch.nn.functional as F
# InfoNce_loss
def info_nce_loss(embedding_model_1,embedding_model_2, temperature=0.07):
    batch_size = embedding_model_1.shape[0]
    similarity_matrix = torch.mm(embedding_model_1, embedding_model_2.t()) / temperature

    labels = torch.arange(batch_size).cuda()
    info_nce_loss = F.cross_entropy(similarity_matrix, labels)

    return info_nce_loss

class Trainer():
    def __init__(self,
                model,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 Alignment_dataloader,
                 weight_decay,
                 save_param_path,
                 epoches,
                 model_name,
                 save_threshold = 0.0, 
                 start_epoch = 0,
                 ):
        self.Alignment_model = Alignment().cuda()
        self.model = model
        self.device = device
        self.model_name = model_name

        self.dataloaders = dataloaders
        self.Alignment_dataloader = Alignment_dataloader
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.save_threshold = save_threshold

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path= save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
    
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        since = time.time()
        self.model.cuda()
        best_acc_val = 0.0
        best_epoch_val = 0
        is_earlystop = False

        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
            print('-' * 50)

            # 更新学习率
            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            # 创建优化器
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
            
            for phase in ['train', 'val', 'test']:
                if phase == 'train':
                    self.model.train()  
                else:
                    self.model.eval()   
                print('-' * 10)
                print (phase.upper())
                print('-' * 10)

                running_loss = 0.0 
                tpred = []
                tlabel = []
                Contrastive_loss_total = []

                for batch in tqdm(self.dataloaders[phase]):
                    batch_data=batch
                    # 把每个样本都放到gpu上
                    for k,v in batch_data.items():
                        batch_data[k]=v.cuda()
                    label = batch_data['label']

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs,clip_text_out,clap_text_out = self.model(**batch_data)
                        _, preds = torch.max(outputs, 1)
                        Contrastive_loss = info_nce_loss(clip_text_out, clap_text_out)
                        loss = self.criterion(outputs, label)
                        total_loss = loss + 0.5 * Contrastive_loss
                        # total_loss = loss
                        if phase == 'train':
                            total_loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            Contrastive_loss_total.append(Contrastive_loss.item())
                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)

                if phase == 'train':
                    print(f" Contrastive_loss: {np.mean(np.array(Contrastive_loss_total)):.4f}")

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print (results)
                if phase == 'test' and results['acc'] > best_acc_val:
                    best_acc_val = results['acc']
                    best_epoch_val = epoch + 1
                    if best_acc_val > self.save_threshold:
                        torch.save(self.model.state_dict(),
                                   self.save_param_path + "_test_epoch" + str(best_epoch_val) + "_{0:.4f}".format(
                                       best_acc_val))
                        print("saved " + self.save_param_path + "_test_epoch" + str(
                            best_epoch_val) + "_{0:.4f}".format(best_acc_val))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))

        print ("test result when using best model on val")
        return True

