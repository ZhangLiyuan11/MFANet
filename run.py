from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from network import MulModel
from dataloader import *
from Trainer import Trainer

def _init_fn(worker_id):
    np.random.seed(2025)

class Run():
    def __init__(self,config):
        self.model_name = config['model_name']
        self.dataset = config['dataset']
        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.epoch_stop = config['epoch_stop']
        self.seed = config['seed']
        self.device = config['device']
        self.lr = config['lr']
        self.lambd=config['lambd']
        self.save_param_dir = config['path_param']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']


    def get_dataloader_temporal(self, dataset):
        dataset_train = ModelDataset(dataset=dataset, path_vid='vid_time3_train.txt')
        dataset_val = ModelDataset(dataset=dataset, path_vid='vid_time3_val.txt')
        dataset_test = ModelDataset(dataset=dataset, path_vid='vid_time3_test.txt')

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
        test_dataloader=DataLoader(dataset_test, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
 
        dataloaders =  dict(zip(['train', 'val', 'test'],[train_dataloader, val_dataloader, test_dataloader]))
        return dataloaders

    def get_Alignment_dataloader(self,dataset):
        dataset = AlignmentDataset(dataset=dataset,path_vid='vid_time3_train.txt')

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      pin_memory=True,
                                      shuffle=True,
                                      worker_init_fn=_init_fn,
                                      collate_fn=AlignmentDataset_collate_fn)

        return dataloader

    def get_model(self):
        if self.model_name == 'Model':
                self.model = MulModel(fea_dim=128, dropout=self.dropout)
        return self.model

    def main(self):
        self.model = self.get_model()
        Alignment_dataloader = self.get_Alignment_dataloader(dataset=self.dataset)
        dataloaders = self.get_dataloader_temporal(dataset=self.dataset)
        trainer = Trainer(model=self.model, device = self.device, lr = self.lr, dataloaders = dataloaders,
                          Alignment_dataloader = Alignment_dataloader, epoches = self.epoches,
                          dropout = self.dropout, weight_decay = self.weight_decay, model_name = self.model_name,
                          save_param_path = self.save_param_dir+"/"+self.dataset+"/",)
        result = trainer.train()
        return result
