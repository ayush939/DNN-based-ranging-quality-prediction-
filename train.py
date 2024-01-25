import argparse
import collections
import torch
import numpy as np
import nvidia_smi
import nvidia_smi
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchmetrics import Accuracy

from data_loader.custom_dataset import CustomCIRDataset, CustomCrossVal
import data_loader.custom_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer, Trainer_multitask
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')
    #dataPath =  Path.home() / config["data_loader"]["args"]["data_dir"]

    dataset_name = config["dataset"]

    # setup data_loader instances
     
    train_dataset = CustomCIRDataset(dataset=dataset_name, 
                                     data=0, 
                                     scale=config["preprocess"]["scale"],
                                     split_type = config["preprocess"]["split_type"],
                                     split_ratio= config["preprocess"]["split_ratio"],
                                     )

    val_dataset = CustomCIRDataset(dataset=dataset_name, 
                                   data=1, 
                                   scale=config["preprocess"]["scale"],  
                                   split_type = config["preprocess"]["split_type"],
                                   split_ratio= config["preprocess"]["split_ratio"],
                                   mean=0, 
                                   std=1)
    
    print(len(train_dataset), len(val_dataset))

    #train_dataset = CustomCrossVal(data_dir=str(dataPath), val = [6,7], validation_mode=False, scale=False)
    #val_dataset = CustomCrossVal(data_dir=str(dataPath), val=[6,7], validation_mode=True, scale=False)
    
    
    weight_sampler = None
    if  config["preprocess"]["sampling"]:
       # oversampling 
        #weights = [1/2988, 1/5315]
        weights = [1/1000, 1/6000]
        
        w_class = []
        for t in train_dataset.error:
            if 0.00<= t < 0.50:
                w_class.append(weights[1])
            else: 
                w_class.append(weights[0])
        
        w_class= np.array(w_class)
        weight_sampler = WeightedRandomSampler(weights=torch.from_numpy(w_class), num_samples=int(len(train_dataset)), replacement=True)
       

    train_dataloader = DataLoader(train_dataset, batch_size=config["preprocess"]["batch_size"], sampler=weight_sampler, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["preprocess"]["batch_size"], shuffle=config["preprocess"]["shuffle"])


    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = [getattr(module_loss, loss) for loss in config["loss"]]
    #criterion.weight = torch.tensor([2.16, 0.65])

    metrics = [getattr(module_metric, met) for met in config['metrics']]
    #metrics = [Accuracy(task="binary", num_classes=2)]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    #lr_scheduler = None

    if config["mode"] == "multi-task":
        trainer = Trainer_multitask(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_dataloader,
                      valid_data_loader=val_dataloader,
                      lr_scheduler=lr_scheduler)
    else:
        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=train_dataloader,
                        valid_data_loader=val_dataloader,
                        lr_scheduler=lr_scheduler)

    trainer.train()
    


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
