import argparse
import math
import numpy as np
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
from torch.utils.data import Dataset, DataLoader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from pathlib import Path
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryROC
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from data_loader.custom_dataset import CustomCIRDataset, CustomCrossVal
import matplotlib.pyplot as plt

def main(config):
    logger = config.get_logger('test')
    dataset_name = config["dataset"]
    testPath =  Path.home() / "/speech/dbwork/mul/spielwiese3/students/demittaa/UWB-localization/localization"

    # setup data_loader instances

    """
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )
    """
   

    test_dataset = CustomCIRDataset(dataset=dataset_name, 
                                   data=2, 
                                   scale=config["preprocess"]["scale"],  
                                   split_type = config["preprocess"]["split_type"],
                                   split_ratio= config["preprocess"]["split_ratio"],
                                   mean=0, 
                                   std=1)
    
    data_loader = DataLoader(test_dataset, batch_size=config["preprocess"]["batch_size"], shuffle=False)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn =[getattr(module_loss, loss) for loss in config["loss"]][1]
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    if config["mode"] == "reg":
        mae = MeanAbsoluteError().to(device)
    else:
        confmat = ConfusionMatrix(task="binary").to(device)
        confmat.reset()
        roc = BinaryROC().to(device)

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)
    

    with torch.no_grad():
        for i, (data, target, error) in enumerate(tqdm(data_loader)):

            data, target, error = data.to(device), target.to(device), error.to(device)
            data = torch.unsqueeze(data, dim=1)
            output = model(data.float())

            # computing loss, metrics on test set
            loss = loss_fn(output.float(), error.float())
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for j, metric in enumerate(metric_fns):
                total_metrics[j] += metric(output, error) * batch_size
            
            if config["mode"] == "cls":
                confmat.update(output, target)
                roc.update(output, target.int())
            else:
                if i == 0:
                    ae = np.array((output.cpu().numpy() - error.cpu().numpy()))
                    out = np.array(output.cpu().numpy())
                    gt = np.array(error.cpu().numpy())

                else:
                    ae = np.vstack(((output.cpu().numpy() - error.cpu().numpy()), ae))
                    out = np.vstack((output.cpu().numpy(), out))
                    gt = np.vstack((error.cpu().numpy(), gt))
                    #print(ae.shape)
                mae.update(output, error)

    if config["mode"] == "cls":                
        print(confmat.compute())
        fig_ , ax_ =  confmat.plot()
        fig_.savefig("confusion_mat.jpg")

        fig1_, ax1_ = roc.plot(score=True)
        fig1_.savefig("ROC.jpg")
    else:
        
        ae = np.absolute(ae.flatten())
        output = ae
        print('MAE val. after ',np.mean(np.absolute(out.flatten())))
        print(np.std(ae.flatten()), np.mean(ae.flatten()))
        #print(mae.compute())
        
        # crete bins
        b1_gt, b1_out, b2_out, b2_gt, b3_out, b3_gt, b4_out, b4_gt = [], [], [], [], [], [], [], []
        labels = gt.flatten()
        print('MAE val set before:', np.mean(np.absolute(labels)))
        print(len(labels))
        quantiles = np.quantile(labels, np.array([0.00, 0.25, 0.5, 0.75, 1.00]))


        quantiles = [0.00, 1.25, 2]
        print(quantiles)
        for i, elem in enumerate(labels):
            if   elem <= quantiles[0]:
                b1_gt.append(elem)
                b1_out.append(output[i])
            elif quantiles[0] <= elem < quantiles[1]:
                b2_gt.append(elem)
                b2_out.append(output[i])
            elif quantiles[1] <= elem < quantiles[2]:
                b3_gt.append(elem)
                b3_out.append(output[i])
            elif quantiles[2] <= elem:
                b4_gt.append(elem)
                b4_out.append(output[i])
        
        y = [np.mean(b1_gt), np.mean(b2_gt), np.mean(b3_gt), np.mean(b4_gt)]
        k = [np.mean(b1_out), np.mean(b2_out), np.mean(b3_out), np.mean(b4_out)]
        l = [len(b1_out), len(b2_out), len(b3_out), len(b4_out)]

        N = 4
        ind = np.arange(N)  # the x locations for the groups
        width = 0.27       # the width of the bars

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)


        #rects1 = ax.bar(ind, y, width, color='r')
        rects2 = ax1.bar(ind+width, l, width, color='g')
        rects1 = ax2.bar(ind+width, k, width, color='g')
        ax1.set_xticks(ind+width)
        ax2.set_xticks(ind+width)
        #ax.set_xticklabels( ('<q0={:0.2f}'.format(quantiles[0]), 'q0-q2 ({:0.2f} , {:0.2f})'.format(quantiles[0], quantiles[1]), 'q2-q4({:0.2f}, {:0.2f})'.format(quantiles[1], quantiles[2]), '>q4 = {:0.2f}'.format(quantiles[2])) )
        #ax.set_xticklabels(('<0cm', '0-50cm', '50-100cm', '>100cm'))
        ax1.set_xticklabels(('<0.00m', '0.00-1.25m', '1.25-4.00m', '>4.00m'))
        ax2.set_xticklabels(('<0.00m', '0.00-1.25m', '1.25-4.00m', '>4.00m'))
        fig1.savefig('bar_plot_size.jpg')
        fig2.savefig('bar_plot.jpg')

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    
    config = ConfigParser.from_args(args)
    main(config)
