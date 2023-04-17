from models import  CNN, RNNModel, RNNAttentionModel
from utils import   args_parser
from worker import  WorkerBase
import numpy as np
import torch
from datasets import  get_dataloader
import copy
import  pandas as pd
import torch.nn as nn

from worker import  train, evaluate
import  random
def main():
    args = args_parser()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device_id)
    if args.model == "RNNAttentionModel":
        model = RNNAttentionModel(1, 64, 'lstm', False)
    elif args.model == "RNNModel":
        model = RNNModel(1, 64, 'lstm', True)
    elif args.model == "CNN":
        model = CNN(num_classes=5, hid_size=128)
    else:
        raise NameError
    train_dataloaders, test_dataloaders = get_dataloader(args)


    model_list = []
    optimizer_list = []
    scheduler_list = []
    for i in range(args.num_workers):

        local_model = copy.deepcopy(model)
        local_model = local_model.to(device)

        optimizer = torch.optim.AdamW(local_model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)
        model_list.append(local_model)
        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)
    severe_model = model
    criterion = nn.CrossEntropyLoss()
    train_df_logs = pd.DataFrame()
    val_df_logs = pd.DataFrame()
    for epoch in range(args.epochs):


        train_logs_list = []
        val_logs_list = []
        for i in range(args.num_workers):
            metric,train_log = train(model_id = i,model=model_list[i],train_iter=train_dataloaders[i],device=device,optimizer=optimizer_list[i],criterion=criterion,args=args,train_df_logs=train_df_logs)
            _,val_log = evaluate(model_id = i,model=model_list[i], test_iter=test_dataloaders[i], device=device,criterion= criterion, args =args, val_df_logs = val_df_logs)
            train_logs_list.append(train_log)
            val_logs_list.append(val_log)
            scheduler_list[i].step()






        # Aggregation in the server to get the global model
        # Server Aggregation
        Sub_model_list = random.sample(model_list, args.num_sample_submodels)
        for param_tensor in Sub_model_list[0].state_dict():
            avg = (sum(c.state_dict()[param_tensor] for c in Sub_model_list)) / len(Sub_model_list)
            severe_model.state_dict()[param_tensor].copy_(avg)
            for cl in model_list:
                cl.state_dict()[param_tensor].copy_(avg)

    for i in range(args.num_workers):
        train_logs = train_logs_list[i]
        train_logs.columns = ["train_"+ colname for colname in train_logs.columns]
        val_logs = val_logs_list[i]
        val_logs.columns = ["val_"+ colname for colname in val_logs.columns]

        logs = pd.concat([train_logs,val_logs], axis=1)
        logs.reset_index(drop=True, inplace=True)
        logs = logs.loc[:, [
            'train_loss', 'val_loss',
            'train_accuracy', 'val_accuracy',
            'train_f1', 'val_f1',
            'train_precision', 'val_precision',
            'train_recall', 'val_recall']
                                         ]
        logs.head()
        logs.to_csv(args.log_save_path+'Fed_model-{}-log-client-{}.csv'.format(args.model,i), index=False)
if __name__ == '__main__':
    main()