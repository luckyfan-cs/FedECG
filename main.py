from models import  CNN, RNNModel, RNNAttentionModel
from utils import   args_parser
from worker import  WorkerBase
import numpy as np
import torch
from datasets import  get_dataloader
import copy
import  pandas as pd


def server_robust_agg(args, grad):  ## server aggregation
    grad_in = np.array(grad).reshape((args.num_workers, -1)).mean(axis=0)
    return grad_in.tolist()


class ClearDenseClient(WorkerBase):
    def __init__(self, client_id, model,  train_iter,  test_iter, optimizer, device,  args, scheduler):
        super(ClearDenseClient, self).__init__(model = model, train_iter =train_iter,  test_iter=test_iter, args =args, optimizer = optimizer, device =device)
        self.client_id = client_id
        self.args = args
        self.scheduler = scheduler

    def update(self):
        pass


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self



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

    client = []
    for i in range(args.num_workers):
        local_model = copy.deepcopy(model)
        local_model = local_model.to(device)

        optimizer = torch.optim.AdamW(local_model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)


        train_loader = train_dataloaders[i]
        test_loader  = test_dataloaders[i]

        client.append(ClearDenseClient(client_id=i, model=local_model,  train_iter=train_loader,
                                       test_iter=test_loader,optimizer=optimizer, device=device, args=args,
                                       scheduler=scheduler))

    print(client)
    weight_history = []
    for epoch in range(args.epochs):


        train_logs_list = []
        val_logs_list = []
        for i in range(args.num_workers):
            print("i",i)
            result = client[i].gnn_train()
            metrics, train_log, val_log = result
            train_logs_list.append(train_log)
            val_logs_list.append(val_log)
            client[i].scheduler.step()




        weights = []
        for i in range(args.num_workers):
            weights.append(client[i].get_weights())
            weight_history.append(client[i].get_weights())

        # Aggregation in the server to get the global model

        result = server_robust_agg(args, weights)

        for i in range(args.num_workers):
            client[i].set_weights(weights=result)
            client[i].upgrade()
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