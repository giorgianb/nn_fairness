import numpy as np
import random
import pickle
import os
from os.path import join as os_join

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from tqdm import tqdm as tqdm
import json

res_path = '/Users/xingzhiguo/Documents/git_project/NN-verification/results'
cache_path = '/Users/xingzhiguo/Documents/git_project/NN-verification/cache'


def sigmoid(z):
    return 1/(1 + np.exp(-z))


class SimpleNet_3_8(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet_3_8, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.out = nn.Linear(8, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # init weights ?

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x) # raw digits
        return x
    
    def loss(self, X, y):
        y_pred = self.forward(X).ravel()
        return self.criterion(y_pred, y), y_pred

class SimpleNet_4_8(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet_4_8, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 8)
        self.out = nn.Linear(8, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # init weights ?

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x) # raw digits
        return x
    
    def loss(self, X, y):
        y_pred = self.forward(X).ravel()
        return self.criterion(y_pred, y), y_pred


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y[:,0]
        self.y_2year = y[:,0]
        self.size = self.X.shape[0]
        assert self.size == self.y.shape[0]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return [self.X[index,:], self.y[index], self.y_2year[index]]


def collate_fn(batch):
    X = [  _[0] for _ in batch] #  # float tensor
    y = [  _[1] for _ in batch] # long tensor
    y_2year = [  _[2] for _ in batch]  # long tensor
    
    X = torch.tensor(np.array(X), dtype = torch.float)
    y = torch.tensor(np.array(y), dtype = torch.float)
    y_2year = torch.tensor(np.array(y_2year), dtype = torch.float)

    #y = torch.LongTensor(np.array(y))
    #y_2year = torch.LongTensor(np.array(y_2year))
    return (X, y, y_2year)


def stratify_permute_row_inplace(a, reference_col_ind, permute_col_ind, RS):
    ref_col_vals = a[:,reference_col_ind]
    unique_val_in_ref_col = np.unique(ref_col_vals, axis=0)

    for _val in unique_val_in_ref_col:
        row_group_index = np.where( (ref_col_vals==_val).all(1), 1, 0).nonzero()[0]
        #print (0, _val, row_group_index)
        _permute_col_of_row = a[np.ix_(row_group_index, permute_col_ind)]
        #print (1,_permute_col_of_row)
        permuted_permute_col_of_row = RS.permutation(_permute_col_of_row)
        #print (2,permuted_permute_col_of_row)
        a[np.ix_(row_group_index, permute_col_ind)] = permuted_permute_col_of_row


def main(random_seed, is_race_permute, is_sex_permute, is_sex_race_both_permute, is_random_weight, model_config):
    RS = np.random.RandomState(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # read data from cache
    cache_file_path = os_join(cache_path,f'np-adult-data-rs={random_seed}.pkl')
    with open(cache_file_path, 'rb') as f:
        data_dict = pickle.load(f)
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_dev = data_dict["X_dev"]
    y_dev = data_dict["y_dev"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    input_shape = X_train.shape[1]
    
    if is_race_permute:
        # input_feature_list = [age_feat, edu_feat, hours_per_week_feat, sex_feat, race_feat]
        print ('permute race attribute in a stratify manner')
        reference_col_ind = [3]
        permute_col_ind = [4,5,6,7,8] # permute 4-8 features wrt 3
        stratify_permute_row_inplace(X_train, reference_col_ind, permute_col_ind, RS)
        
    # permute gender
    if is_sex_permute:
        # permute the protected features inplace
        print ('permute sex attribute in a stratify manner')
        reference_col_ind = [4,5,6,7,8] 
        permute_col_ind = [3] # permute 3 features wrt 4-8
        stratify_permute_row_inplace(X_train, reference_col_ind, permute_col_ind, RS)
    if is_sex_race_both_permute:
        print ('permute both sex and race attributes')
        RS.shuffle(X_train[:,3:])

    print ('Train feature/label shape:',X_train.shape, y_train.shape)
    print ('Dev. feature/label shape:',X_dev.shape, y_dev.shape)
    print ('Test feature/label shape:',X_test.shape, y_test.shape)
    train_dataset = SimpleDataset(X_train, y_train)
    dev_dataset = SimpleDataset(X_dev, y_dev)
    test_dataset = SimpleDataset(X_test, y_test)
    
    # training loop
    max_epoch = 10
    train_bs = 32
    eval_bs = 128
    lr = 0.01
    model, train_stats, dev_stats, test_stats = train_loop(RS, train_dataset, dev_dataset, test_dataset, max_epoch, train_bs, eval_bs, lr=lr, input_shape = input_shape, is_random_weight= is_random_weight, model_config = model_config )
    
    # save model, with dev/test results
    model_save_dir = os_join(res_path, f'adult-model_config-{model_config}-max_epoch={max_epoch}-train_bs={train_bs}-random_seed={random_seed}-is_random_weight-{is_random_weight}-race_permute={is_race_permute}-sex_permute={is_sex_permute}-both_sex_race_permute={is_sex_race_both_permute}')
    print (f'saving to : {model_save_dir}')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save(model, model_save_dir, train_stats, dev_stats, test_stats, input_shape)
    

def train_loop(RS, train_dataset, dev_dataset, test_dataset, max_epoch, train_bs, eval_bs, lr = 0.01, input_shape = 10, is_random_weight = False, model_config = 'small'):
    # let's use cpu parameter
    if model_config == 'small':
        model= SimpleNet_3_8(input_shape)
    if model_config == 'medium':
        model= SimpleNet_4_8(input_shape)
    # if model_config == 'big':
    #     model= SimpleNet_3_8(input_shape)
    if is_random_weight: # random weight.
        return model, {}, {},{}

    # let' use adam 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # get dataloader
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=train_bs, shuffle=True, num_workers=1, collate_fn = collate_fn )
    dev_loader = torch.utils.data.DataLoader( dev_dataset, batch_size=eval_bs, shuffle=False, num_workers=1, collate_fn = collate_fn)
    test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=eval_bs, shuffle=False, num_workers=1, collate_fn = collate_fn)

    dev_loss_history = []
    dev_auc_history = []
    train_loss_history = []
    
    for epoch in tqdm(range(max_epoch)):
        # epoch check validation loss
        dev_mean_loss, dev_auc = eval_model(model, dev_loader)
        print (f'epoch: {epoch} \t dev-loss: {dev_mean_loss} \t dev-auc: {dev_auc}')

        # early stop s
        if len(dev_loss_history)>0 and dev_mean_loss>dev_loss_history[-1]: 
            print ("early break")
            break
        else:
            dev_loss_history.append(dev_mean_loss)
            dev_auc_history.append(dev_auc)

        # train
        model.train()
        _train_loss = []
        for x, y, y_2year in train_loader:
            optimizer.zero_grad()
            loss, y_pred = model.loss(x, y)
            loss.backward()
            optimizer.step()
            _train_loss.append(loss)
        _tensor_train_loss = torch.tensor(_train_loss, dtype = torch.float)
        train_mean_loss = _tensor_train_loss.mean().detach().numpy().reshape(-1).item() 
        train_loss_history.append(train_mean_loss)
        print (f'epoch: {epoch} \t train-loss: {train_mean_loss}')
    
    test_auc_history = []
    test_loss_history = []
    test_mean_loss, test_auc = eval_model(model, test_loader)
    print (f'epoch: {epoch} \t test-loss: {test_mean_loss} \t test-auc: {test_auc}')
    test_loss_history.append(test_mean_loss)
    test_auc_history.append(test_auc)
    
    train_stats = {
        'loss_history': train_loss_history,
    }
    dev_stats = {
        'loss_history': dev_loss_history,
        'auc_history':dev_auc_history
    }
    test_stats = {
        'loss_history': test_loss_history,
        'auc_history':test_auc_history
    }
    
    return model, train_stats, dev_stats, test_stats

def eval_model(model, d_loader):
    _loss = []
    _y_pred = []
    _y = []
    model.eval()
    for x, y, y_2year in d_loader:
        dev_loss, y_pred = model.loss(x, y)
        _loss.append(dev_loss)
        _y_pred.append(y_pred.detach().numpy().reshape(-1))
        _y.append(y.detach().numpy().reshape(-1))
    _tensor_loss = torch.tensor(_loss, dtype = torch.float)
    dev_mean_loss = _tensor_loss.mean().detach().numpy().reshape(-1).item() 
    _y = np.hstack(_y)
    _y_pred = np.hstack(_y_pred)
    dev_auc = eval_model_metric(_y_pred, _y)
    return dev_mean_loss, dev_auc

def eval_model_metric(y_pred, y):
    """ eval model performance, y_pred is expected to be the logits before sigmoid"""    
    y_pred = sigmoid(y_pred)
    auc = roc_auc_score(y, y_pred)
    return auc

def model_save(model, path, train_stats, dev_stats, test_stats, input_shape):
    """save model with assigned path"""
    model.eval() 
    # traditional save
    torch.save(model.state_dict(), os_join(path, "model.ckpt"))

    # onnx
    dummy_input = torch.randn(1, input_shape, requires_grad=True) # input 
    torch.onnx.export(
        model,         # model being run
        dummy_input,       # model input (or a tuple for multiple inputs)
        os_join(path, "model.onnx"),       # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,    # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['modelInput'],   # the model's input names
        output_names=['modelOutput'],  # the model's output names
        dynamic_axes={
                    'modelInput': {0: 'batch_size'},    # variable length axes
                    'modelOutput': {0: 'batch_size'}}
        )
    print('Model has been converted to ONNX')
    
    with open (os_join(path, 'res-stats.json'), 'w') as f:
        output_dict = {
            "train": train_stats,
            "dev": dev_stats,
            "test": test_stats,
        }
        json_str = json.dumps(output_dict)
        f.write(json_str)

# def load_model_and_eval():
#     """load the model and play with it"""
#     max_epoch = 10
#     train_bs = 32
#     random_seed = 0

#     model = SimpleNet()
#     model_save_dir = os_join(res_path, f'adult-max_epoch={max_epoch}-train_bs={train_bs}-random_seed={random_seed}/model.ckpt')
#     model.load_state_dict(torch.load(model_save_dir))
#     model.eval()
    
#     cache_file_path = os_join(cache_path,f'np-adult-data-rs={random_seed}.pkl')
#     with open(cache_file_path, 'rb') as f:
#         data_dict = pickle.load(f)
#     X_train = data_dict["X_train"]
#     y_train = data_dict["y_train"]
#     X_dev = data_dict["X_dev"]
#     y_dev = data_dict["y_dev"]
#     X_test = data_dict["X_test"]
#     y_test = data_dict["y_test"]
#     print ('Train feature/label shape:',X_train.shape, y_train.shape)
#     print ('Dev. feature/label shape:',X_dev.shape, y_dev.shape)
#     print ('Test feature/label shape:',X_test.shape, y_test.shape)
    
#     # get dataset
#     train_dataset = SimpleDataset(X_train, y_train)
#     dev_dataset = SimpleDataset(X_dev, y_dev)
#     test_dataset = SimpleDataset(X_test, y_test)
    
#     # get dataloader
#     train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=train_bs, shuffle=True, num_workers=1, collate_fn = collate_fn )
#     # dev_loader = torch.utils.data.DataLoader( dev_dataset, batch_size=train_bs, shuffle=False, num_workers=1, collate_fn = collate_fn)
#     # test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=train_bs, shuffle=False, num_workers=1, collate_fn = collate_fn)

#     _loss = []
#     _y_pred = []
#     _y = []
#     _x = []
#     model.eval()
#     for x, y, y_2year in train_loader:
#         dev_loss, y_pred = model.loss(x, y)
#         _loss.append(dev_loss)
#         _x.append(x.detach().numpy())
#         _y_pred.append(y_pred.detach().numpy().reshape(-1))
#         _y.append(y.detach().numpy().reshape(-1))
#     _tensor_loss = torch.tensor(_loss, dtype = torch.float)
#     dev_mean_loss = _tensor_loss.mean().detach().numpy().reshape(-1).item() 
#     _y = np.hstack(_y)
#     _y_pred = np.hstack(_y_pred)
#     _x = np.vstack(_x)
#     dev_auc = eval_model_metric(_y_pred, _y)
#     print ('input:')
#     print (_x[:5])
#     print ('raw-output:')
#     print (_y_pred[:5])
#     print ('Sigmoid(raw-output):')
#     print (sigmoid(_y_pred[:5]))
#     print ('Ground truth:')
#     print (_y[:5])
    


if __name__ == '__main__':
    # training 
    
    model_configs = ['small','medium']
    num_random_seed = 1
    
    for model_config in model_configs:
        # normal model
        is_race_permute = False 
        is_sex_permute =  False
        is_sex_race_both_permute = False
        is_random_weight = False
        for randseed in range(num_random_seed): # repeat for 3 times
            main(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, is_random_weight, model_config)
        
        # random model
        is_race_permute = False 
        is_sex_permute =  False
        is_sex_race_both_permute = False
        is_random_weight = True
        for randseed in range(num_random_seed): # repeat for 3 times
            main(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, is_random_weight, model_config)

        # race-permute only model
        is_race_permute = True 
        is_sex_permute =  False
        is_sex_race_both_permute = False
        is_random_weight = False
        for randseed in range(num_random_seed): # repeat for 3 times
            main(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, is_random_weight, model_config)

        # sex-permute only model
        is_race_permute = False 
        is_sex_permute =  True
        is_sex_race_both_permute = False
        is_random_weight = False
        for randseed in range(num_random_seed): # repeat for 3 times
            main(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, is_random_weight, model_config)
            
        # both-sex-race-permute model
        is_race_permute = False 
        is_sex_permute =  False
        is_sex_race_both_permute = True
        is_random_weight = False
        for randseed in range(num_random_seed): # repeat for 3 times
            main(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, is_random_weight, model_config)

