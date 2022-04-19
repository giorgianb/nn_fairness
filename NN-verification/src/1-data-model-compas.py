import numpy as np
import random
import pickle
import os
from os.path import join as os_join
from functools import partial
from scipy.special import expit

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from tqdm import tqdm as tqdm
import json

res_path = '../results'
cache_path = '../cache'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.y_2year = y[:,1] # 2 year recid 
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


def collate_fn_flip(batch, flip_index):
    # flip the feature (race)
    X = [  _[0] for _ in batch] #  # float tensor
    y = [  _[1] for _ in batch] # long tensor
    y_2year = [  _[2] for _ in batch]  # long tensor
    
    # invert 4'th col
    for i in range(len(X)):
        if X[i][flip_index] == 0:
            X[i][flip_index] = 1
        else:
            X[i][flip_index] = 0
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
        


def main(random_seed, is_race_permute, is_sex_permute, is_sex_race_both_permute, 
            is_random_weight, 
            is_fair_train_remove_sym_diff,
            is_fair_train_flip_duplicate_sym_diff,
            is_fair_train_iter_flip_duplicate_sym_diff,
            model_config):
    RS = np.random.RandomState(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # read data from cache
    cache_file_path = os_join(cache_path,f'np-compas-data-rs={random_seed}.pkl')
    with open(cache_file_path, 'rb') as f:
        data_dict = pickle.load(f)
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_dev = data_dict["X_dev"]
    y_dev = data_dict["y_dev"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    input_shape = X_train.shape[1]
    
    # age_feat, edu_feat, hours_per_week_feat, sex_feat, race_white_black_feat, country_is_native_feat, occupation_managerial_feat, occupation_is_gov_feat
    if is_race_permute:
        # input_feature_list = [np_age, np_prior, np_race, np_sex]
        print ('permute race attribute in a stratify manner')
        reference_col_ind = [3]
        permute_col_ind = [2] #
        stratify_permute_row_inplace(X_train, reference_col_ind, permute_col_ind, RS)
    
    # permute gender
    if is_sex_permute:
        # permute the protected features inplace
        print ('permute sex attribute in a stratify manner')
        reference_col_ind = [2] 
        permute_col_ind = [3] # permute 3 features wrt 4-8
        stratify_permute_row_inplace(X_train, reference_col_ind, permute_col_ind, RS)
        
    if is_sex_race_both_permute:
        print ('permute both sex and race attributes')
        RS.shuffle(X_train[:,[2,3]])
        

        
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
    model, train_stats, dev_stats, test_stats = train_loop(RS, train_dataset, dev_dataset, test_dataset, max_epoch, train_bs, eval_bs, lr=lr, input_shape = input_shape, is_random_weight = is_random_weight, model_config = model_config, model = None )
    
    # save model, with dev/test results
    model_save_dir = os_join(res_path, f'compas-model_config-{model_config}-max_epoch={max_epoch}-train_bs={train_bs}-random_seed={random_seed}-is_random_weight-{is_random_weight}-race_permute={is_race_permute}-sex_permute={is_sex_permute}-both_sex_race_permute={is_sex_race_both_permute}')
    print (f'saving to : {model_save_dir}')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save(model, model_save_dir, train_stats, dev_stats, test_stats, input_shape, model_config)
    
def main_based_on_sym_diff(random_seed, is_race_permute, is_sex_permute, is_sex_race_both_permute, 
            is_random_weight, 
            is_fair_train_remove_sym_diff,
            is_fair_train_flip_duplicate_sym_diff,
            is_fair_train_iter_flip_duplicate_sym_diff,
            model_config):
    RS = np.random.RandomState(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # read data from cache
    cache_file_path = os_join(cache_path,f'np-compas-data-rs={random_seed}.pkl')
    with open(cache_file_path, 'rb') as f:
        data_dict = pickle.load(f)
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_dev = data_dict["X_dev"]
    y_dev = data_dict["y_dev"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    input_shape = X_train.shape[1]
    
    # age_feat, edu_feat, hours_per_week_feat, sex_feat, race_white_black_feat, country_is_native_feat, occupation_managerial_feat, occupation_is_gov_feat
    # if is_race_permute:
    #     # input_feature_list = [age_feat, edu_feat, hours_per_week_feat, sex_feat, race_feat]
    #     print ('permute race attribute in a stratify manner')
    #     reference_col_ind = [3]
    #     permute_col_ind = [4] #
    #     stratify_permute_row_inplace(X_train, reference_col_ind, permute_col_ind, RS)
    
    # # permute gender
    # if is_sex_permute:
    #     # permute the protected features inplace
    #     print ('permute sex attribute in a stratify manner')
    #     reference_col_ind = [4] 
    #     permute_col_ind = [3] # permute 3 features wrt 4-8
    #     stratify_permute_row_inplace(X_train, reference_col_ind, permute_col_ind, RS)
        
    # if is_sex_race_both_permute:
    #     print ('permute both sex and race attributes')
    #     RS.shuffle(X_train[:,[3,4]])

    print ('Train feature/label shape:',X_train.shape, y_train.shape)
    print ('Dev. feature/label shape:',X_dev.shape, y_dev.shape)
    print ('Test feature/label shape:',X_test.shape, y_test.shape)
    train_dataset = SimpleDataset(X_train, y_train)
    dev_dataset = SimpleDataset(X_dev, y_dev)
    test_dataset = SimpleDataset(X_test, y_test)
    
    original_train_data = X_train.shape[0]
    # training loop
    max_epoch = 10
    train_bs = 32
    eval_bs = 128
    lr = 0.01
    K = 20
    flip_index = 2 #index for race_white_black_feat #TODO change to race
    print ("flip_feature_index=",2)
    # iteration
    model = None # need to init model
    for _ in range(K):
        # print ('dataset shape', len(train_dataset))
        if is_fair_train_remove_sym_diff is True:
            model = None
        if is_fair_train_flip_duplicate_sym_diff is True:
            model = None 
        model, train_stats, dev_stats, test_stats = train_loop(RS, train_dataset, dev_dataset, test_dataset, max_epoch, train_bs, eval_bs, lr=lr, input_shape = input_shape, is_random_weight = is_random_weight, model_config = model_config , model = model)
        # save model, with dev/test results
        model_save_dir = os_join(res_path, f'compas-model_config-{model_config}-max_epoch={max_epoch}-train_bs={train_bs}-random_seed={random_seed}-is_random_weight-{is_random_weight}-race_permute={is_race_permute}-sex_permute={is_sex_permute}'+
                                    f'-both_sex_race_permute={is_sex_race_both_permute}-is_fair_rmv_sym_diff={is_fair_train_remove_sym_diff}-is_fair_copy_sym_diff={is_fair_train_flip_duplicate_sym_diff}-is_fair_iter_copy_sym_diff={is_fair_train_iter_flip_duplicate_sym_diff}')
        print (f'saving to : {model_save_dir}')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        model_save_iter(model, model_save_dir, train_stats, dev_stats, test_stats, input_shape, _, train_dataset, model_config)
        
        # do training_set 
        if is_fair_train_remove_sym_diff is True:
            # flip gender, make prediction, see if the result is different.
            # if different, remove it, and reset the model to train
            changed_indices_bool = get_sym_diff_dataset_idx(flip_index, train_dataset, model)
            keep_index = (~changed_indices_bool).nonzero()[0]
            print (keep_index.shape)
            # remove from train data--- only keep those without flip prediction.
            X_train = X_train[keep_index,:]
            y_train = y_train[keep_index,:]
            assert y_train.shape[0]>0
            train_dataset = SimpleDataset(X_train, y_train)
            print ('train_set after',len(train_dataset))
        if is_fair_train_flip_duplicate_sym_diff is True or is_fair_train_iter_flip_duplicate_sym_diff is True:
            # flip gender, make prediction, see if different.
            # if different, duplicate it, and reset/incrementally train the  model 
            changed_indices_bool = get_sym_diff_dataset_idx(flip_index, train_dataset, model)
            changed_indices = (changed_indices_bool).nonzero()[0]
            aug_X_train = []
            aug_y_train = []
            
            for _ in changed_indices:
                _x = X_train[_]
                if _x[flip_index] == 1 :
                    _x[flip_index] =0
                else:
                    _x[flip_index] =1
                _y = y_train[_]
                aug_X_train.append(_x)
                aug_y_train.append(_y)
            aug_X_train = np.array(aug_X_train)
            aug_y_train = np.array(aug_y_train)
            
            assert aug_y_train.shape[0]>0
            
            X_train = np.vstack([X_train,aug_X_train]) 
            y_train = np.vstack([y_train,aug_y_train])
            
            assert X_train.shape[0]<=original_train_data*3 
            
            # print (X_train.shape, y_train.shape)
            train_dataset = SimpleDataset(X_train, y_train)
            # print ('train_set after',len(train_dataset) )

def get_sym_diff_dataset_idx(flip_index, train_dataset, model):
    _y_pred = []
    model.eval()
    # genuine dataset
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=32, shuffle=False , 
                                                num_workers=1, collate_fn = collate_fn )
    for x, y, y_2year in train_loader:
        x = x.to(device)
        y = y.to(device)
        dev_loss, y_pred = model.loss(x, y)
        _y_pred.append(y_pred.detach().cpu().numpy().reshape(-1))
    _y_pred = expit(np.hstack(_y_pred))
    
    # flipped dataset
    _y_pred_flip = []
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=32, shuffle=False , 
                                                num_workers=1, collate_fn = partial(collate_fn_flip, flip_index = flip_index) )
    for x, y, y_2year in train_loader:
        x = x.to(device)
        y = y.to(device)
        dev_loss, y_pred = model.loss(x, y)
        _y_pred_flip.append(y_pred.detach().cpu().numpy().reshape(-1))
    _y_pred_flip = expit(np.hstack(_y_pred_flip))
    # print (_y_pred.shape, _y_pred_flip.shape)
    # print (_y_pred[:5], _y_pred_flip[:5])
    
    # find the indices with the highest differece
    # round -> threshold by 0.5
    changed_ind = (np.round(_y_pred,0) - np.round(_y_pred_flip,0))!=0
    return changed_ind


def train_loop(RS, train_dataset, dev_dataset, test_dataset, max_epoch, train_bs, eval_bs, lr = 0.01, input_shape = 10, is_random_weight = False, model_config = 'small', model = None):
    
    # let's use cpu parameter
    if model_config == 'small' and model is None:
        model= SimpleNet_3_8(input_shape)
    if model_config == 'medium' and model is None:
        model= SimpleNet_4_8(input_shape)
    # if model_config == 'big':
    #     model= SimpleNet_3_8(input_shape)
    if is_random_weight: # random weight.
        print ('random weights')
        return model, {}, {},{}

    early_stopping = 3 # train at least 3 epochs.
    # let' use adam 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    # get dataloader
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=train_bs, shuffle=False , num_workers=1, collate_fn = collate_fn )
    dev_loader = torch.utils.data.DataLoader( dev_dataset, batch_size=eval_bs, shuffle=False, num_workers=1, collate_fn = collate_fn)
    test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=eval_bs, shuffle=False, num_workers=1, collate_fn = collate_fn)
    print ('dataloader #batch:', len(train_loader))
    dev_loss_history = []
    dev_auc_history = []
    train_loss_history = []
    
    for epoch in tqdm(range(max_epoch)):
        # epoch check validation loss
        dev_mean_loss, dev_auc = eval_model(model, dev_loader)
        print (f'epoch: {epoch} \t dev-loss: {dev_mean_loss} \t dev-auc: {dev_auc}')

        # early stop s
        if len(dev_loss_history)>0 and dev_mean_loss>dev_loss_history[-1] and epoch > early_stopping: 
            print ("early break")
            break
        else:
            dev_loss_history.append(dev_mean_loss)
            dev_auc_history.append(dev_auc)

        # train
        model.train()
        _train_loss = []
        for x, y, y_2year in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss, y_pred = model.loss(x, y)
            loss.backward()
            optimizer.step()
            _train_loss.append(loss)
        _tensor_train_loss = torch.tensor(_train_loss, dtype = torch.float)
        train_mean_loss = _tensor_train_loss.mean().detach().cpu().numpy().reshape(-1).item() 
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
        x = x.to(device)
        y = y.to(device)
        dev_loss, y_pred = model.loss(x, y)
        _loss.append(dev_loss)
        _y_pred.append(y_pred.detach().cpu().numpy().reshape(-1))
        _y.append(y.detach().cpu().numpy().reshape(-1))
    _tensor_loss = torch.tensor(_loss, dtype = torch.float)
    dev_mean_loss = _tensor_loss.mean().detach().cpu().numpy().reshape(-1).item() 
    _y = np.hstack(_y)
    _y_pred = np.hstack(_y_pred)
    dev_auc = eval_model_metric(_y_pred, _y)
    return dev_mean_loss, dev_auc

def eval_model_metric(y_pred, y):
    """ eval model performance, y_pred is expected to be the logits before sigmoid"""    
    y_pred = sigmoid(y_pred)
    auc = roc_auc_score(y, y_pred)
    return auc

def model_save(model, path, train_stats, dev_stats, test_stats, input_shape, model_config):
    """save model with assigned path"""
    model.eval() 
    # traditional save
    torch.save(model.state_dict(), os_join(path, "model.ckpt"))

    if model_config == 'small' :
        model_cpu= SimpleNet_3_8(input_shape)
    if model_config == 'medium':
        model_cpu= SimpleNet_4_8(input_shape)
    model_cpu.load_state_dict(torch.load(os_join(path, "model.ckpt"), map_location=torch.device('cpu')))

    # onnx
    dummy_input = torch.randn(1, input_shape, requires_grad=True) # input 
    torch.onnx.export(
        model_cpu,         # model being run
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

def model_save_iter(model, path, train_stats, dev_stats, test_stats, input_shape, K, train_dataset, model_config):
    """save model with assigned path"""
    model.eval() 
    # traditional save
    torch.save(model.state_dict(), os_join(path, f'model-iter-{K}.ckpt'))

    if model_config == 'small':
        model_cpu= SimpleNet_3_8(input_shape)
    if model_config == 'medium':
        model_cpu= SimpleNet_4_8(input_shape)
    model_cpu.load_state_dict(torch.load(os_join(path, f'model-iter-{K}.ckpt'), map_location=torch.device('cpu')))

    # onnx
    dummy_input = torch.randn(1, input_shape, requires_grad=True)# input 
    torch.onnx.export(
        model_cpu,         # model being run
        dummy_input,       # model input (or a tuple for multiple inputs)
        os_join(path, f'model-iter-{K}.onnx'),       # where to save the model
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
    
    with open (os_join(path, f'res-stats-iter-{K}.json'), 'w') as f:
        output_dict = {
            "train": train_stats,
            "num_train": len(train_dataset),
            "dev": dev_stats,
            "test": test_stats,
        }
        json_str = json.dumps(output_dict)
        f.write(json_str)


if __name__ == '__main__':
    # training 
    
    model_configs = ['small', 'medium']  #['medium'] 
    num_random_seed = 5
    
    
    for model_config in model_configs:
        # normal model
        is_race_permute = False 
        is_sex_permute =  False
        is_sex_race_both_permute = False
        is_random_weight = False
        is_fair_train_remove_sym_diff = False
        is_fair_train_flip_duplicate_sym_diff = False
        is_fair_train_iter_flip_duplicate_sym_diff = False
        for randseed in range(num_random_seed): # repeat for 3 times
            main(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, 
                is_random_weight, 
                is_fair_train_remove_sym_diff,
                is_fair_train_flip_duplicate_sym_diff,
                is_fair_train_iter_flip_duplicate_sym_diff,
                model_config)
        torch.cuda.empty_cache()
        
        # random model
        is_race_permute = False 
        is_sex_permute =  False
        is_sex_race_both_permute = False
        is_random_weight = True
        is_fair_train_remove_sym_diff = False
        is_fair_train_flip_duplicate_sym_diff = False
        is_fair_train_iter_flip_duplicate_sym_diff = False
        for randseed in range(num_random_seed): # repeat for 3 times
            main(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, 
                is_random_weight, 
                is_fair_train_remove_sym_diff,
                is_fair_train_flip_duplicate_sym_diff,
                is_fair_train_iter_flip_duplicate_sym_diff,
                model_config)
        torch.cuda.empty_cache()
        
        # race-permute only model
        is_race_permute = True 
        is_sex_permute =  False
        is_sex_race_both_permute = False
        is_random_weight = False
        is_fair_train_remove_sym_diff = False
        is_fair_train_flip_duplicate_sym_diff = False
        is_fair_train_iter_flip_duplicate_sym_diff = False
        for randseed in range(num_random_seed): # repeat for 3 times
            main(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, 
                is_random_weight, 
                is_fair_train_remove_sym_diff,
                is_fair_train_flip_duplicate_sym_diff,
                is_fair_train_iter_flip_duplicate_sym_diff,
                model_config)
        torch.cuda.empty_cache()
        
        # sex-permute only model
        is_race_permute = False 
        is_sex_permute =  True
        is_sex_race_both_permute = False
        is_random_weight = False
        is_fair_train_remove_sym_diff = False
        is_fair_train_flip_duplicate_sym_diff = False
        is_fair_train_iter_flip_duplicate_sym_diff = False
        for randseed in range(num_random_seed): # repeat for 3 times
            main(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, 
                is_random_weight, 
                is_fair_train_remove_sym_diff,
                is_fair_train_flip_duplicate_sym_diff,
                is_fair_train_iter_flip_duplicate_sym_diff,
                model_config)
        torch.cuda.empty_cache()
        
        # both-sex-race-permute model
        is_race_permute = False 
        is_sex_permute =  False
        is_sex_race_both_permute = True
        is_random_weight = False
        is_fair_train_remove_sym_diff = False
        is_fair_train_flip_duplicate_sym_diff = False
        is_fair_train_iter_flip_duplicate_sym_diff = False
        for randseed in range(num_random_seed): # repeat for 3 times
            main(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, 
                is_random_weight, 
                is_fair_train_remove_sym_diff,
                is_fair_train_flip_duplicate_sym_diff,
                is_fair_train_iter_flip_duplicate_sym_diff,
                model_config)
        torch.cuda.empty_cache()
        
        # based on sym diff
        # remove sym diff and re-train
        is_race_permute = False 
        is_sex_permute =  False
        is_sex_race_both_permute = False
        is_random_weight = False
        is_fair_train_remove_sym_diff = True
        is_fair_train_flip_duplicate_sym_diff = False
        is_fair_train_iter_flip_duplicate_sym_diff = False
        for randseed in range(num_random_seed): # repeat for 3 times
            try:
                main_based_on_sym_diff(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, 
                    is_random_weight, 
                    is_fair_train_remove_sym_diff,
                    is_fair_train_flip_duplicate_sym_diff,
                    is_fair_train_iter_flip_duplicate_sym_diff,
                    model_config)
            except Exception as e:
                print (e)
        torch.cuda.empty_cache()
        
        # based on sym diff
        # duplicate sym diff and re-train
        is_race_permute = False 
        is_sex_permute =  False
        is_sex_race_both_permute = False
        is_random_weight = False
        is_fair_train_remove_sym_diff = False
        is_fair_train_flip_duplicate_sym_diff = True
        is_fair_train_iter_flip_duplicate_sym_diff = False
        for randseed in range(num_random_seed): # repeat for 3 times
            try:
                main_based_on_sym_diff(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, 
                    is_random_weight, 
                    is_fair_train_remove_sym_diff,
                    is_fair_train_flip_duplicate_sym_diff,
                    is_fair_train_iter_flip_duplicate_sym_diff,
                    model_config)
            except Exception as e:
                print (e)
        torch.cuda.empty_cache()

        # based on sym diff
        # duplicate sym diff and incrementally train
        # is_race_permute = False 
        # is_sex_permute =  False
        # is_sex_race_both_permute = False
        # is_random_weight = False
        # is_fair_train_remove_sym_diff = False
        # is_fair_train_flip_duplicate_sym_diff = False
        # is_fair_train_iter_flip_duplicate_sym_diff = True
        # for randseed in range(num_random_seed): # repeat for 3 times
        #     main_based_on_sym_diff(randseed, is_race_permute, is_sex_permute, is_sex_race_both_permute, 
        #         is_random_weight, 
        #         is_fair_train_remove_sym_diff,
        #         is_fair_train_flip_duplicate_sym_diff,
        #         is_fair_train_iter_flip_duplicate_sym_diff,
        #         model_config)