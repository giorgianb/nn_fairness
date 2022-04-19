import pandas as pd 
import numpy as np 
import pickle
from os.path import join as os_join
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = "serif"
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', markersize=2)


def clean_data(file_path):
    """Filter out dirty data, keep binary feature for gender and race"""
    df_two_year = pd.read_csv(file_path)
    df_two_year = df_two_year[  
                                (df_two_year['days_b_screening_arrest']<=30) & \
                                (df_two_year['days_b_screening_arrest']>=-30) & \
                                (df_two_year['is_recid']!= -1) & \
                                (df_two_year['c_charge_degree'] != 'O') & \
                                (df_two_year['score_text'] != 'N/A') &\
                                ((df_two_year['race'] == 'African-American') | (df_two_year['race'] == 'Caucasian') )
                                ]
    selected_cols = [ "age", "race", "sex", "priors_count", "decile_score", "two_year_recid"]
    df_clean =  df_two_year.loc[:,selected_cols ]
    return df_clean

def feature_gen(df_raw_data):
    """genereate numerical features from raw data and normalize"""
    # x = "age", "race", "sex", "priors_count", 
    # y = "decile_score", "two_year_recid"

    np_age = df_raw_data['age'].to_numpy().reshape(-1,1)
    np_prior = df_raw_data['priors_count'].to_numpy().reshape(-1,1)
    np_race = np.array([ race == 'African-American' for race in df_raw_data['race'].tolist()]).reshape(-1,1)
    np_sex = np.array([ sex == 'Male' for sex in df_raw_data['sex'].tolist()]).reshape(-1,1)
    np_score = np.array([ score >=5 for score in df_raw_data['decile_score'].tolist()]).reshape(-1,1).astype(int)
    #np_score = df_raw_data['decile_score'].to_numpy().reshape(-1,1)
    np_truth = df_raw_data['two_year_recid'].to_numpy().reshape(-1,1)
    np_feature_mat = np.hstack((np_age, np_prior, np_race, np_sex, np_score, np_truth))
    return np_feature_mat


def split_set(np_data,train_ratio, RS):
    """Split feature-label matrix into train/dev/test"""
    X = np_data[:,:4].astype(float)
    Y = np_data[:,4:].astype(int)
    X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=(1.0-train_ratio), random_state=RS)
    X_dev, X_test, y_dev, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=RS)
    
    # use training data to normalize all data
    min_max_scaler_age = preprocessing.MinMaxScaler()
    min_max_scaler_age.fit(X_train[:,0].reshape(-1,1))
    X_train[:,0] = min_max_scaler_age.transform( X_train[:,0].reshape(-1,1)).reshape(-1)
    X_dev[:,0] = min_max_scaler_age.transform( X_dev[:,0].reshape(-1,1)).reshape(-1)
    X_test[:,0] = min_max_scaler_age.transform( X_test[:,0].reshape(-1,1)).reshape(-1)
    
    min_max_scaler_prior = preprocessing.MinMaxScaler()
    min_max_scaler_prior.fit(X_train[:,1].reshape(-1,1))
    X_train[:,1] = min_max_scaler_prior.transform( X_train[:,1].reshape(-1,1)).reshape(-1)
    X_dev[:,1] = min_max_scaler_prior.transform( X_dev[:,1].reshape(-1,1)).reshape(-1)
    X_test[:,1] = min_max_scaler_prior.transform( X_test[:,1].reshape(-1,1)).reshape(-1)

    return X_train, y_train, X_dev, y_dev, X_test, y_test

def plot_2D_scatter(x,y,title, ax):
    ax.scatter(x =x, y =y, s = 0.2, alpha = 0.5)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_title(title)
    ax.set_xlabel("normalized age")
    ax.set_ylabel("normalized prior")


def main(random_seed = 621):
    RS = np.random.RandomState(random_seed)
    # clean data
    data_path = '/Users/xingzhiguo/Documents/git_project/NN-verification/data'
    res_path = '/Users/xingzhiguo/Documents/git_project/NN-verification/results'
    cache_path = '/Users/xingzhiguo/Documents/git_project/NN-verification/cache'
    
    if os.path.exists(cache_path) is not True:
        os.makedirs(cache_path)
    
    if os.path.exists(res_path) is not True:
        os.makedirs(res_path)


    file_path =os_join(data_path, 'compas-scores-two-years-violent.csv')
    df_clean = clean_data(file_path)

    # construct feature and label
    np_feature_mat = feature_gen(df_clean)
    print (f'np_feature_mat.shape:{np_feature_mat.shape}')
    
    # train/dev/test set
    train_ratio = 0.7 # dev and test share the rest
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_set(np_feature_mat,train_ratio, RS)
    
    print ('Train feature/label shape:',X_train.shape, y_train.shape)
    print ('Dev. feature/label shape:',X_dev.shape, y_dev.shape)
    print ('Test feature/label shape:',X_test.shape, y_test.shape)
    
    # plot joint distribution of age/prior 
    fig, axes = plt.subplots(1,3, figsize=(8,3))
    plot_2D_scatter(X_train[:,0],X_train[:,1],"train", axes[0])
    plot_2D_scatter(X_dev[:,0],X_dev[:,1],"dev", axes[1])
    plot_2D_scatter(X_test[:,0],X_test[:,1],"test", axes[2])
    fig.tight_layout()
    fig.savefig(os_join(res_path, f'fig-feature-2d-distribution-rs={random_seed}.pdf'), format='pdf',dpi=300)
    
    # save results for nn prediction
    data_output = {
        "X_train":X_train,
        "y_train":y_train,
        "X_dev":X_dev,
        "y_dev":y_dev,
        "X_test":X_test,
        "y_test":y_test,
    }
    cache_file_path = os_join(cache_path,f'np-compas-data-rs={random_seed}.pkl')
    with open (cache_file_path,'wb') as f:
        pickle.dump(data_output,f)
    print (f'saved data matrix to {cache_file_path}')

if __name__ == '__main__':
    for randseed in range(10):
        main(random_seed = randseed)
