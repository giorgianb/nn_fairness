#!/usr/bin/env python
# coding: utf-8

# In[1]:


from responsibly.dataset import AdultDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from os.path import join as os_join

pd.set_option('display.max_rows', 300)


# In[13]:





# In[15]:


adult_ds = AdultDataset()
adult_ds._validate()
sub_columns = ['age', 'education-num', 'race', 'sex','capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_per_year', 'occupation', 'workclass', 'dataset']
df_data = adult_ds.df.loc[:,sub_columns]
print (df_data.head(4))
df_country_gdp = pd.read_csv('../data/adult-dataset/gdp-pc.csv').loc[:, ['Country Name','1996']] # the dataset is from 1996
for col in ['race', 'sex', 'native_country']:
    print(df_data[col].value_counts())


# In[16]:


# adult_ds.df['occupation'].value_counts()
# adult_ds.df['occupation'].value_counts()


# In[17]:


def _normalize_country_name(x):
    x = " ".join(str.lower(x).split('-'))
    if x== "trinadad&tobago":
        x = 'Trinidad and Tobago'
    elif x == 'england' or x == 'scotland':
        x = 'United Kingdom'
    elif x == 'holand netherlands':
        x = 'Netherlands'
    elif x == 'hong':
        x ='Hong Kong SAR, China'
    elif x == 'laos':
        x ='Lao PDR'
    elif x == 'iran':
        x ='Iran, Islamic Rep.'
    elif x == 'outlying us(guam usvi etc)':
        x ='United States'
    return " ".join(str.lower(x).split('-'))

df_data['native_country'] = df_data['native_country'].apply(_normalize_country_name).to_list() 
df_country_gdp['Country Name'] = df_country_gdp['Country Name'].apply(_normalize_country_name).to_list()

aset = set( df_country_gdp.dropna()['Country Name'].to_list())
bset = set( df_data.loc[:,'native_country'].to_list())
dff_set = bset - aset
print ('countries not presented in world bank:',dff_set)
df_data = df_data[~df_data['native_country'].isin(dff_set)]

gdp_lut = df_country_gdp.dropna().set_index('Country Name').to_dict()['1996']

df_data['gdp_pc'] = [gdp_lut[country] for country in df_data['native_country'].to_list()]


# ## Feature constructor

# In[29]:


from sklearn import preprocessing

age_feat = df_data['age'].to_numpy().reshape(-1,1)
# print (age_feat.shape)
min_max_scaler_age = preprocessing.MinMaxScaler()
age_feat = min_max_scaler_age.fit_transform(age_feat)
print ('age_feat',age_feat.shape, min_max_scaler_age.data_min_, min_max_scaler_age.data_max_)


edu_feat = df_data['education-num'].to_numpy().reshape(-1,1)
# print (edu_feat.shape)
min_max_scaler_age = preprocessing.MinMaxScaler()
edu_feat = min_max_scaler_age.fit_transform(edu_feat)
print ('edu_feat',edu_feat.shape, min_max_scaler_age.data_min_, min_max_scaler_age.data_max_)


def _race_encoder(x):
    race_dict = {
        "White":0,
        "Black":1,
        "Asian-Pac-Islander":2,
        "Amer-Indian-Eskimo":3,
        "Other":4
    }
    _feat = np.zeros(len(race_dict.keys()))
    _feat[race_dict[x]] = 1
    return _feat
race_feat = [_race_encoder(_) for _ in df_data['race'].to_list()]
race_feat = np.array(race_feat)
print ('race_feat',race_feat.shape)



def _gender_encoder(x):
    gender_dict = {
        "Male":1,
        "Female":0,
    }
    return gender_dict[x]
sex_feat = [_gender_encoder(_) for _ in df_data['sex'].to_list()]
sex_feat = np.array(sex_feat).reshape(-1,1)
print ('sex_feat',sex_feat.shape)

capital_gain_feat = df_data['capital_gain'].to_numpy().reshape(-1,1)
# print (edu_feat.shape)
min_max_scaler_age = preprocessing.MinMaxScaler()
capital_gain_feat = min_max_scaler_age.fit_transform(capital_gain_feat)
print ('capital_gain_feat',capital_gain_feat.shape)

capital_loss_feat = df_data['capital_loss'].to_numpy().reshape(-1,1)
# print (edu_feat.shape)
min_max_scaler_age = preprocessing.MinMaxScaler()
capital_loss_feat = min_max_scaler_age.fit_transform(capital_loss_feat)
print ('capital_loss_feat',capital_loss_feat.shape)



hours_per_week_feat = df_data['hours_per_week'].to_numpy().reshape(-1,1)
# print (edu_feat.shape)
min_max_scaler_age = preprocessing.MinMaxScaler()
hours_per_week_feat = min_max_scaler_age.fit_transform(hours_per_week_feat)
print ('hours_per_week_feat',hours_per_week_feat.shape, min_max_scaler_age.data_min_, min_max_scaler_age.data_max_)



gdp_pc_feat = df_data['gdp_pc'].to_numpy().reshape(-1,1)
# print (edu_feat.shape)
min_max_scaler_age = preprocessing.MinMaxScaler()
gdp_pc_feat = min_max_scaler_age.fit_transform(gdp_pc_feat)
print ('gdp_pc_feat',gdp_pc_feat.shape, min_max_scaler_age.data_min_, min_max_scaler_age.data_max_)


def _income_encoder(x):
    gender_dict = {
        "<=50K":0,
        ">50K":1,
    }
    return gender_dict[x]
income_feat = [_income_encoder(_) for _ in df_data['income_per_year'].to_list()]
income_feat = np.array(income_feat).reshape(-1,1)
print ('income_feat',income_feat.shape)




def _white_black_encoder(x):
    race_dict = {
        "White":0,
        "Black":1,
        "Asian-Pac-Islander":-1,
        "Amer-Indian-Eskimo":-1,
        "Other":-1
    }
    _feat = race_dict[x]
    return _feat
race_white_black_feat = [_white_black_encoder(_) for _ in df_data['race'].to_list()]
race_white_black_feat = np.array(race_white_black_feat).reshape(-1,1)
print ('race_white_black_feat',race_white_black_feat.shape)




def _immigrant_encoder(x):
    if x == 'United States'.lower():
        return 1 
    else: 
        return 0
country_is_native_feat = [_immigrant_encoder(_) for _ in df_data['native_country'].to_list()]
country_is_native_feat = np.array(country_is_native_feat).reshape(-1,1)
print ('country_is_native_feat',country_is_native_feat.shape)



def _is_manager_encoder(x):
    # professional_occupation = {'Craft-repair', 'Prof-specialty', 'Sales', 
    #                            'Other-service', 'Machine-op-inspct', 
    #                            'Transport-moving', 'Handlers-cleaners',
    #                            'Farming-fishing', 'Tech-support', 'Protective-serv','Priv-house-serv','Armed-Forces'}
    managerial_occupation = {'Exec-managerial', 'Adm-clerical'}
    if x in managerial_occupation:
        return 1 
    else: 
        return 0
occupation_managerial_feat = [_is_manager_encoder(_) for _ in df_data['occupation'].to_list()]
occupation_managerial_feat = np.array(occupation_managerial_feat).reshape(-1,1)
print ('occupation_managerial_feat',occupation_managerial_feat.shape)



def _is_gov_employ_encoder(x):
    # professional_occupation = {'Craft-repair', 'Prof-specialty', 'Sales', 
    #                            'Other-service', 'Machine-op-inspct', 
    #                            'Transport-moving', 'Handlers-cleaners',
    #                            'Farming-fishing', 'Tech-support', 'Protective-serv','Priv-house-serv','Armed-Forces'}
    
    if 'gov' in x:
        return 1 
    else: 
        return 0
occupation_is_gov_feat = [_is_gov_employ_encoder(_) for _ in df_data['workclass'].to_list()]
occupation_is_gov_feat = np.array(occupation_is_gov_feat).reshape(-1,1)
print ('occupation_is_gov_feat',occupation_is_gov_feat.shape)




# In[ ]:





# In[7]:


def split_set(np_data,train_ratio, RS):
    """Split feature-label matrix into train/dev/test"""
    X = np_data[:,:-1].astype(float)
    Y = np_data[:,-1].astype(int).reshape(-1,1)
    X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=(1.0-train_ratio), random_state=RS)
    X_dev, X_test, y_dev, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=RS)
    return X_train, y_train, X_dev, y_dev, X_test, y_test

# input_feature_list = [age_feat, edu_feat, capital_gain_feat, capital_loss_feat ,sex_feat, hours_per_week_feat, gdp_pc_feat, race_feat]
input_feature_list = [age_feat, edu_feat, hours_per_week_feat, sex_feat, race_feat]
Y = income_feat
X = np.hstack(input_feature_list)

data_set = np.hstack([X,Y])
for random_seed in range(3):
    RS = np.random.RandomState(random_seed)
    # train/dev/test set
    train_ratio = 0.7 # dev and test share the rest
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_set(data_set,train_ratio, RS)
    print ('Train feature/label shape:',X_train.shape, y_train.shape)
    print ('Dev. feature/label shape:',X_dev.shape, y_dev.shape)
    print ('Test feature/label shape:',X_test.shape, y_test.shape)

    data_output = {
        "X_train":X_train,
        "y_train":y_train,
        "X_dev":X_dev,
        "y_dev":y_dev,
        "X_test":X_test,
        "y_test":y_test,
    }
    cache_path = '/home/giorgian/Documente/Fairness/NN-verification/cache'
    cache_file_path = os_join(cache_path,f'np-adult-data-rs={random_seed}.pkl')
    with open (cache_file_path,'wb') as f:
        pickle.dump(data_output,f)
    print (f'saved data matrix to {cache_file_path}')


# In[30]:


def split_set(np_data,train_ratio, RS):
    """Split feature-label matrix into train/dev/test"""
    X = np_data[:,:-1].astype(float)
    Y = np_data[:,-1].astype(int).reshape(-1,1)
    X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=(1.0-train_ratio), random_state=RS)
    X_dev, X_test, y_dev, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=RS)
    return X_train, y_train, X_dev, y_dev, X_test, y_test

# input_feature_list = [age_feat, edu_feat, capital_gain_feat, capital_loss_feat ,sex_feat, hours_per_week_feat, gdp_pc_feat, race_feat]
input_feature_list = [age_feat, edu_feat, hours_per_week_feat, sex_feat, race_white_black_feat, country_is_native_feat, occupation_managerial_feat, occupation_is_gov_feat]
print(f'len(input_feature_list): {len(input_feature_list)}')
Y = income_feat
X = np.hstack(input_feature_list)

# keep only white black 

selected_index = np.where(X[:,4]!=-1)[0]
X = X[selected_index,:]
Y = Y[selected_index,:]


data_set = np.hstack([X,Y])
for random_seed in range(3):
    RS = np.random.RandomState(random_seed)
    # train/dev/test set
    train_ratio = 0.7 # dev and test share the rest
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_set(data_set,train_ratio, RS)
    print ('Train feature/label shape:',X_train.shape, y_train.shape)
    print ('Dev. feature/label shape:',X_dev.shape, y_dev.shape)
    print ('Test feature/label shape:',X_test.shape, y_test.shape)

    data_output = {
        "X_train":X_train,
        "y_train":y_train,
        "X_dev":X_dev,
        "y_dev":y_dev,
        "X_test":X_test,
        "y_test":y_test,
    }
    cache_path = '/home/giorgian/Documente/Fairness/NN-verification/cache'
    #cache_file_path = os_join(cache_path,f'np-adult-data-v2-rs={random_seed}.pkl')
    cache_file_path = os_join(cache_path,f'test.pkl')
    with open (cache_file_path,'wb') as f:
        pickle.dump(data_output,f)
    print (f'saved data matrix to {cache_file_path}')


# In[12]:



def stratify_permute_row_inplace(a, reference_col_ind, permute_col_imd, RS):
    ref_col_vals = a[:,reference_col_ind]
    unique_val_in_ref_col = np.unique(ref_col_vals)

    for _val in unique_val_in_ref_col:
        row_group_index = np.where( (ref_col_vals==_val).all(1), 1, 0).nonzero()[0]
        #print (0, _val, row_group_index)
        _permute_col_of_row = a[np.ix_(row_group_index, permute_col_ind)]
        #print (1,_permute_col_of_row)
        permuted_permute_col_of_row = RS.permutation(_permute_col_of_row)
        #print (2,permuted_permute_col_of_row)
        a[np.ix_(row_group_index, permute_col_ind)] = permuted_permute_col_of_row
        
a = np.arange(16).reshape(4,4)
a[1,1] = 1
a[2,1] = 1

random_seed = 4
RS = np.random.RandomState(random_seed)

print (a)
reference_col_ind = []
permute_col_ind = np.array([0,1,2,3])

stratify_permute_row_inplace(a, reference_col_ind, permute_col_ind, RS)
print (a)


# In[25]:


a = np.arange(16).reshape(4,4)
a[1,1] = 1
a[2,1] = 1

a[:,1]==1


# In[27]:


np.where(a[:,1]==1)[0]


# In[31]:


a[:,[1,2]]


# In[ ]:




