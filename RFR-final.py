
# coding: utf-8

# # Contents:
# 
# 1. RFR1: Using everything - 70 features
# 2. RFR2: Without spectral features - 68 features
# 3. RFR3: Using both spectral and tempo features with feature selection
# 4. Evaluation of rfr perfoemance wrt the datasets.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()


# In[2]:


df = pd.read_csv('data-v4.csv', index_col=0)


# In[3]:


df.head(3)


# # RFR1

# In[4]:


data_tr1 = df.iloc[:264, 0:70]
data_ts1 = df.iloc[264:, 0:70]
valence_tr1 = df.iloc[:264, 70]
arousal_tr1 = df.iloc[:264, 71]
valence_ts1 = df.iloc[264:, 70]
arousal_ts1 = df.iloc[264:, 71]


# In[5]:


train1 = data_tr1.values
test1 = data_ts1.values


# In[6]:


rfr_params = {'min_impurity_split':[1e-1],
             'min_samples_split':[7],
             'verbose':[2],
             'max_depth':[3, 5, 7],
             'min_samples_leaf':[1],
             'n_estimators':[100, 1000]}
grid_search_rfr1_valence = GridSearchCV(rfr, rfr_params, cv = 10, scoring = 'neg_mean_squared_error')
grid_search_rfr1_valence.fit(train1, valence_tr1)


# In[7]:


grid_search_rfr1_valence.best_params_


# In[8]:


results1_valence = grid_search_rfr1_valence.cv_results_
valence_pred_tr1 = grid_search_rfr1_valence.predict(train1)
valence_pred_ts1 = grid_search_rfr1_valence.predict(test1)


# In[9]:


grid_search_rfr1_arousal = GridSearchCV(rfr, rfr_params, cv = 10, scoring = 'neg_mean_squared_error')
grid_search_rfr1_arousal.fit(train1, arousal_tr1)
results1_arousal = grid_search_rfr1_arousal.cv_results_
arousal_pred_tr1 = grid_search_rfr1_arousal.predict(train1)
arousal_pred_ts1 = grid_search_rfr1_arousal.predict(test1)


# # RFR2

# In[10]:


data_tr2 = df.iloc[:264, 0:68]
data_ts2 = df.iloc[264:, 0:68]

valence_tr2 = df.iloc[:264, 70]
arousal_tr2 = df.iloc[:264, 71]
arousal_ts2 = df.iloc[264:, 71]
valence_ts2 = df.iloc[264:, 70]


# In[11]:


data_tr2.head(4)


# In[12]:


valence_tr2.head(4)


# In[13]:


train2 = data_tr2.values
test2 = data_ts2.values


# In[14]:


grid_search_rfr2_valence = GridSearchCV(rfr, rfr_params, cv = 10, scoring = 'neg_mean_squared_error')
grid_search_rfr2_valence.fit(train2, valence_tr2)


# In[15]:


results2_valence = grid_search_rfr2_valence.cv_results_
valence_pred_tr2 = grid_search_rfr2_valence.predict(train2)
valence_pred_ts2 = grid_search_rfr2_valence.predict(test2)


# In[16]:


grid_search_rfr2_arousal = GridSearchCV(rfr, rfr_params, cv = 10, scoring = 'neg_mean_squared_error')
grid_search_rfr2_arousal.fit(train2, arousal_tr2)
results2_arousal = grid_search_rfr2_arousal.cv_results_
arousal_pred_tr2 = grid_search_rfr2_arousal.predict(train2)
arousal_pred_ts2 = grid_search_rfr2_arousal.predict(test2)


# # RFR3

# In[17]:


data3 = df.loc[:, ['Tempo Feature1',
 'Spectral Centroid',
 'ZCR',
 'Spectral Entropy',
 'CV.7',
 'Spectral Flux',
 'Spectral Rolloff',
 'MFCC.4',
 'MFCC.2',
 'CV.5',
 'Entropy of Energy',
 'MFCC.3',
 'CV.8',
 'Tempo Feature2',
 'MFCC.13',
 'Spectral Centroid_SD',
 'ZCR_SD',
 'Spectral Entropy_SD',
 'CV.7_SD',
 'Spectral Flux_SD',
 'Spectral Rolloff_SD',
 'MFCC.4_SD',
 'MFCC.2_SD',
 'CV.5_SD',
 'Entropy of Energy_SD',
 'MFCC.3_SD',
 'CV.8_SD',
 'MFCC.13_SD']]
data_tr3 = data3.iloc[:264, :]
data_ts3 = data3.iloc[264:, :]

valence_tr3 = df.iloc[:264, 70]
arousal_tr3 = df.iloc[:264, 71]
arousal_ts3 = df.iloc[264:, 71]
valence_ts3 = df.iloc[264:, 70]


# In[18]:


data_tr3.head(4)


# In[19]:


valence_tr3[0:4]


# In[20]:


train3 = data_tr3.values
test3 = data_ts3.values
grid_search_rfr3_valence = GridSearchCV(rfr, rfr_params, cv = 10, scoring = 'neg_mean_squared_error')
grid_search_rfr3_valence.fit(train3, valence_tr3)
results3_valence = grid_search_rfr3_valence.cv_results_
valence_pred_tr3 = grid_search_rfr3_valence.predict(train3)
valence_pred_ts3 = grid_search_rfr3_valence.predict(test3)


# In[21]:


grid_search_rfr3_arousal = GridSearchCV(rfr, rfr_params, cv = 10, scoring = 'neg_mean_squared_error')
grid_search_rfr3_arousal.fit(train3, arousal_tr3)
results3_arousal = grid_search_rfr3_arousal.cv_results_
arousal_pred_tr3 = grid_search_rfr3_arousal.predict(train3)
arousal_pred_ts3 = grid_search_rfr3_arousal.predict(test3)


# # Performance Eval

# In[22]:


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2


# In[23]:


#Arousal Error(mae) vs dataset 1,2,3 
print('1. Training Error :' + str(mae(arousal_tr1,arousal_pred_tr1)) + ' Test Error :' + str(mae(arousal_ts1,arousal_pred_ts1)))
print('2. Training Error :' + str(mae(arousal_tr2,arousal_pred_tr2)) + ' Test Error :' + str(mae(arousal_ts2,arousal_pred_ts2)))
print('3. Training Error :' + str(mae(arousal_tr3,arousal_pred_tr3)) + ' Test Error :' + str(mae(arousal_ts3,arousal_pred_ts3)))


# In[24]:


#Arousal Error(mse) vs dataset 1,2,3 
print('1. Training Error :' + str(mse(arousal_tr1,arousal_pred_tr1)) + ' Test Error :' + str(mse(arousal_ts1,arousal_pred_ts1)))
print('2. Training Error :' + str(mse(arousal_tr2,arousal_pred_tr2)) + ' Test Error :' + str(mse(arousal_ts2,arousal_pred_ts2)))
print('3. Training Error :' + str(mse(arousal_tr3,arousal_pred_tr3)) + ' Test Error :' + str(mse(arousal_ts3,arousal_pred_ts3)))


# In[25]:


#Arousal Error(r2) vs dataset 1,2,3 
print('1. Training r2 score :' + str(r2(arousal_tr1,arousal_pred_tr1)) + ' Test r2 score :' + str(r2(arousal_ts1,arousal_pred_ts1)))
print('2. Training r2 score :' + str(r2(arousal_tr2,arousal_pred_tr2)) + ' Test r2 score :' + str(r2(arousal_ts2,arousal_pred_ts2)))
print('3. Training r2 score :' + str(r2(arousal_tr3,arousal_pred_tr3)) + ' Test r2 score :' + str(r2(arousal_ts3,arousal_pred_ts3)))


# In[26]:


#Valence Error(mae) vs dataset 1,2,3 
print('1. Training Error :' + str(mae(valence_tr1,valence_pred_tr1)) + ' Test Error :' + str(mae(valence_ts1,valence_pred_ts1)))
print('2. Training Error :' + str(mae(valence_tr2,valence_pred_tr2)) + ' Test Error :' + str(mae(valence_ts2,valence_pred_ts2)))
print('3. Training Error :' + str(mae(valence_tr3,valence_pred_tr3)) + ' Test Error :' + str(mae(valence_ts3,valence_pred_ts3)))


# In[27]:


#Valence Error(mse) vs dataset 1,2,3 
print('1. Training Error :' + str(mse(valence_tr1,valence_pred_tr1)) + ' Test Error :' + str(mse(valence_ts1,valence_pred_ts1)))
print('2. Training Error :' + str(mse(valence_tr2,valence_pred_tr2)) + ' Test Error :' + str(mse(valence_ts2,valence_pred_ts2)))
print('3. Training Error :' + str(mse(valence_tr3,valence_pred_tr3)) + ' Test Error :' + str(mse(valence_ts3,valence_pred_ts3)))


# In[28]:


#Valence Error(r2) vs dataset 1,2,3 
print('1. Training Error :' + str(r2(valence_tr1,valence_pred_tr1)) + ' Test Error :' + str(r2(valence_ts1,valence_pred_ts1)))
print('2. Training Error :' + str(r2(valence_tr2,valence_pred_tr2)) + ' Test Error :' + str(r2(valence_ts2,valence_pred_ts2)))
print('3. Training Error :' + str(r2(valence_tr3,valence_pred_tr3)) + ' Test Error :' + str(r2(valence_ts3,valence_pred_ts3)))


# # Save the data for vizualization and model comparision

# In[33]:


#Arousal test data -- for viz
np.savetxt("Vizdata/RFR_arousal1_groundthruth-pred.csv", np.column_stack((arousal_ts1.values,arousal_pred_ts1)), delimiter=',', header="Ground truth value,Predicted value", comments="")
np.savetxt("Vizdata/RFR_arousal2_groundthruth-pred.csv", np.column_stack((arousal_ts1.values,arousal_pred_ts1)), delimiter=',', header="Ground truth value,Predicted value", comments="")
np.savetxt("Vizdata/RFR_arousal3_groundthruth-pred.csv", np.column_stack((arousal_ts1.values,arousal_pred_ts1)), delimiter=',', header="Ground truth value,Predicted value", comments="")
#Arousal test error -- for model comp
arousal_mse = np.column_stack((mse(arousal_ts1,arousal_pred_ts1),mse(arousal_ts2,arousal_pred_ts2),mse(arousal_ts3,arousal_pred_ts3)))
np.savetxt("Compdata/RFR_arousal_performance-mse.csv", arousal_mse, delimiter=',', header="MSE1,MSE2,MSE3", comments="")
arousal_mae = np.column_stack((mae(arousal_ts1,arousal_pred_ts1),mae(arousal_ts2,arousal_pred_ts2),mae(arousal_ts3,arousal_pred_ts3)))
np.savetxt("Compdata/RFR_arousal_performance-mae.csv", arousal_mae, delimiter=',', header="MAE1,MAE2,MAE3", comments="")
arousal_r2 = np.column_stack((r2(arousal_ts1,arousal_pred_ts1),r2(arousal_ts2,arousal_pred_ts2),r2(arousal_ts3,arousal_pred_ts3)))
np.savetxt("Compdata/RFR_arousal_performance-r2.csv", arousal_r2, delimiter=',', header="r2-1,r2-2,r2-3", comments="")


# In[34]:


#Valence test data -- for viz
np.savetxt("Vizdata/SVM_valence1_groundthruth-pred.csv", np.column_stack((valence_ts1.values,valence_pred_ts1)), delimiter=',', header="Ground truth value,Predicted value", comments="")
np.savetxt("Vizdata/SVM_valence2_groundthruth-pred.csv", np.column_stack((valence_ts1.values,valence_pred_ts1)), delimiter=',', header="Ground truth value,Predicted value", comments="")
np.savetxt("Vizdata/SVM_valence3_groundthruth-pred.csv", np.column_stack((valence_ts1.values,valence_pred_ts1)), delimiter=',', header="Ground truth value,Predicted value", comments="")
#Valence test error -- for model comp
valence_mse = np.column_stack((mse(valence_ts1,valence_pred_ts1),mse(valence_ts2,valence_pred_ts2),mse(valence_ts3,valence_pred_ts3)))
np.savetxt("Compdata/SVM_valence_performance-mse.csv", valence_mse, delimiter=',', header="MSE1,MSE2,MSE3", comments="")
valence_mae = np.column_stack((mae(valence_ts1,valence_pred_ts1),mae(valence_ts2,valence_pred_ts2),mae(valence_ts3,valence_pred_ts3)))
np.savetxt("Compdata/SVM_valence_performance-mae.csv", valence_mae, delimiter=',', header="MAE1,MAE2,MAE3", comments="")
valence_r2 = np.column_stack((r2(valence_ts1,valence_pred_ts1),r2(valence_ts2,valence_pred_ts2),r2(valence_ts3,valence_pred_ts3)))
np.savetxt("Compdata/SVM_valence_performance-r2.csv", valence_r2, delimiter=',', header="r2-1,r2-2,r2-3", comments="")


# 
# ### Visualization:

# In[30]:


import matplotlib.pyplot as pl
pl.plot(valence_ts3.values, 'r--',valence_pred_ts3, 'g--', linewidth =1.0)
pl.ylim(ymin=-1,ymax=+1)
pl.xlim(xmin=0, xmax=66)
pl.ylabel("Valence")
pl.xlabel("Index")
fig_size = pl.rcParams["figure.figsize"]
fig_size[0] = 50
fig_size[1] = 4
pl.rcParams["figure.figsize"] = fig_size
pl.show()


# In[31]:


pl.plot(arousal_ts2.values, 'r--',arousal_pred_ts2, 'g--', linewidth =1.0)
pl.ylim(ymin=-1,ymax=+1)
pl.xlim(xmin=0, xmax=66)
pl.ylabel("Arousal")
pl.xlabel("Index")
fig_size = pl.rcParams["figure.figsize"]
fig_size[0] = 50
fig_size[1] = 4
pl.rcParams["figure.figsize"] = fig_size
pl.show()


# In[ ]:


#pl.savefig('RFR3-arousal.png', dpi = 300)

