
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import numpy as np

df_oil = pd.read_csv('oil.csv.',header=0)
df_store = pd.read_csv('stores.csv',header=0)
df_item = pd.read_csv('items.csv', header=0) 
all_items = pd.read_csv('train.csv',header = 0)
tst = pd.read_csv('test.csv',header=0)

def getFormattedInput(it):
    df_X = pd.DataFrame(pd.np.empty((len(it), 5)) * pd.np.nan,columns=['month','date','day','store_nbr','item_nbr'])
    
    it['onpromotion']=it['onpromotion'].fillna(0)

    date = pd.to_datetime(it['date'],infer_datetime_format=True) # series object
    temp_dict = {'month':date.dt.month,'date':date.dt.day,'day':date.dt.dayofweek}
    df_X = pd.DataFrame(temp_dict)
    df_X = pd.concat([df_X,it['store_nbr'],it['item_nbr']],axis=1)
  
    df_X = pd.get_dummies(df_X)
    df_X = pd.concat([df_X, it['onpromotion']], axis=1)

    return df_X

item_nbrs = df_item.values[0:-1,0]

group_size = 10

for i in range(0,len(item_nbrs),group_size):
    print('-----------------\n',i,'\n---------------\n')
    temp_item_nbrs = item_nbrs[i:(i+group_size)]
    it = pd.DataFrame(columns=['id','date','store_nbr','item_nbr','unit_sales','onpromotion'])
    
    for j in range(0,len(temp_item_nbrs)):
        itt = all_items.loc[(all_items['item_nbr']==int(temp_item_nbrs[j]))]
        it = it.append(itt)
        
    len_it = len(it)  
    it_max = it.nlargest(int(np.around(len_it/100)),'unit_sales')
    it = it.drop(it_max.index.get_level_values(0))
    it_min = it.nsmallest(int(np.around(len_it/200)),'unit_sales')
    it = it.drop(it_min.index.get_level_values(0))
    
    Y = np.reshape(it['unit_sales'].values,[-1,1])

    all_sample= np.arange(len(Y))
    val_sample = np.random.randint(len(Y),size=int(np.around(len(Y)/20)))
    train_sample = np.setdiff1d(all_sample,val_sample)

    del it['unit_sales']
    for j in range(0,group_size):
        itt = tst.loc[(tst['item_nbr']==int(temp_item_nbrs[j]))]
        it = it.append(itt)
    
    test_sample = np.arange(len(Y),len(it))
    
    df_X = getFormattedInput(it)
    X = df_X.values
    
    tr_X = X[train_sample,:]
    val_X = X[val_sample,:]
    tr_Y = Y[train_sample]
    val_Y = Y[val_sample]
    test_X = X[test_sample,:]
    
    model = Sequential()
    model.add(Dense(units=120,activation='relu',bias_initializer='ones',input_dim=X.shape[1]))
    model.add(Dense(units=60,activation='relu'))
    model.add(Dense(units=50,activation='relu'))
    model.add(Dense(units=40,activation='relu'))
    model.add(Dense(units=30,activation='relu'))
    model.add(Dense(units=20,activation='relu'))
    model.add(Dense(units=10,activation='relu'))
    model.add(Dense(units=1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.fit(tr_X, tr_Y, validation_data=(val_X, val_Y), epochs=100, batch_size=10000)

    Y_hat = model.predict(val_X)

    min_val = np.min(val_Y)

    for j in range(0,len(Y_hat)):
        if Y_hat[j]<min_val:
            Y_hat[j]=min_val

    print('NWRMSLE = ',np.dot((np.log(Y_hat+1)-np.log(val_Y+1)).T,(np.log(Y_hat+1)-np.log(val_Y+1))))
    
    Y_hat_test = model.predict(test_X)

    for j in range(0,len(Y_hat_test)):
        if Y_hat_test[j]<min_val:
            Y_hat_test[j]=min_val
    
    print(len(Y_hat_test))
    
    temp = it['id'].values
    ids = np.reshape(temp[test_sample],[-1,1])
    result_array = np.concatenate((ids,Y_hat_test), axis=1)
    result = pd.DataFrame(result_array, columns=['id','unit_sales'])

    result.to_csv(str(i)+'.csv',index=False)

df_subm = pd.read_csv('sample_submission.csv',header=0)
df_subm['unit_sales']=0

for i in range(0,len(item_nbrs),group_size):
    df_result = pd.read_csv(str(i)+'.csv',header=0)
    
    df_subm = pd.merge(df_subm, df_result, on='id', how='left')
    df_subm = df_subm.fillna(0)
    df_subm['unit_sales'] = df_subm['unit_sales_x']+df_subm['unit_sales_y']
    del df_subm['unit_sales_x']
    del df_subm['unit_sales_y']

df_subm.to_csv('final_submission.csv',index=False)

