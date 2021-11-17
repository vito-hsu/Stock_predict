# In this .py file, we will demo how to use common Deep Learning model to predict Stock !!
# And we will demo TSMC(Taiwan Semiconductor Manufacturing Corp., 2330.TW) stock
import yfinance as yf
from matplotlib import pyplot
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



# Recommand you skip what this function do.
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg




# prepare for the stock code
k           =   1
stock_name  =   "2330"                                                                  # select your interested StockCode to be predicted 
df          =   yf.download(stock_name + ".TW")




                                                                                        # Because we need to find what model is the best, we do the following things non-stop.
while(True):
    #region 1.  Parameter setting
    batchsize       =   random.choices([1, 8, 16, 32, 64, 128, 256], k=1, weights=[0.1, 0.01, 0.1, 0.3, 0.3, 0.15, 0.05])[0]       
                                                                                        # Due to Stateful Model, we need batchsize = 1
    data_len        =   600                                                             # setting how many recent trading days data we take 
    timesteps_head  =   12                                                              # setting the last 12 days data as input data
    timesteps_tail  =   3                                                               # setting the next 3 days data as output data (or said predicted data)
    var_n           =   2                                                               # we get two variables ( var1 : main valuable , var2 : auxiliary variable predict )
    dropout_units   =   random.choices([0.1, 0.2], k=1, weights=[0.5, 0.5])[0] if batchsize !=1 else 0.05
    num_units       =   round(data_len/batchsize)                                       # my suggestion unit value for RNN model        
    lr              =   0.01                                                            # normally default value is 0.001
    mindelta        =   0                                                               # this is for Keras EarlyStopping, if you are new, you could skip it
    patience        =   25                                                              # this is for Keras EarlyStopping, if you are new, you could skip it
    stateful        =   1 if batchsize ==1 else 0                                       # this is very important for Stateful RNN model
    model_name      =   random.choices(['GRU', 'LSTM'], k=1, weights=[0.5, 0.5])[0]     # deciding to use GRU or LSTM model
    bidirectional   =   random.choices([0, 1], k=1, weights=[0.5, 0.5])[0]              # deciding whether to use Bidirectional RNN structure
    l1l2            =   random.choices([0, 1], k=1, weights=[0.5, 0.5])[0]              # deciding whether to use L1L2 Normalization structure
    twolayer        =   random.choices([0, 1], k=1, weights=[0.5, 0.5])[0]              # deciding whether to use multi-layer DL model structure
                                                                                        # if you are newer, it might be difficult to distinguish between 
                                                                                        # the meaning of data_len and the meaning of timesteps_head
    #endregion






    #region 2. Model Building ( Totally we will randomly choose one of the 24 different model ) 
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Bidirectional, LSTM, GRU
    from keras.regularizers import L1L2, L2
    from keras.callbacks import EarlyStopping, LearningRateScheduler
    from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
    if stateful == 1:                                                                   # The following is stateful RNN
        if l1l2 == 1:
            if bidirectional == 1:
                model           =   Sequential()                                       
                model.add(Bidirectional(
                    eval(model_name)(
                        units=num_units, 
                        stateful=True,
                        batch_input_shape=(1, timesteps_head, var_n),                   
                        kernel_regularizer=L1L2(l1=0.0001, l2=0.001),                                           
                        bias_regularizer=L2(0.001),
                        activity_regularizer=L2(0.0001)
                    )
                ))                    
                model.add(Dropout(dropout_units))
                model.add(Dense(1))  
            else:
                model           =   Sequential()                                    
                model.add(eval(model_name)(
                    num_units, 
                    stateful=True,
                    batch_input_shape=(1, timesteps_head, var_n),                   
                    kernel_regularizer=L1L2(l1=0.0001, l2=0.001),
                    bias_regularizer=L2(0.001),
                    activity_regularizer=L2(0.0001)            
                ))                    
                model.add(Dropout(dropout_units))
                model.add(Dense(1))                                                    
        else:
            if twolayer == 1:
                if bidirectional == 1:
                    model           =   Sequential()                                                                                                    
                    model.add(Bidirectional(eval(model_name)(num_units, stateful=True, batch_input_shape=(1, timesteps_head, var_n), return_sequences=True)))     
                    model.add(Dropout(dropout_units))
                    model.add(Bidirectional(eval(model_name)(int(num_units/4))))
                    model.add(Dense(1))  
                else:
                    model           =   Sequential()                                                                                               
                    model.add(eval(model_name)(num_units, stateful=True, batch_input_shape=(1, timesteps_head, var_n), return_sequences=True))                                
                    model.add(Dropout(dropout_units))
                    model.add(eval(model_name)(int(num_units/4)))
                    model.add(Dense(1))   
            else:
                if bidirectional == 1:
                    model           =   Sequential()                                                                                                   
                    model.add(Bidirectional(eval(model_name)(num_units, stateful=True, batch_input_shape=(1, timesteps_head, var_n))))                                         
                    model.add(Dropout(dropout_units))
                    model.add(Dense(1))  
                else:
                    model           =   Sequential()                                                                                                  
                    model.add(eval(model_name)(num_units, stateful=True, batch_input_shape=(1, timesteps_head, var_n)))                                                       
                    model.add(Dropout(dropout_units))
                    model.add(Dense(1))      
    else:                                                                                   # The following is stateless RNN
        if l1l2 == 1:
            if bidirectional == 1:
                model           =   Sequential()                                                                                                      
                model.add(Bidirectional(
                    eval(model_name)(
                        num_units, 
                        input_shape=(timesteps_head, var_n),                                                                                            
                        kernel_regularizer=L1L2(l1=0.0001, l2=0.001),                                           
                        bias_regularizer=L2(0.001),
                        activity_regularizer=L2(0.0001)
                    )
                ))                    
                model.add(Dropout(dropout_units))
                model.add(Dense(1))  
            else:
                model           =   Sequential()                                                                                                        
                model.add(eval(model_name)(
                    num_units, 
                    input_shape=(timesteps_head, var_n),                                                                                                
                    kernel_regularizer=L1L2(l1=0.0001, l2=0.001),
                    bias_regularizer=L2(0.001),
                    activity_regularizer=L2(0.0001)            
                ))                    
                model.add(Dropout(dropout_units))
                model.add(Dense(1))                                                                                                                    
        else:
            if twolayer == 1:
                if bidirectional == 1:
                    model           =   Sequential()                                                                                                   
                    model.add(Bidirectional(eval(model_name)(num_units, input_shape=(timesteps_head, var_n), return_sequences=True)))                  
                    model.add(Dropout(dropout_units))
                    model.add(Bidirectional(eval(model_name)(int(num_units/4))))
                    model.add(Dense(1))  
                else:
                    model           =   Sequential()                                                                                                    
                    model.add(eval(model_name)(num_units, input_shape=(timesteps_head, var_n), return_sequences=True))                                  
                    model.add(Dropout(dropout_units))
                    model.add(eval(model_name)(int(num_units/4)))
                    model.add(Dense(1))   
            else:
                if bidirectional == 1:
                    model           =   Sequential()                                                                                            
                    model.add(Bidirectional(eval(model_name)(num_units, input_shape=(timesteps_head, var_n))))                                          
                    model.add(Dropout(dropout_units))
                    model.add(Dense(1))  
                else:
                    model           =   Sequential()                                                                                                    
                    model.add(eval(model_name)(num_units, input_shape=(timesteps_head, var_n)))                                                        
                    model.add(Dropout(dropout_units))
                    model.add(Dense(1))      
    #endregion







    #region 3. Model Compiling
    adam    =   Adam(learning_rate = lr)
    RMSp    =   RMSprop(learning_rate = lr)
    Adad    =   Adadelta(learning_rate = lr)
    Adag    =   Adagrad(learning_rate = lr)
    opt     =   random.choices(['adam', 'RMSp', 'Adad', 'Adag'], k=1, weights=[0.7, 0.1, 0.1, 0.1])[0]       
    model.compile(
        loss        =   'mae', 
        optimizer   =   eval(opt)                                  
    )
    # model.summary()
    #endregion







    #region 4. Preparing data (data preprocessing) and Model fitting
    # The following design is according to the Stateful RNN characteristic
    df1         =   df.iloc[(df.shape[0]-data_len)-1:,:]    if batchsize != 1 else df.iloc[(df.shape[0]-data_len)-1:,:]
    df2         =   df.iloc[(df.shape[0]-data_len*2)-1:,:]  if batchsize != 1 else df.iloc[(df.shape[0]-data_len*2)-1:(df.shape[0]-data_len),:]
    df3         =   df.iloc[(df.shape[0]-data_len*3)-1:,:]  if batchsize != 1 else df.iloc[(df.shape[0]-data_len*3)-1:(df.shape[0]-data_len*2),:]
    df4         =   df.iloc[(df.shape[0]-data_len*4)-1:,:]  if batchsize != 1 else df.iloc[(df.shape[0]-data_len*4)-1:(df.shape[0]-data_len*3),:]


    train_stock =   ["df4", "df3", "df2", "df1"]            # because df4 happened before df3, and then df2, df1             
    mae_test = [] ;  mae_Stupid = []


    import time
    start = time.time()                                     # to compute the total training time
    for stock in train_stock:                               # According to my experience, training four times for divided four data will be better 
                                                            # than training one time for non-devided data
        # stock = "df4"
        dataset     =   eval(stock) 
        linspace_n  =   1                   
        linspace    =   np.linspace(0, data_len, int(data_len/linspace_n)+1) 
        dataset     =   dataset.iloc[linspace].reset_index()
        variable    =   random.choices(["Volume", "High", "Low"], k=1, weights=[0.3, 0.3, 0.4])[0] if batchsize != 1 else "Close"      # deciding auxiliary variable                                                     # 固定使用 svid_10050
        dataset     =   dataset[['Adj Close', variable]]            
        values      =   dataset.values.astype('float32')                                                           
        scaler      =   MinMaxScaler(feature_range=(0, 1))                                                          
        scaled      =   scaler.fit_transform(values)                                                                   
        
        
        if timesteps_head != 12   or timesteps_tail != 3:                                   # remind us of the two (or more) param. settings what we've set now  
            break
                                    
        reframed    =   series_to_supervised(scaled, timesteps_head, timesteps_tail)        # if you are new, I recommand you read line by line from here
        reframed.drop(reframed.columns[[-1, -3, -5]], axis=1, inplace=True)                 # if timesteps_tail=2, should be modified as the following code:
                                                                                            # reframed.drop(reframed.columns[[-1, -3]], axis=1, inplace=True)
        values      =   reframed.values                                                     
        train_ratio =   random.randrange(35, 36, 1)/100                                    
        valid_ratio =   random.randrange(25, 26, 1)/100                                     
        train_n     =   round(values.shape[0]*train_ratio)                                 
        valid_n     =   round(values.shape[0]*valid_ratio)                                  
        test_n      =   values.shape[0]-train_n-valid_n                                    
        train       =   values[:train_n, :]                                                 
        valid       =   values[train_n:train_n+valid_n, :]                                  
        test        =   values[train_n+valid_n:, :]                                         
        train_X, train_y    =   train[:, :-3], train[:, -3:]                                # if timesteps_tail=2, should be modified as the following code:
        valid_X, valid_y    =   valid[:, :-3], valid[:, -3:]                                # train_X, train_y = train[:, :-2], train[:, -2:]
        test_X, test_y      =   test[:, :-3], test[:, -3:]                                  # so do test_X, test_y     &     valid_X, valid_y    
        train_X     =   train_X.reshape(train_X.shape[0], timesteps_head, int(train_X.shape[1]/timesteps_head))     
        valid_X     =   valid_X.reshape(valid_X.shape[0], timesteps_head, int(valid_X.shape[1]/timesteps_head))     
        test_X      =    test_X.reshape( test_X.shape[0], timesteps_head, int( test_X.shape[1]/timesteps_head))     




        def my_schedule(epoch, patience=patience, lr=lr):                                   # to enable accelerating/deccelerating learning rate 
            if epoch < patience*10:                                                         # according to the specific epoch number
                return lr                                                                   # here, set to deccelerate learning rate
            elif epoch < patience*100:
                return lr*0.5
            elif epoch < patience*1000:
                return lr*0.1
            else:
                return lr

        lr_schedule = LearningRateScheduler(                                                
            schedule=my_schedule, 
            verbose=1                                                                       # verbose 0 : quiet ,  1 : update
        )        

        restore_best_weights = True if batchsize != 1 else False                                                   
        early_stopping  =  EarlyStopping(
            monitor     = 'val_loss',                                                                               
            min_delta   = mindelta, 
            patience    = patience,
            restore_best_weights = restore_best_weights                                                    
        )


        history     =   model.fit(                    
            train_X,                                  
            train_y,                                   
            epochs = 3000,                                                                               
            batch_size = batchsize,                    
            validation_data = (valid_X, valid_y),                                                                   
            shuffle = False,                                                                # this setting is important
            callbacks = [early_stopping, lr_schedule]
        )          

    end = time.time()
    #endregion





    #region 5. Model Evaluation 
    if stateful == 1 and stock == train_stock[-1]:
        model.reset_states()                                                                                    # this is recommandation for Stateful RNN 
        yhat        =   model.predict_generator(test_X, steps=test_X.shape[0])
        yhat        =   yhat.reshape(yhat.shape[0], 1)                                                         
        test_X      =   test_X.reshape((test_X.shape[0], test_X.shape[1]*test_X.shape[2]))                      
        inv_yhat    =   np.concatenate((yhat, test_X[:, test_X.shape[1]-1:]), axis=1)                           # because we have one auxiliary variable
                                                                                                                # we need to minus 1 after test_X.shape[1]                       
        inv_yhat    =   scaler.inverse_transform(inv_yhat)                                                      
        inv_yhat    =   inv_yhat[:,0]         
        test_y      =   test_y[:,-1].reshape(test_y.shape[0], 1)                                                 
        inv_y       =   np.concatenate((test_y, test_X[:, test_X.shape[1]-1:]), axis=1)                         # here is the same
        inv_y       =   scaler.inverse_transform(inv_y)                                                          
        inv_y       =   inv_y[:,0]
    elif stock == train_stock[-1]:
        yhat        =   model.predict(test_X)[:,-1]                                                              
        yhat        =   yhat.reshape(yhat.shape[0], 1)                                                           
        test_X      =   test_X.reshape((test_X.shape[0], test_X.shape[1]*test_X.shape[2]))                       
        inv_yhat    =   np.concatenate((yhat, test_X[:, test_X.shape[1]-1:]), axis=1)                           # here is the same                        
        inv_yhat    =   scaler.inverse_transform(inv_yhat)                                                       
        inv_yhat    =   inv_yhat[:,0]                                            
        # invert scaling for actual                                                                     
        test_y      =   test_y[:,-1].reshape(test_y.shape[0], 1)                                                 
        inv_y       =   np.concatenate((test_y, test_X[:, test_X.shape[1]-1:]), axis=1)                         # here is the same     
        inv_y       =   scaler.inverse_transform(inv_y)                                                        
        inv_y       =   inv_y[:,0]


    from sklearn.metrics import mean_absolute_error
    mae_test.append(mean_absolute_error(inv_y[3:], inv_yhat[3:]))                                               # calculate MAE of test data with GRU method
    mae_Stupid.append(mean_absolute_error(inv_y[3:], inv_y[:-3]))                                               # calculate MAE of test data with stupid method
    #endregion 5. Model Evaluation 





    #region 6. Log File 
    from os.path import isfile
    from datetime import datetime, date
    today = date.today()
    now = datetime.now()
    save_date = "1117"        
    if isfile(rf"model/stock_{save_date}.txt"):                                                                 # you could use your interested variable by yourself here
        a = open(rf"model/stock_{save_date}.txt", "a")
        text = f"""model={model_name}, bi={bidirectional}, l1l2={l1l2}, two={twolayer}, state={stateful}, 
                     d.len={data_len}, t.df={stock_name}, lin={linspace_n}, t.head={timesteps_head}, d.o.={dropout_units}, 
                     units={num_units}, lr={lr}, opt={opt}, b.s.={batchsize}, pa.={patience}, var={variable}, 
                     mae_t-S={mae_test[-1]-mae_Stupid[-1]:.1f}, r.b.w={restore_best_weights}, r.t.={round(end-start)}""" 
        a.writelines(text+"\n")
        a.close()
    else:
        b = open(rf"model/stock_{save_date}.txt","w")
        text = f"""model={model_name}, bi={bidirectional}, l1l2={l1l2}, two={twolayer}, state={stateful}, 
                     d.len={data_len}, t.df={stock_name}, lin={linspace_n}, t.head={timesteps_head}, d.o.={dropout_units}, 
                     units={num_units}, lr={lr}, opt={opt}, b.s.={batchsize}, pa.={patience}, var={variable}, 
                     mae_t-S={mae_test[-1]-mae_Stupid[-1]:.1f}, r.b.w={restore_best_weights},  r.t.={round(end-start)}"""
        b.writelines(text+"\n")
        b.close()
    #endregion 6. Log File 





    #region 7. Image File
    # loss, val_loss plot
    pyplot.plot(history.history['loss'], label='train_loss')                                          
    pyplot.plot(history.history['val_loss'], label='valid_loss')
    pyplot.legend()
    pyplot.savefig(rf"model/stock_{save_date}/"+f'{k}.pic1_' + today.strftime("%m-%d-") + now.strftime("%H-%M-%S") + 
                    f"-mae_test_diff-{mae_test[-1]-mae_Stupid[-1]:.3f}.jpg", dpi=400)
    pyplot.close()
    pyplot.plot(inv_y[3:], label = 'Real')
    pyplot.plot(inv_y[:-3], label = 'Stupid predicted')                                                 # Stupid predicted means that you do nothing but make the last value 
    pyplot.plot(inv_yhat[3:], label = 'LSTM predicted')                                                 # as the predicted value (It's important for time series prediction.)
    pyplot.title(f"val_loss={np.min(history.history['val_loss']):.3f}, state={stateful}, d.o.={dropout_units}, mae.diff={mae_test[-1]-mae_Stupid[-1]:.3f}")                                 # 這邊跟 early_stopping 裡面的 restore_best_weights 有關 !!
    pyplot.legend()
    pyplot.savefig(rf"model/stock_{save_date}/"+f'{k}.pic2_' + today.strftime("%m-%d-") + now.strftime("%H-%M-%S") + 
                    f"-mae_test_diff-{mae_test[-1]-mae_Stupid[-1]:.3f}.jpg", dpi=400)
    pyplot.close()




    #region 8. Model Save
    model.save(rf"model/stock_{save_date}/"+f'{k}.multi_model_' + today.strftime("%m_%d") + f"_{variable}_{linspace_n}_jj_3.h5")      
    #endregion 8. Model Save


    k += 1







