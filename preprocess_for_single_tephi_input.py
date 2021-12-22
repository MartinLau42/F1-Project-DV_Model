import os
import shutil
import glob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def round_(x, digits=0):
    # Create a new round_() to tackle the problem of round() 
    # https://medium.com/thefloatingpoint/pythons-round-function-doesn-t-do-what-you-think-71765cfa86a8 
    if (type(x) is not int) & (~np.isnan(x)):
        if len(str(x).split('.')[1]) >= digits + 1:
            if x > 0:
                if str(x).split('.')[1][digits] >= '5':
                    return np.ceil(x*10**digits)/10**digits
                else:
                    return np.floor(x*10**digits)/10**digits
            else:
                if str(x).split('.')[1][digits] >= '5':
                    return np.floor(x*10**digits)/10**digits
                else:
                    return np.ceil(x*10**digits)/10**digits
        else:
            return x
    else:
        return x


def determine_dv_class(dv, n_class):
    # Convert DV value into class [0, 1, 2] or binary [0, 1]
    if n_class == 3:
        if dv >= 5.5:
            return 2
        elif dv < 2.5:
            return 0
        elif (dv < 5.5) and (dv >= 2.5):
            return 1
        else:
            raise Exception('Check DV values')        
    elif n_class == 2:
        if dv >= 5.5:
            return 1
        elif dv < 5.5:
            return 0
        else:
            raise Exception('Check DV values')


def get_DV_df(csv_path, n_class):
    # Preprocess ; the csv should include 'Datetime'(yyyymmdd) and 'Abs_DV'(°C) column
    df_dv = pd.read_csv(csv_path)
    df_dv = df_dv[df_dv['Abs_DV'].notnull()]
    df_dv['dv_class'] = df_dv['Abs_DV'].apply(lambda x: determine_dv_class(x, n_class))

    df_dv['Datetime'] = df_dv['Datetime'].astype(int)
    df_dv['Datetime'] = df_dv['Datetime'].apply(lambda x: int(str(x) + '00'))
    df_dv['date'] = df_dv['Datetime'].apply(lambda x: str(x))        
    return df_dv   


def get_sounding_df(csv_path):
    # Preprocess ; the csv should include 'date'(yyyymmddZZ), 'PRES'(hPa), 'TEMP'(°C), 'DWPT'(°C), 'Wdir'(deg) and 'Wspd'(m/s) column
    # PRES -> pressure ; TEMP -> temperature ; DWPT -> dew point temperature ; Wdir -> wind direction ; Wspd -> wind speed
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['date', 'PRES']).reset_index(drop=True)
    df['date'] = df['date'].astype(str)
    df.loc[df['Wspd']==0, 'Wdir'] = 0                          # Define Wdir = 0 for no wind
    df.loc[(df['Wdir']==0) & (df['Wspd']!=0), 'Wdir'] = 360    # Define Wdir = 360 for northerly wind
    df['depression'] = df['TEMP'] - df['DWPT']
    return df


def save_all_sounding_data(df, public_dir, feature_to_scale): 
    # Save all available sounding data (not scaled) by date to public_dir ; format {tephi-KPUpper_yyyymmddZZ.npy}
    for d, df_d in df.groupby('date'):
        w = df_d[feature_to_scale].values

        filename = 'tephi-KPUpper_{}.npy'.format(d)
        target = os.path.join(public_dir, filename)
        if not os.path.exists(target): # Save the file if not exists
            np.save(target, w)        

            
def remove_no_dv(df_dv, df):
    # Clean the df with no DV and return the new one
    df_date = df_dv[['date']]
    df_merged = df_date.merge(df, how='inner', on='date')
    return df_merged            


def define_train_valid_test(df):
    # training: 0.7 ; validation: 0.15 ; test: 0.15
    Ndate = df['date'].nunique()
    Ntrain = Ndate*7//10
    Nvalid = int((Ndate*1.5)//10)
    #Ntest = Ndate - Ntrain - Nvalid    
    
    sorted_date = np.sort(df['date'].unique())
    train_date = sorted_date[:Ntrain]
    valid_date = sorted_date[Ntrain:Ntrain+Nvalid]
    test_date = sorted_date[Ntrain+Nvalid:]    
    
    df['train'] = df['date'] <= np.max(train_date)
    df['valid'] = (df['date'] >= np.min(valid_date)) & (df['date'] <= np.max(valid_date))
    df['test'] = df['date'] >= np.min(test_date)    
    return df


def rescale_sounding_data(df, feature_to_scale, scaler=None):
    """
    rescale the sounding data.
        if train data in indicated in df, a new MinMax scaler would be trained and applied to the dataset
        otherwise, need to provided a trained scaler to rescale the data
    """
    if ('train' not in df.columns) and (scaler is None):
        raise Exception('you need to provided a trained scaler to rescale the data or indicate training data in the dataframe')
    if 'train' in df.columns:
        scaler = MinMaxScaler()
        scaler.fit(df[df['train']][feature_to_scale].values)    # reshape(Ntrain, -1)
    df_scaled = pd.DataFrame(scaler.transform(df[feature_to_scale].values))  # reshape(Ndate, -1)
    df_scaled.columns = [x+'_scaled' for x in feature_to_scale]
    df = pd.concat([df, df_scaled], axis=1)
    return df, scaler
        

def copy_tephi(source_dir, target_dir, date=None, model_run=None, forecast=None):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if (model_run is not None) & (forecast is not None):
        filename = 'tephi-HK05_{}_{}.png'.format(model_run, forecast)
    else:
        filename = 'tephi-KPUpper_{}.png'.format(date)
    source = os.path.join(source_dir, filename)
    target = os.path.join(target_dir, filename)
    if os.path.exists(source):
        if not os.path.exists(target): # Copy the file if not exists
            shutil.copy(source, target)
    else:
        print('Missing: %s' % filename)

        
def save_scaled_sounding_data(df, target_dir, feature_scaled, date=None, model_run=None, forecast=None):
    if (model_run is not None) & (forecast is not None):
        w = df[(df['model_run']==model_run) & (df['forecast']==forecast)][feature_scaled].values
        filename = 'tephi-HK05_{}_{}.npy'.format(model_run, forecast)
    else:
        w = df[df['date']==date][feature_scaled].values
        filename = 'tephi-KPUpper_{}.npy'.format(date)
        
    if w.size != 0:
        target = os.path.join(target_dir, filename)
        if not os.path.exists(target): # Save the file if not exists
            np.save(target, w)    

        
def copy_tephi_and_save_sounding(df_dv, df_sounding, source_dir, target_dir, feature_scaled, n_class, stage, date):
    # Save tephi and sounding data where DV is available (in our case, df_sounding = df_00 such that 00Z tephi and data are saved)
    dv = df_dv[df_dv['date'] == date]['Abs_DV'].values[0]
    dv_class = determine_dv_class(dv, n_class)
    target_dir = os.path.join(target_dir, stage, str(dv_class))

    copy_tephi(source_dir, target_dir, date)
    save_scaled_sounding_data(df_sounding, target_dir, feature_scaled, date)

    
def get_tephi_path(target_dir):
    train_img = glob.glob(os.path.join(target_dir, 'train/*/*.png'))
    valid_img = glob.glob(os.path.join(target_dir, 'valid/*/*.png'))
    test_img = glob.glob(os.path.join(target_dir, 'test/*/*.png'))
    
    # Re-order train_img for mixture of three classes that can be evenly 'seen' by the machine; valid_img and test_img for convenience
    train_date = [x.split('.')[0][-10:] for x in train_img]
    train_img = [x for _, x in sorted(zip(train_date, train_img))]
    
    valid_date = [x.split('.')[0][-10:] for x in valid_img]
    valid_img = [x for _, x in sorted(zip(valid_date, valid_img))]
    
    test_date = [x.split('.')[0][-10:] for x in test_img]
    test_img = [x for _, x in sorted(zip(test_date, test_img))]
    return train_img, valid_img, test_img


def get_sounding_path(target_dir):
    train_img, valid_img, test_img = get_tephi_path(target_dir)
    
    train_npy = [x.replace('.png', '.npy') for x in train_img] 
    valid_npy = [x.replace('.png', '.npy') for x in valid_img] 
    test_npy = [x.replace('.png', '.npy') for x in test_img] 
    return train_npy, valid_npy, test_npy


def get_y_true(target_dir):
    train_img, valid_img, test_img = get_tephi_path(target_dir)
    
    y_train = np.asarray([int(x.split('/')[-2]) for x in train_img])
    y_valid = np.asarray([int(x.split('/')[-2]) for x in valid_img])
    y_test = np.asarray([int(x.split('/')[-2]) for x in test_img])
    return y_train, y_valid, y_test


def get_label(file_path):
    # Like determine_dv_class() but return a Tensor of tf.int64
    class_names = ['0', '1', '2']
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)


def read_img(img_path):
    encoded_img = tf.io.read_file(img_path)
    return encoded_img


def decode_img(encoded_img, resize=False, img_height=None, img_width=None):
    # Resize here seems to lower the resolution, suggest adjust the image size when you draw the tephigram if want to
    img = tf.io.decode_png(encoded_img, channels=3)
    if resize:
        img = tf.image.resize(img, [img_height, img_width], preserve_aspect_ratio=True)
    return img


def load_np(npy_path):
    nparray = np.load(npy_path)
    return nparray


@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def tf_npload(npy_path):
    nparray = tf.numpy_function(load_np, [npy_path], tf.float64)
    return nparray


def process_path(img_path, npy_path):
    # map this function into the dataset
    encoded_img = read_img(img_path)
    img = decode_img(encoded_img)
    nparray = tf_npload(npy_path)
    label = get_label(img_path)
    return (img, nparray), label


def create_dataset(target_dir, batch_size):
    train_img, valid_img, test_img = get_tephi_path(target_dir)
    train_npy, valid_npy, test_npy = get_sounding_path(target_dir)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_img, train_npy))
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_img, valid_npy))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_img, test_npy))
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    valid_dataset = valid_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    
    # cache() + prefetch() can speed up the process
    train_dataset = train_dataset.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)
    valid_dataset = valid_dataset.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)
    return train_dataset, valid_dataset, test_dataset
