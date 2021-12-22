from preprocess_for_single_tephi_input import *
from PIL import Image

# For 2-tephi model, check_tephi_avaliabililty(..., num_tephi=2) & get_list(..., num_tephi=2)
# For 4-tephi model, check_tephi_avaliabililty(..., num_tephi=4) & get_list(..., num_tephi=4)

def save_all_scaled_sounding_data(df, target_dir, feature_scaled):
    for d, df_d in df.groupby('date'):
        w = df_d[feature_scaled]
        w = w.values

        filename = 'tephi-KPUpper_{}_scaled.npy'.format(d)
        target = os.path.join(target_dir, filename)
        if not os.path.exists(target): # Save the file if not exists
            np.save(target, w)
            
            
def check_tephi_avaliabililty(df, tephi_date_list, num_tephi=2): # num_tephi=4 if want to train 4-tephi model
    """
    loop over all datetime between and min and max datetime in df and check if tephi images are avaliable
    tephis_avaliable is True is all the following four tephigrams are avaliable:
        1. tephigram at 12 hours earlier
        2. tephigram at current datetime
        3. tephigram at 12 hours later
        4. tephigram at 24 hours later
    """
    min_date = datetime.strptime(df['date'].min(), '%Y%m%d%H')
    max_date = datetime.strptime(df['date'].max(), '%Y%m%d%H')
    N = int((max_date - min_date)/timedelta(hours=12))
    temp_date_list = [min_date + timedelta(hours=n*12) for n in range(N+1)]
    temp_date_list = [datetime.strftime(x, '%Y%m%d%H') for x in temp_date_list]

    # Determine the datetime without tephigram
    date_no_tephi = set(temp_date_list) - set(tephi_date_list)
    date_with_tephi = set(temp_date_list).intersection(set(tephi_date_list))

    df_x = pd.DataFrame(list(date_no_tephi))
    df_x['has_tephi'] = False

    df = pd.DataFrame(list(date_with_tephi))
    df['has_tephi'] = True
    # Compose a dataframe which determine whether there is tephigram for the corresponding datetime
    df = pd.concat([df, df_x])
    df.columns = ['date', 'has_tephi']
    df = df.sort_values(by='date')
    df = df.reset_index(drop=True)
    
    # Shift the dataframe to check if tephigrams are avaliable for all 2 or 4 different times
    df['tephis_avaliable'] = df['has_tephi']
    if num_tephi == 2:         # 2-tephi model (d+0 00Z & 12Z) --> default
        time_zone = [-1]
    elif num_tephi == 4:       # 4-tephi model (d-1 12Z, d+0 00Z & 12Z, d+1 00Z)
        time_zone = [1, -1, -2]
    for shift in time_zone:
        hour_shift = shift*12
        df['has_tehpi_shift_{}'.format(-hour_shift)] = df['has_tephi'].shift(shift)
        df['tephis_avaliable'] &= df['has_tehpi_shift_{}'.format(-hour_shift)]
    return df[['date', 'tephis_avaliable']]    


def remove_no_tephis(df, tephi_date_list):
    tephi_avali_df = check_tephi_avaliabililty(df, tephi_date_list)
    df = df.merge(tephi_avali_df, on='date')
    df = df[df['tephis_avaliable']]
    del df['tephis_avaliable']    
    return df


def get_list(date_time_str, search_list, model_run=None, num_tephi=2): # num_tephi=4 if want to train 4-tephi model
    date_time = datetime.strptime(date_time_str, '%Y%m%d%H')
    
    _list = []
    if num_tephi == 2:
        time_zone = [0, 12]
    elif num_tephi == 4:
        time_zone = [-12, 0, 12, 24]
    for shift in time_zone:
        date_time_shift = date_time + timedelta(hours=shift)
        date_time_shift_str = datetime.strftime(date_time_shift, '%Y%m%d%H')
        if model_run is None:
            data = list(filter(lambda x: date_time_shift_str in x, search_list))[0]
        else:
            model_run_00 = model_run + '00'
            date_time_shift_str = date_time_shift_str + '00'
            date_join = model_run_00 + '_' + date_time_shift_str
            data = list(filter(lambda x: date_join in x, search_list))[0]
            
        _list.append(data) 
        
    return _list


def decode_images(image):
    """this function reads an image and converts it to a numpy array"""
    image = np.asarray(Image.open(image).convert('RGB'))
    return image


def get_image_bytes_list(date_time_str, img_list, model_run=None):
    img_list = get_list(date_time_str, img_list, model_run)
    image_bytes_list = []
    for img in img_list:
        img = decode_images(img)
        img_bytes = img.tobytes()
        img_bytes = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))
        image_bytes_list.append(img_bytes)   
    return image_bytes_list


def get_sounding_data_list(date_time_str, soundings_list, model_run=None):
    sounding_list = get_list(date_time_str, soundings_list, model_run)
    sounding_data_list = []
    for s in sounding_list:
        s = np.load(s)
        s_bytes = s.tobytes()
        s_bytes = tf.train.Feature(bytes_list=tf.train.BytesList(value=[s_bytes]))
        sounding_data_list.append(s_bytes)   
    return sounding_data_list


def get_label_sequence(df_dv, date_time_str):
    """this function takes a list of labels and returns the list in int64"""
    label = df_dv[df_dv['date'] == date_time_str]['dv_class'].iloc[0]
    label_int_list = []
    label_int = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    label_int_list.append(label_int)

    return label_int_list


def write_tfrecords(date_list, filename, df_dv, img_list, soundings_list, model_run_list=None):
    writer = tf.io.TFRecordWriter(filename)
    
    if model_run_list is None:
        model_run_list = [model_run_list]*len(date_list)
        
    for model_run, date in zip(model_run_list, date_list):
        target_d = date[:-2]
        image_bytes_list = get_image_bytes_list(date, img_list, model_run)
        sounding_data_list = get_sounding_data_list(date, soundings_list, model_run)
        label_sequence = get_label_sequence(df_dv, date)
        
        target_d = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(target_d)]))
        images = tf.train.FeatureList(feature=image_bytes_list)
        sounding = tf.train.FeatureList(feature=sounding_data_list)
        label = tf.train.FeatureList(feature=label_sequence)
        
        context_dict = {'date': target_d}
        sequence_dict = {'image': images, '1d_data': sounding, 'label': label}
        
        sequence_context = tf.train.Features(feature=context_dict)
        sequence_list = tf.train.FeatureLists(feature_list=sequence_dict)
        
        example = tf.train.SequenceExample(context=sequence_context, feature_lists=sequence_list)
        writer.write(example.SerializeToString())
        
        
class DataSequenceReader():
    def __init__(self, batch_size, sequence_length, img_height, img_width, n_layer, n_feature):
        self.batch_size = batch_size
        self.seq_length = sequence_length
        self.height = img_height
        self.width = img_width
        self.n_layer = n_layer
        self.n_feature = n_feature
        #self.num_epochs = num_epochs

    def parse_sequence(self, sequence_example):
        
        context_features =  {
                             'date': tf.io.FixedLenFeature([], dtype=tf.string)
                            }
        sequence_features = {
                             'image': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                             '1d_data': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                             'label': tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
                            }

        context, sequence = tf.io.parse_single_sequence_example(
                            sequence_example, context_features=context_features, sequence_features=sequence_features)

        # get features context
        target_date = context['date']

        # decode image
        image = tf.io.decode_raw(sequence['image'], tf.uint8)
        image = tf.reshape(image, shape=(self.seq_length, self.height, self.width, 3))
        
        data_1d = tf.io.decode_raw(sequence['1d_data'], tf.float64)
        data_1d = tf.reshape(data_1d, shape=(self.seq_length, self.n_layer, self.n_feature))

        label = tf.cast(sequence['label'], dtype=tf.uint8)

        return (image, data_1d), label

    def read_batch(self, filename):
        dataset = tf.data.TFRecordDataset(filename)
        #dataset = dataset.repeat(self.num_epochs)
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = dataset.map(self.parse_sequence, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        #dataset = dataset.shuffle(buffer_size=10 * self.batch_size)

        return dataset
    
