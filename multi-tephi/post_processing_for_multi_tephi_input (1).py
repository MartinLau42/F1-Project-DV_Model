from post_processing_for_single_tephi_input import *
from preprocess_for_multi_tephi_input import *
from build_model_for_multi_tephi_input import *


def get_ec_data(start_UTC='00', lead_day=1): 
    # start_UTC='00'/'12' ; lead_day=1,2,...9
    initial_path = sorted(glob.glob('/home/deeplearn/cslau/input_full_137/*{}'.format(start_UTC)))
    initial_dir = sorted(os.listdir('/home/deeplearn/cslau/input_full_137'))
    initial_dir = [x for x in initial_dir if x[-2:]==start_UTC]

    if start_UTC == '00':
        forecast_t = [datetime.strftime(datetime.strptime(x, '%Y%m%d%H')+timedelta(days=lead_day), '%Y%m%d') + '00' for x in initial_dir]
    elif start_UTC == '12':
        forecast_t = [datetime.strftime(datetime.strptime(x, '%Y%m%d%H')+timedelta(days=1+lead_day), '%Y%m%d') + '00' for x in initial_dir]
    else:
        raise TypeError('start_UTC only includes "00" and "12"')
        
    EC_tephi_list = []
    date_not_available = []
    initial_not_available = []

    for i, path in enumerate(initial_path):
        try:
            ec_list = get_list(forecast_t[i], glob.glob(path + '/*'), initial_dir[i])
            EC_tephi_list.extend(ec_list)
        except:
            date_not_available.append(forecast_t[i])
            initial_not_available.append(initial_dir[i])
            
    model_run = sorted(list(set(initial_dir) - set(initial_not_available)))        
    forecast = sorted(list(set(forecast_t) - set(date_not_available)))
    
    return EC_tephi_list, model_run, forecast


def get_ec_dataset(start_UTC, lead_day, feature_to_scale, scaler, df_dv, DataSequenceReader, public_dir_ec, ver_dir):
    target_dir = os.path.join(ver_dir, 'ec_2tephi_data_{}Z_lead-{}d'.format(start_UTC, lead_day)) # ec_multi_tephi_data_{}Z_lead-{}d
    feature_scaled = [x + '_scaled' for x in feature_to_scale]
    
    EC_tephi_list, model_run, forecast = get_ec_data(start_UTC, lead_day)
    
    ec_df = get_ec_df(EC_tephi_list, feature_to_scale, scaler)

    model_run_all = [x.split('_')[-3] for x in EC_tephi_list]
    forecast_all = [x.split('_')[-2] for x in EC_tephi_list]
    for mr, fc in zip(model_run_all, forecast_all):
        source_dir = os.path.join(public_dir_ec, mr)
        copy_ec_tephi_and_save_sounding(source_dir, target_dir, ec_df, feature_scaled, mr, fc)

    ec_png_list = glob.glob(target_dir + '/*.png')
    ec_npy_list = [x.replace('.png', '.npy') for x in ec_png_list]

    tfrecord_dir = os.path.join(target_dir, 'tfrecords')
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    filename_verification = os.path.join(tfrecord_dir, 'verification.tfrecords')
    if not os.path.exists(filename_verification):
        write_tfrecords(forecast, filename_verification, df_dv, ec_png_list, ec_npy_list, model_run)

    verification_dataset = DataSequenceReader.read_batch([filename_verification])
    return verification_dataset, model_run, forecast

        
def get_9day_result_(model, feature_to_scale, scaler, df_dv, DataSequenceReader, n_class, public_dir_ec, ver_dir, directory=None):
    df_9day_00 = pd.DataFrame()
    df_9day_12 = pd.DataFrame()
    for start_UTC in ['00', '12']:
        for lead_day in range(1, 10):
            if (start_UTC == '12') & (lead_day == 9):
                break
                
            verification_dataset, model_run, forecast = get_ec_dataset(start_UTC, lead_day, feature_to_scale, scaler, 
                                                                       df_dv, DataSequenceReader, public_dir_ec, ver_dir)
            
            ec_pred = model.predict(verification_dataset)
            ec_pred_label = get_pred_label(ec_pred, threshold=0.5)
            ec_pred_class = get_str_class(ec_pred_label, n_class)
            
            y_true = df_dv[df_dv['date'].apply(lambda x: x in forecast)]['dv_class']
            
            day = pd.DataFrame({'day':[lead_day]*n_class})
            result = get_result(ec_pred_class, y_true, to_file=None)
            df_one_day = pd.concat([day, result], axis=1)
            
            if start_UTC == '00':
                df_9day_00 = pd.concat([df_9day_00, df_one_day], axis=0)
            elif start_UTC == '12':
                df_9day_12 = pd.concat([df_9day_12, df_one_day], axis=0)
                
    if directory is not None:
        df_9day_00.to_csv(os.path.join(directory, 'model_9day_result_00Z.csv'), index=False)
        df_9day_12.to_csv(os.path.join(directory, 'model_9day_result_12Z.csv'), index=False)
    
    return df_9day_00, df_9day_12
