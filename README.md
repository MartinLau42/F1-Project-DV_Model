# F1-Project-DV_Model
Application of Tephigram in Forecasting Diurnal Variation of Surface Temperature - Deep Learning based

## Target
  • To forecast 9-day DV classes in a probabilistic manner using tephigram and sounding data

## DV Class
  Small : DV < 2.5 <br>
  Normal : 2.5 <= DV < 5.5 <br>
  Large : DV >= 5.5

## Data
  ### Training
    • King's Park sounding data (2-second resolution)
    • King's Park tephigram (drawn by the sounding data using custom python script)
    • HKO DV (abs_Tmax - abs_Tmin) of the training period
  ### Verification
    • Model sounding data (ECMWF 137 levels)
    • Model tephigram (drawn by the sounding data using custom python script)
    • CFO and OCF 9-day performance of DV forecast (developed by 9-day forecast of Tmax and Tmin)
    
## Flow
  ### Data preparation
    1) Interpolate King's Park sounding data (2-second resolution) according to the average values of pressure levels in ECMWF data which is limited to >= 50hPa (137 levels turns out to be 90 levels) --> interpolate_137_levels.ipynb (Scripts - 1)
    2) Draw King's Park tephigram --> loop_file_90.sh (without background lines) / loop_file_90_bg.sh (with background lines) (Scripts - 2)
    3) Collect DV data
    4) Collect Model sounding data (ECMWF 137 levels)
    5) Draw model tephigram --> loop_file.sh (without background lines) / loop_file_bg.sh (with background lines) (Scripts - 3)
    6) Image cleaning if needed --> remove_png_not_00Z_12Z.ipynb (Scripts - 4)
    7) Collect 9-day forecast of Tmax and Tmin by CFO and OCF
  ### Preprocess (detailed please see preprocess_for_single_tephi_input.py / preprocess_for_multi_tephi_input.py) (Scripts - 5)
    1) Data preprocessing
    2) Define training, validation and test set
    3) Scale the sounding data into range of [0, 1] or [-1, 1]
    4) Create Dataset / TFRecords
  ### Building model (detailed please see build_model_for_single_tephi_input.py / build_model_for_multi_tephi_input.py) (Scripts - 6)
    1) Define initial_bias (for binary classification)
    2) Build your model
    3) Compile the model
    4) Define callback (including the loss function and the metrics)
    5) Define class_weight
  ### Training
    1) Train the model
    2) Save the model
    3) Plot the history (validation and test)
  ### Fine-tuning (detailed please see build_model_for_single_tephi_input.py / build_model_for_multi_tephi_input.py) (Scripts - 6)
    1) Unfreeze some layers in pretrained model
    2) Compile the model
    3) Retrain
    4) Save the model
    5) Plot the history (validation and test)
  ### Performance Checking (detailed please see post_processing_for_single_tephi_input.py / post_processing_for_multi_tephi_input.py) (Scripts - 7)
    1) Loss function
    2) POD, FAR, CSI
    3) ROC curve
    4) Precision-Recall curve
    5) Classification report (precision, recall, f1-score)
    6) Reliability diagram
    7) Class Activation Map (CAM)
  ### Verification on model tephigram (detailed please see post_processing_for_single_tephi_input.py / post_processing_for_multi_tephi_input.py) (Scripts - 7)
    1) Plot POD-FAR-CSI comparison between model, CFO and OCF

## Model
  Single-tephi : Self-build CNN with pretrained model VGG19 <br>
  Multi-tephi : On top of single-tephi model for each tephi input, RNN is added after concatenation

## Scripts
  ### 1) interpolate_137_levels.ipynb
    Dependencies 
      a) King's Park sounding data (2-second resolution) 
      b) Model sounding data (ECMWF 137 levels) 
      c) problem_date_upto100hPa.txt 
    Return 
      a) /base_directory/137_interpolate_KP_ascent_data/KPUpper_yyyymmddZZ.csv (ZZ : 00 or 12) 
      b) tephi_interpolate_90_levels_00Z.csv 
      c) tephi_interpolate_90_levels_12Z.csv 
  
  ### $2) loop_file_90.sh / loop_file_90_bg.sh 
    Dependencies 
      $ All in directory : tephigram/ 
        a) tephigram_90_levels.py / tephigram_90_levels_bg.py 
        b) thermo.py 
        c) xyplot.py 
        d) constants.py 
    Return 
      a) ~/public_html/tephi_png_without_background/tephi-KPUpper_yyyymmddZZ.png (ZZ : 00 or 12) 
  
  ### $3) loop_file.sh / loop_file_bg.sh --> Similar to Scripts - 2 
  
  ### 4) remove_png_not_00Z_12Z.ipynb 
    Dependencies 
      a) Directory with filename {tephi-KPUpper_yyyymmddZZ.png} (ZZ : 00 or 12) 
    --> Remove files that not in these formats
  
  ### 5) preprocess_for_single_tephi_input.py / preprocess_for_multi_tephi_input.py
  
  ### 6) build_model_for_single_tephi_input.py / build_model_for_multi_tephi_input.py 
  
  ### 7) post_processing_for_single_tephi_input.py / post_processing_for_multi_tephi_input.py
  
  ### 8) train_single_tephi_model.ipynb / train_multi_tephi_model.ipynb 
  --> The scripts where you actually do the experiments on, enjoy it! <br>
  --> Suggest you can read and try single-tephi scripts before getting into multi-tephi scripts
    
$ Not stored in github, please look at the documentation (.docx)
