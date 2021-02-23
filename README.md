# Predicting forest fire in Indonesia using remote sensing data and machine learning


This is the repository for our paper ["Predicting forest fire in Indonesia using remote sensing data and machine learning"](https://arxiv.org/abs/2101.01975).

This document contains the necessary information about setting up the environment and reproducing the results in the paper.

NOTE: all the scripts assume usage of python 2.7.x and ubuntu os

### Folder contents
```bash
paper-supplementary-materials
	>evaluation_scripts
		>auc_calculation
			-auc_calculation.py
		>plot_roc_curve
			-plot_roc_curve.ipynb
		-evaluation_script_50_epoch_stride1.py
		-evaluation_script_50_epoch_stride1_3month.py
		-evaluation_script_50_epoch_stride1_6month.py
		-evaluation_script_50_epoch_stride1_9month.py
		-evaluation_script_script_logistic_baseline.py
	>models
		-baseline_logistic_regression_model.hdf5
		-model.hdf5
		-model_3month.hdf5
		-model_6month.hdf5
		-model_9month.hdf5
	>preprocessing_scripts
		-eval_set_preprocessing_dates
		-fire_only_dates
		-preprocessing_dates
		-preprocessing_script_new_mask.py
		-preprocessing_script_new_mask_fire_only.py
		-test_set_preprocessing_dates
	>training_scripts
		-baseline_logistic_regression_training_script_12ts_mse.py
		-training_script_50_epoch_stride1.py
		-training_script_50_epoch_stride1_3month.py
		-training_script_50_epoch_stride1_6month.py
		-training_script_50_epoch_stride1_9month.py
	-readme.html
	-README.md
	-requirements.txt
```

## 1. Requirements

### 1.1 Required accounts

1) **Google earth engine account**

- Google earth engine account is required to access the satellite image data and perform data preprocessing.

2) **Google account with access to google drive api v3**

- Google drive api will need to be enabled for downloading of dataset. Each satellite image time series is first exported to google drive and subsequently download.

<br>

### 1.2 Required software and packages

1. Gdal 2

Install Gdal 2.1.3 (version used during implementation) If you are installing a different version of of Gdal, make sure to change the version of `pygdal` in requirements.txt to the same version. You can check the available version [here](https://pypi.org/project/pygdal/#history)

```bash
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get install libgdal20=2.1.3
sudo apt-get install libgdal2-dev=2.1.3
```

2. Python 2.x

Python 2 is used because Google Earth Engine only supported python 2 in its API at the time of implementation

To install required python packages:

```bash
pip install -r requirements.txt
```
If you are not using GPU, change tensorflow-gpu to tensorflow in the requirements.txt

You will need to [authenticate](https://developers.google.com/earth-engine/command_line#authenticate) the google earth engine python api to access your account. You can do so by executing the following in terminal:

```bash
earthengine authenticate
```
You also need to have Google Drive API enabled to run the data retrieval and preprocessing script. Follow steps 1 and 2 [here](https://developers.google.com/drive/api/v3/quickstart/python)

<br>

## 2. Data retrieval and preprocessing

Note: Ensure that there is around 200gb of free disk space

There are 2 version of the data retrieval and preprocessing script in the `preprocessing_scripts` folder:
`preprocessing_script_new_mask.py` and
`preprocessing_script_new_mask_fire_only.py`

As the names suggest `preprocessing_script_new_mask.py` retrieves and process data that contains both hotspot and non-hotspot labels, whereas `preprocessing_script_new_mask_fire_only.py` retrieves and process data only for hotspot labels.

There are a few other parameters that you should set in both of the data retrieval and preprocessing scripts.
- `save_folder_location`  -- where to save the processed data files
- `period_start` and `period_end` -- the interval in which to get reference images. The date of the reference images is time t, as referred in the paper. For each reference image, historical data for t - 52 weeks to t is retrieved and hotspot labels in the period t + 4 weeks to t + 5 weeks are retrieved. 

<br>

### 2.1 To retrieve data used in paper

The data used in paper is retrieved by running the following:
For training data:
- `preprocessing_script_new_mask_fire_only.py` script with dates in `fire_only_dates`
- `preprocessing_script_new_mask.py` script with dates in `preprocessing_dates`

1. Set different folders for `save_folder_location` parameter for `preprocessing_script_new_mask_fire_only.py` and `preprocessing_script_new_mask.py`
2. Run `preprocessing_script_new_mask_fire_only.py` script with dates in `fire_only_dates`
3. Run `preprocessing_script_new_mask.py` script with dates in `preprocessing_dates`

For test set data:
1. Set desired test data folder location for `save_folder_location` parameter in `preprocessing_script_new_mask.py`
2. Run `preprocessing_script_new_mask.py` script with dates in `test_set_preprocessing_dates`

For evaluation set data:
For each pair of `period_start` and `period_end` in `eval_set_preprocessing_dates`:
1. Set desired test data folder location for `save_folder_location` parameter in `preprocessing_script_new_mask.py` (the test data folder location should be different for each pair of `period_start` and `period_end`)
2. `preprocessing_script_new_mask.py` script with dates in `eval_set_preprocessing_dates`

**NOTE:** the data retrieved using the above instruction may not be the exact data used in our experiments, but rather as close as possible. This is because we used to import the Indonesia boundary KML format file via Google Fusion Tables (now discontinued). As such, we found a more recent source of country boundary data provided by United States Department of State, Office of the Geographer, via Google Earth Engine.

<br>

#### Data sources

1. Indonesia boundary

    The Indonesian boundary used is loaded from: https://developers.google.com/earth-engine/datasets/catalog/USDOS_LSIB_SIMPLE_2017

	As this country boundary is different from the one used in our paper, there are minor differences in the data retrieved. However, we still expect our model to perform similarly.

2. Landsat 7 data

    The Landsat 7 images used are the existing version within google earth engine. The ImageCollectionID is `'LANDSAT/LE07/C01/T1_RT'`. Information on the Landsat 7 dataset used is available here: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C01_T1_RT

3. Hotspot data (FIRMS)

    The Fire Information for Resource Management System (FIRMS) hotspot dataset is being used. It is an existing dataset on google earth engine with the ImageCollectionID: `'FIRMS'`. Additional information is available: https://developers.google.com/earth-engine/datasets/catalog/FIRMS

<br>

## 3. Training

To train the model(s) in the paper, run the training scripts in the `training_scripts` folder.

Step 1 : Set at the user parameters at start of the training scripts:
- saved model file name
- log file name
- directories of dataset location (training dataset, training dataset with fire only and testset)

`training_dataset_fire_only_directory` --- set it to directory of data downloaded by running `preprocessing_script_new_mask_fire_only.py` script with dates in `fire_only_dates`

`training_dataset_directory` --- set it to directory of data downloaded by running `preprocessing_script_new_mask.py` script with dates in `preprocessing_dates`

`testset_directory` --- set it to the directory of test set data downloaded using the preprocessing script for example `'./preprocessed_data_test_set/'`

**NOTE: when setting directory parameters include backslash at the end eg **`'./preprocessed_data_test_set/'` **instead of **`'./preprocessed_data_test_set'`


Step 2: Run the training script as follows
```
python <script_name>.py
```
There are 5 included training scripts, `<script_name>` can be:
`training_script_50_epoch_stride1.py` -- trains model on 1 year of historical data
`training_script_50_epoch_stride1_3month.py` -- trains model on recent 3 month of historical data
`training_script_50_epoch_stride1_6month.py` -- trains model on recent 6 month of historical data
`training_script_50_epoch_stride1_9month.py` -- trains model on recent 9 month of historical data
`baseline_logistic_regression_training_script_12ts_mse.py` -- trains baseline logistic regression model

The dataset location can be the same, processing is done within the script to ensure the right amount of data is passed to model during training.

In the scripts, the logs output the performance on test set, with 0.5 as prediction threshold

<br>

## 4. Evaluation

The scripts to perform evaluation of performance, on evaluation set is located in the `evaluation_scripts` folder.

Step 1: Set parameters at the start of the training scripts:
- saved model file name
- log file name
- prediction pickle file name
- directory of evaluation dataset location

The default values in the script are for evaluating data instances with reference time t in August 2019, predicting for hotspots in September 2019. If evaluating on different data, change location parameter in the script accordingly.

Step 2: Run the training script as follows
```
python <script_name>.py
```
There are 5 inluded evaluation scripts,  `<script_name>` can be:
`evaluation_script_50_epoch_stride1.py` -- evaluates model on 1 year of historical data
`evaluation_script_50_epoch_stride1_3month.py` -- evaluates model on recent 3 month of historical data
`evaluation_script_50_epoch_stride1_6month.py` -- evaluates model on recent 6 month of historical data
`evaluation_script_50_epoch_stride1_9month.py` -- evaluates model on recent 9 month of historical data
`evaluation_script_script_logistic_baseline.py` -- evaluates baseline logistic regression model

The dataset location can be the same, processing is done within the script to ensure the right amount of data is passed to model during evaluation.

The saved pickle file from the evaluation script is a tuple of 2 lists, the fiirst being the model prediction and the second being the ground truth label `([model prediction list], [ground truth label list])`.

The pickle file is then used to calculate the area under curve.

<br>

#### 4.1 Area under ROC

In the `auc_calculation` folder, there is a script (`auc_calculation.py`) to calculate AUC from the pickle files generated from evaluation scripts.

Step 1: In the script, there are  2 parameters to be set:
- `prediction_filename`-- the pickle file from the evaluation script to have AUC calculated
- `output_log_file` -- the name of the log file where the AUC result will be written to

Step 2: Run the script, AUC result will be saved to the log file.
```
python auc_calculation.py
```
<br>

#### 4.2 Plotting ROC curve

In the `plot_roc_curve` folder under `evaluation_scripts` folder, there is a jupyter notebook `plot_roc_curve.ipynb` for plotting the ROC curve from pickle files that are output by the evaluation scripts.

Start jupyter on python 2.x by running `jupyter lab` in terminal and follow instructions within notebook.

<br>

## 5. Model files

We have included some of the trained model files in the `models` folder. The naming of the model files matches the preset names in user parameters in the training and evaluation scripts.

The models included are:
```
model.hdf5 --- model trained on 1 year of historical satellite image histogram data
model_3month.hdf5 --- model trained on ~3 month of historical satellite image histogram data
model_6month.hdf5 --- model trained on ~6 month of historical satellite image histogram data
model_9month.hdf5 --- model trained on ~9 month of historical satellite image histogram data
baseline_logistic_regresion_model.hdf5 --- baseline model trained on same data as model.hdf5
```

The model trained and evaluated on 9 month data is omitted due to size limitations of the supplementary file allowed.

You will be able to run the respective evaluation scripts on the models provided to get an idea of the model's performance.

<br>

## 6. Reproducing results in paper

To replicate the tables and figure in the paper, please first [retrieve the evaluation set using the preprocessing script](#21-to-retrieve-data-used-in-paper)

<br>

#### 6.1 Reproducing values in Table 4

To reproduce the values in table 4: comparison of our model with baseline, you need to run the [evaluation script](#4-evaluation) on the provided model and baseline.

The two models are the `model.hdf5`(our model) and `baseline_logistic_regression_model.hdf5`(baseline model) in the model folder. Run the evaluation script for each model for each of the month of the evaluation dataset, when [downloading them using the preprocessing script](#21-to-retrieve-data-used-in-paper), change the `save_folder_location` parameter to save each month of the evaluation dataset in a separate location.

Each run of the evaluation script should produce a `.pickle` file, consisting of the model's prediction and the ground truth labels for all the instances in that evaluation. Run the `auc_calculation.py` [script](#41-area-under-roc) (setting `prediction_filename` to be the `.pickle` file output by the evaluation script) to calculate the area under ROC curve result that is in the table.

<br>

#### 6.2 Reproducing values in Table 5

To reproduce the values in table 5: auc values for reduced data, you need to first [train the model](#3-training) for 9 month as we did not include it due to space constraints. The other models are provided in the `models` folder.

Then once all the models are trained, [run the different evaluation scripts](#4-evaluation), corresponding to the different amount of data each model is trained with (eg 3 month, 6 month, 9 month, 1 year). The evaluation scripts need to be run for the different months in the evaluation data for each model. The `1 year` column of Table 5 is the same as `Agni (our model)` in Table 4.

<br>

#### 6.3 Reproducing Figure 3

Step 1: Run the evaluation script for `model.hdf5` with September 2019 and August 2019 hotspot evaluation dataset. The data are retrieved by running `preprocessing_script_new_mask.py` with `period_start = '2019-08-01', period_end = '2019-08-28'`and `period_start = '2019-07-01', period_end = '2019-07-28'`Refer to [Section 2.1](#21-to-retrieve-data-used-in-paper) for more information.

Step 2: Follow instructions in [Section 4.2](#42-plotting-roc-curve) to use provided jupter notebook `plot_roc_curve.ipynb`to plot roc curve.

<br>

#### 6.4 Other tables in paper

The other tables in paper are not results. Table 1 is the specification of Landsat 7 satellite which can also be found [here](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C01_T1_RT#bands)

Table 2 is the architecture of our model, it corresponds to the code found in the training scripts.
