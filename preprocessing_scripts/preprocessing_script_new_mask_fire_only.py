from __future__ import print_function

import ee
import time
import sys
import numpy as np
import pandas as pd
import itertools
import os
import io
import urllib
import datetime
import pickle


from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
from googleapiclient.http import MediaIoBaseDownload

from osgeo import gdal
import matplotlib.pyplot as plt

##############################################
# USER SET PARAMETERS
##############################################
# set location to save preprocessed files
save_folder_location = './preprocessed_data/'
# set the location of the mask resource on google earth engine - this is account dependent

# period start and period end is for reference image (time t)
# get 1 year historical satellite image (t - 52 weeks to t)
# hotspot labels for 5th week into future is retrieved (t + 4weeks to t + 5 weeks)
period_start = '2019-08-01'
period_end = '2019-08-28'
##############################################

mask_feature_collection_path = 'USDOS/LSIB_SIMPLE/2017'


# if folder does not exist, create it
if not os.path.exists(save_folder_location):
    os.makedirs(save_folder_location)

##############################################
# Initializing google earth engine and google drive API
##############################################

ee.Initialize()

# If modifying these scopes, delete the file token.json.
SCOPES = 'https://www.googleapis.com/auth/drive'


store = file.Storage('token.json')
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
    creds = tools.run_flow(flow, store)
service = build('drive', 'v3', http=creds.authorize(Http()))



# set the size of one side for each image for prediction (square), in number of pixels, 1 px = 30m * 30m for landsat 7
# 66 * 30 ~ 2km, currently set to 8km * 8km area
target_image_data_size = 66
# set date format here for python datetime conversion (strptime and strftime)
image_date_format = '%Y-%m-%d'

# set time in the future for prediction - currently 30 days in advance
prediction_time_distance = datetime.timedelta(days=30)
# set to predict 1 week worth of fire
prediction_interval_time_distance = datetime.timedelta(days=7)
# set satellite data interval to 16 days - according to landsat 7 data frequency
satellite_revisit_interval = datetime.timedelta(days=16)
# set period in the past to find data for 1 piece of land - currently set to 365 day
satellite_data_period = datetime.timedelta(days=365)

# functions to handle downloading and preprocessing########################################################################
def drive_export_and_preprocess(image, image_id, image_date):
    '''
    export image to google drive, download from google drive
    and convert image to histogram and save
    '''
    drive_filename = image_date + image_id
    taskname = 'export' + '-' + drive_filename
    # ee.batch.Export.image(image, taskname, {})
    # task = ee.batch.Export.image(image,
    #     taskname, {
    #         'driveFolder': 'gee_export_Data',
    #         'driveFileNamePrefix': drive_filename,
    #         'scale':30
    #     }
    # )
    # run it at 1/4 res per side, 1/16 res in area
    task = ee.batch.Export.image(image,
        taskname, {
            'driveFolder': 'gee_export_Data',
            'driveFileNamePrefix': drive_filename,
            'scale':120
        }
    )
    # task = ee.batch.Export.image(image,
    #     taskname, {
    #         'driveFolder': 'gee_export_Data',
    #         'driveFileNamePrefix': drive_filename,
    #         'scale':60
    #     }
    # )


    # this starts the task
    task.start()
    # wait until task is either done or failed
    while(task.status()['state'] != 'COMPLETED'):
        print(task.status()['state'] + ' ' + taskname)
        if (task.status() == 'FAILED'):
            print('FAILED ' + taskname)
            break
        else:
            ### wait for 20 sec since task not completed
            time.sleep(20)

    # supposedly wait for file to appear in drive
    time.sleep(20)

    # assume task is done here, now to download from google drive
    drive_search_query = "mimeType='image/tiff' and (name contains '" + drive_filename + "')"
    # Call the Drive v3 API to search for files, can leave as .list() for everything
    results = service.files().list(q=drive_search_query,
        pageSize=10,
        fields="nextPageToken, files(id, name)").execute()
    drive_items = results.get('files', [])
    drive_item = drive_items[0]

    # to download, need to retrieve file id
    file_id = drive_item['id']
    # we need a file name too
    file_name = drive_item['name']
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        # print('Download %d%%.' % int(status.progress() * 100))



    # now we need to import into python and save the image as histograms
    raster_tiff = gdal.Open(file_name, gdal.GA_ReadOnly)
    num_layer = raster_tiff.RasterCount

    # layer, height, width
    combined_data_array = raster_tiff.ReadAsArray()
    # change to height, width, layer, with label image on top
    combined_data_array = combined_data_array.transpose(1, 2, 0)

    # the first layer is always the label image

    # label_layer = raster_tiff.GetRasterBand(1).ReadAsArray()
    # result_layer_list = [label_layer]

    # for i in range(2, num_layer + 1):
    #     # read the individual layer and add it to list
    #     current_layer = raster_tiff.GetRasterBand(i).ReadAsArray()
    #     result_layer_list.append(current_layer)

    # # stack elements of list into a single numpy array-(width, height, layer)
    # stacked_data = np.stack(result_layer_list, axis=2)

    # # free up some memory here
    # del(result_layer_list)

    ## now to iterate the stacked data into smaller pieces and convert to
    ## histogram to save
    convert_to_histogram(combined_data_array, file_name, target_image_data_size)
    # convert_to_histogram(stacked_data, file_name, target_image_data_size)

    # deleting the file on google drive to free up space
    service.files().delete(fileId=file_id).execute()

    # delete the main downloaded image file, since not needed
    if os.path.exists(file_name):
        os.remove(file_name)

def convert_to_histogram(stacked_array, file_name, image_slice_length):
    '''
    Cuts the stacked array up into smaller pieces
    calculates the histogram for each small piece
    saves it using pickle
    '''
    length = stacked_array.shape[0]
    width = stacked_array.shape[1]
    depth = stacked_array.shape[2]
    # using integer division because we want to floor it
    num_slice_length = length // image_slice_length
    num_slice_width = width // image_slice_length
    # this counter helps locate where is the resultant histogram from later on, if needed
    counter = 0
    # histogram bin boundaries for numpy histogram conversion,
    # we are using 32 bins of 8, ignoreing 0 values because undefined areas are 0
    bin_boundaries = [1, 8, 16, 24, 32, 40, 48, 56, 64, 72,
        80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
        160, 168, 176, 184, 192, 200, 208, 216, 224, 232,
        240, 248, 255]

    for i in range(num_slice_length):
        for  j in range(num_slice_width):
            ## calculating indices to slice at
            length_lower = i * image_slice_length
            length_upper = length_lower + image_slice_length

            width_lower = j * image_slice_length
            width_upper = width_lower + image_slice_length
            # slice area of image_slice_length ** 2 along length and width, keep all depth(layers)
            current_slice = stacked_array[length_lower:length_upper, width_lower:width_upper, :]

            # if slice is all 0, ignore
            if(np.max(current_slice) == 0):
                continue
            # separate the label and the data instance
            current_slice_label = np.max(current_slice[:, :, 0])
            # if the label indicates no hotspot, then we ignore it in this script
            if(current_slice_label == 0):
                counter = counter + 1
                continue
            current_slice = current_slice[:, :, 1:]

            current_layer_list = []
            for k in range(current_slice.shape[2]):
                current_layer_array = current_slice[:, :, k]

                current_layer_histogram, _ = np.histogram(current_layer_array, bins=bin_boundaries, range=(1, 255))
                ## NOTE: only saving non 0 pixels, since 0 -> missing data,
                ## real world almost no 0 pixels in images
                current_layer_list.append(current_layer_histogram)

            current_sliced_stacked_histogram = np.stack(current_layer_list, axis=1)

            slice_filename = os.path.splitext(file_name)[0] + '_slice_' + str(counter) + '.pickle'
            pickle.dump((current_sliced_stacked_histogram, current_slice_label),
                open(save_folder_location + slice_filename, 'wb'))

            counter = counter + 1
    # done with processing slices
#########################################################################



# indonesia_mask = ee.FeatureCollection(mask_feature_collection_path).filter(ee.Filter.eq("Name", 'Indonesia'))
# using new source of country boundary
indonesia_mask = ee.FeatureCollection(mask_feature_collection_path).filter(ee.Filter.eq('country_na', 'Indonesia'))


# for time period, get all landsat images within those bounds
# period_start = '2019-07-01'
# period_end = '2019-07-28'


## select all satellite images of indonesia in that time bound
## NOTE: might have to loop over bounds because max num of images seems to be 5000
satellite_image_collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_RT') \
    .filter(ee.Filter.date(period_start, period_end)) \
    .filterBounds(indonesia_mask.geometry()) \
    .sort('DATE_ACQUIRED', False)

def make_list(current_image, prev_list):
    current_image = ee.Image(current_image)
    prev_list = ee.List(prev_list)
    new_list = prev_list.add(current_image)
    return new_list

image_list = satellite_image_collection.iterate(make_list, ee.List([]))

image_list = ee.List(image_list)
num_images = image_list.size().getInfo()

# now we have a list of images that we want
# for each image we want to find its past and roll it into a stacked image
for i in range(num_images):
    current_image = ee.Image(image_list.get(i))
    image_mask = current_image.geometry()

    image_date = current_image.get('DATE_ACQUIRED').getInfo()
    image_date_datetime = datetime.datetime.strptime(image_date, image_date_format)

    image_id = current_image.getInfo()['id']
    image_id = image_id.replace("/", "-")



    ## get the labels x time in the future
    fire_label_period = image_date_datetime + prediction_time_distance
    ### get the time string for period to search for fire data
    fire_label_start_period = fire_label_period.strftime(image_date_format)
    fire_label_end_period = (fire_label_period + prediction_interval_time_distance).strftime(image_date_format)

    ### retrieving the fire data
    current_image_fire_dataset = ee.ImageCollection('FIRMS') \
        .filter(ee.Filter.date(fire_label_start_period, fire_label_end_period))

    current_image_reduced_fire_dataset = current_image_fire_dataset \
        .reduce(ee.Reducer.max()) \
        .select(['confidence_max'])

    final_fire_label_layer = current_image_reduced_fire_dataset.clip(image_mask)
    ## done getting label

    ## get data
    lower_bound_time = image_date_datetime - satellite_data_period
    current_time = image_date_datetime
    stacked_data_image = current_image.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2', 'B7'])

    # temp = 0
    while current_time > lower_bound_time:
        ### go back 16 days and select all images that filter bounds with area
        current_time = current_time - satellite_revisit_interval
        current_interval_lower = current_time - datetime.timedelta(days=8)
        current_interval_upper = current_time + datetime.timedelta(days=8)

        current_interval_lower_string = current_interval_lower.strftime(image_date_format)
        current_interval_upper_string = current_interval_upper.strftime(image_date_format)

        current_interval_image_collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_RT') \
            .filter(ee.Filter.date(current_interval_lower_string, current_interval_upper_string)) \
            .filterBounds(image_mask)
        ### form a single image by max reduction to the images
        current_interval_reduced_image = current_interval_image_collection.reduce(ee.Reducer.max())

        # print(current_interval_reduced_image.getInfo()['bands'])
        # print(temp)
        # temp = temp + 1
        ### if somehow there is no image in this time interval that overlaps
        ### with anchor image, aka satellite missed this spot in this visit
        if len(current_interval_reduced_image.getInfo()['bands']) == 0:
            # print('Empty image')
            continue
        ### clip the image collection
        # current_interval_reduced_image = current_interval_reduced_image \
        #     .select([0, 1, 2, 3, 4, 5, 6, 7]) \
        #     .clip(image_mask)


        current_interval_reduced_image = current_interval_reduced_image \
            .select(['B1_max', 'B2_max', 'B3_max', 'B4_max', 'B5_max', 'B6_VCID_1_max', 'B6_VCID_2_max', 'B7_max']) \
            .clip(image_mask)

        ### stack to current image
        stacked_data_image = ee.Image(stacked_data_image).addBands(current_interval_reduced_image)

    ## finished with past images, adding label image layer into stacked data image
    final_stacked_data_image = ee.Image(final_fire_label_layer).addBands(stacked_data_image)

    ## now we need to export the image to google drive and then download from google drive
    drive_export_and_preprocess(final_stacked_data_image, image_id, image_date)
