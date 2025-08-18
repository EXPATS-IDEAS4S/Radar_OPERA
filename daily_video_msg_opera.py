import os
import boto3
import logging
import h5py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from botocore.exceptions import ClientError
import io
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from mpl_style import CMAP, plot_cities_expats


# Import S3 credentials
from credentials_buckets import S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
from aux_funcs import list_s3_files, read_h5_from_s3, plot_data, open_with_wrl, plot_msg, gif_maker, convert_gif_to_mp4

import pdb  # For debugging purposes


def extract_strings_from_date(date_str):
    
    year = date_str[0:4]
    month = date_str[5:7]
    day = date_str[8:10]
    hh = date_str[11:13]
    mm = date_str[14:16]

    prefix = f"{year}/{month}/{day}/"
    prefix_msg = f"{year}/{month}/"
    date = f"{year}-{month}-{day}T{hh}:{mm}:00"
    # Return all extracted strings
    return prefix, prefix_msg, date, year, month, day, hh, mm


# Initialize S3 client
s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY
)

# define date of case study and return all needed string
date_case_study = '2016-08-20T00:00:00'  # case study 2016

print(f"Extracting strings from date: {date_case_study}")
# Extract strings from date
prefix, prefix_msg, date, year, month, day, hh, mm = extract_strings_from_date(date_case_study)

# loop on 15 mins intervals in the day 
# filename '2016/08/20/2016-08-20T233000Z-reflectivity-composite-opera.h5'


# reading list of OPERA files for the day
daily_files = list_s3_files(s3, S3_BUCKET_NAME, prefix)
print(f'found {len(daily_files)} files for date: {date}')

# Sort files by time
daily_files.sort()


# Loop through each file for the specific day
print(f"Processing files for date: {date}")
ds_day = []  # Initialize a list to store datasets for the day

for file in daily_files:
    print(f"Processing file: {file}")
    
    # Read the specific date for the file
    date_min = file.split('/')[-1].split('-')[0:3]
    print(f"Reading file: {date_min}")
    ds = read_h5_from_s3(s3, S3_BUCKET_NAME, file)
    
    # Only append if dataset was successfully loaded
    if ds is not None:
        ds_day.append(ds)
        

# Concatenate all datasets for the day along the time dimension
if ds_day:
    ds_OPERA = xr.concat(ds_day, dim='time')

ds_OPERA.attrs['description'] = f"OPERA Reflectivity data for {date}"
print(ds_OPERA)


# reading msg data
ds_msg = xr.open_dataset('/home/claudia/downloads/'+f"{year}{month}{day}-EXPATS-RG.nc")  # case study 2016

# crop radar data on expats domain
ds_expats_opera = ds_OPERA.sel(lat=slice(52,42), lon=slice(5,16))

# set Ze to nan where it is below a ze_min threshold
ze_min = ds_expats_opera['reflectivity'].min().values
ze_max = ds_expats_opera['reflectivity'].max().values
ds_expats_opera['reflectivity'] = ds_expats_opera['reflectivity'].where(ds_expats_opera['reflectivity'] >= ze_min, np.nan)

# resample ds_expats to latlon grid of ds_msg
ds_expats_opera = ds_expats_opera.interp(lat=ds_msg['lat'], lon=ds_msg['lon'], method='nearest')

print(ds_expats_opera)
print(ds_msg)
#pdb.set_trace()  # Debugging point

###### PLOTTING VIDEOS OF MSG and OPERA DATA ######
# creating output directories for plots and videos for the selected date
output_dir = f"/home/claudia/figs/{year}{month}{day}/"
os.makedirs(output_dir, exist_ok=True)


time = ds_msg['time'].values

# loop on time stamps to plot radar and msg data
for t in time:
    
    print(f"Processing time: {t}")
    
    # Select data for the current time
    ds_msg_t = ds_msg.sel(time=t, method='nearest')
    ds_expats_opera_t = ds_expats_opera.sel(time=t, method='nearest')
    date_frame = ds_msg_t['time'].values
    date_str = str(date_frame)[:19]
    print(date_str)
    
    # call plotting function for msg and for opera data
    lons = ds_msg_t['lon'].values
    lats = ds_msg_t['lat'].values
    variable = ds_msg_t['IR_108'].values
    label = date_str+'_IR108'
    cmap = "Grays"
    cbar_title = 'IR_10.8 (micron)'
    domain = [5., 16, 42, 52.]
    path_out = output_dir
    key = 'expats'
    back_transparent = True
    vmin = 200
    vmax = 310
    
    plot_msg(lons, lats, variable, label, cmap, cbar_title, domain, path_out, key, back_transparent, vmin, vmax)
    print(f"Plotted MSG data for time: {t}")
    
    plot_data(ds_expats_opera_t, output_dir)
    print(f"Plotted OPERA data for time: {t}")
    

# creating videos from the plots
video_path = f"/home/claudia/videos/{year}{month}{day}/"
os.makedirs(video_path, exist_ok=True)

# Create videos from the plots
if not os.path.exists(f"/home/claudia/videos/{year}{month}{day}/msg_video.gif"):
    # create animated gif of list of plots
    gif_maker(path_out, "msg_video", video_path, 250, "108", "_IR108_expats.png")
    # image_folder, gif_name, gif_path, gif_duration, channel, png_string


    # convert to mp4
    convert_gif_to_mp4(f"/home/claudia/videos/{year}{month}{day}/", f"/home/claudia/videos/{year}{month}{day}/")

if not os.path.exists(f"/home/claudia/videos/{year}{month}{day}/opera_video.gif"):
    
    # create animated gif of list of plots
    gif_maker(path_out, "opera_video", f"/home/claudia/videos/{year}{month}{day}/", 250, "Ze", "_OPERA_reflectivity.png")
    
        # convert to mp4
    convert_gif_to_mp4(f"/home/claudia/videos/{year}{month}{day}/", f"/home/claudia/videos/{year}{month}{day}/")
