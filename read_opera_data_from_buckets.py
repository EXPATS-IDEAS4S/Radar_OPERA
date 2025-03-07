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

# Import S3 credentials
from credentials_buckets import S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
from aux_funcs import list_s3_files, print_h5_structure, read_h5_from_s3, merge_and_save_netcdf, plot_reflectivity_histogram, plot_data, open_with_wrl

# Initialize S3 client
s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY
)

# # List the objects in our bucket
# response = s3.list_objects(Bucket=S3_BUCKET_NAME)
# for item in response['Contents']:
#     print(item['Key'])


# Define year and month range
years = range(2013, 2017)
months = range(4, 10)

out_dir_maps = '/data1/fig/OPERA/original_reflectivity_maps'
out_dir_hist = '/data1/fig/OPERA/reflectivity_hists'

# Collect matching files
datasets = {}
for year in years:
    for month in months:
        prefix = f"{year}/{month:02d}/"
        print(prefix)
        files = list_s3_files(s3, S3_BUCKET_NAME, prefix)
        print(len(files))
        
        for file in files:
            day = file.split('/')[2]  # Extract day from path
            ds = read_h5_from_s3(s3, S3_BUCKET_NAME, file)
            open_with_wrl(file, out_dir_maps)
            exit()
            print(ds)
            print(ds.sel(lat=50, lon=40, method='nearest'))

            #plot_reflectivity_histogram(ds, out_dir_hist, bins=50, log_scale=True)
            #plot_data(ds, out_dir_maps)
            exit()
            if ds is not None:
                datasets.setdefault(day, []).append(ds)

# Process and save daily NetCDF files
for day, ds_list in datasets.items():
    output_filename = f"output_{day}.nc"
    merge_and_save_netcdf(ds_list, output_filename)
    
    # Optional: Plot first dataset
    plot_data(ds_list[0])

