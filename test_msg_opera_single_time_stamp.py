'''

script to read OPERA files. FIles are contained in the bucket with a structure like:
yyyy/mm/dd/yyyy-mm-ddThhMMssZ-reflectivity-composite-opera.h5
We here initially read one file for the case study of the 24th of September 2017 at 16:15 UTC +- 15 minutes.
Cernusco sul Naviglio, Italy (Lombardy region)
45.52 N, 9.32 E < 3km

we test the plotting and we try to match the radar data with te satellite data.
We regrid radar data to the satellite grid and plot the reflectivity histogram.

'''
import pdb
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



out_dir_maps_expats = '/home/claudia/figs/OPERA/expats_reflectivity_maps'
out_dir_maps = '/home/claudia/figs/OPERA/original_reflectivity_maps'
out_dir_hist = '/home/claudia/figs/OPERA/reflectivity_hists'

# select date of case study and return all needed strings
#date = '2017-09-24T16:15:00'  # case study 2017
date_str = '2016-08-20T15:00:00'  # case study 2016

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

prefix, prefix_msg, date, year, month, day, hh, mm = extract_strings_from_date(date_str)

print(prefix)
print(prefix_msg)
print(date)
print(year, month, day, hh, mm)
pdb.set_trace()


# definiting OPERA filename
filename = f"{year}/{month:02d}/{day:02d}/{year}-{month:02d}-{day:02d}T{hh:02d}{mm:02d}00Z-reflectivity-composite-opera.h5"

# reading msg data
ds_msg = xr.open_dataset('/home/claudia/downloads/'+f"{year}{month:02d}{day:02d}-EXPATS-RG.nc")  # case study 2016

# select the time closest to the one of the radar data
ds_msg = ds_msg.sel(time=date, method='nearest')  # case study 2016
print(f"Selected MSG data for time: {ds_msg.time.values}")

# Read the specific file
print(f"Reading file: {filename}")

# Read OPERA data using custom S3 function
ds_OPERA = read_h5_from_s3(s3, S3_BUCKET_NAME, filename)
print(ds_OPERA)

# crop radar data on expats domain
domain_EXPATS = [42., 5., 52., 16.]  # latmin, lonmin, latmax, lonmax
ds_expats_opera = ds_OPERA.sel(lat=slice(52,42), lon=slice(5,16))

# set Ze to nan where it is below a ze_min threshold
ze_min = ds_expats_opera['reflectivity'].min().values
ze_max = ds_expats_opera['reflectivity'].max().values
ds_expats_opera['reflectivity'] = ds_expats_opera['reflectivity'].where(ds_expats_opera['reflectivity'] >= ze_min, np.nan)

# resample ds_expats to latlon grid of ds_msg
ds_expats_opera = ds_expats_opera.interp(lat=ds_msg['lat'], lon=ds_msg['lon'], method='nearest')

# print max and min of reflectivity
print("Max reflectivity:", ds_expats_opera['reflectivity'].max().values)
print("Min reflectivity:", ds_expats_opera['reflectivity'].min().values)


###### PLOTTING ############
# plot radar reflectivity map at its native resolution
print('Plotting radar reflectivity map at its native resolution...')
plot_data(ds_OPERA, out_dir_maps)

# plot radar reflectivity map at its native resolution
print('Plotting radar reflectivity map over expats domain interpolated to MSG grid...')
plot_data(ds_expats_opera, out_dir_maps_expats)


# plot msh 10.8 channel 
lon_msg = ds_msg['lon'].values
lat_msg = ds_msg['lat'].values
channel = 'IR_108'

print('plotting MSG 10.8 micron channel over expats domain...')
# plot the 10.8 micron channel                
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
img = ax.pcolormesh(lon_msg, lat_msg, ds_msg.IR_108.values, transform=ccrs.PlateCarree(),
                    cmap='Grays', vmin=240, vmax=300)
cbar = plt.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, pad=0.02)
cbar.ax.tick_params(labelsize=14)
cbar.set_label(f'10.8 microns', fontsize=14)

ax.coastlines(resolution='10m', color='orangered', linewidth=3)
ax.add_feature(cfeature.BORDERS, color='orangered', linewidth=3)

plt.tight_layout()
fig.savefig(f"{out_dir_maps_expats}/{channel}_plot_{date}.png",
            dpi=300, bbox_inches='tight')
plt.close(fig)


# plot OPERA and msg channels together
print('plotting MSG 10.8 micron channel and OPERA reflectivity together for expats domain..')
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
img = ax.pcolormesh(lon_msg, lat_msg, ds_msg.IR_108.values, transform=ccrs.PlateCarree(),
                    cmap='Grays', vmin=240, vmax=300)
# mask reflectivity data smaller than Ze_min 
Ze_min = -10  # Example threshold
reflectivity = ds_expats_opera['reflectivity']
reflectivity = reflectivity.where(reflectivity >= Ze_min, np.nan)

LAT = ds_expats_opera['lat']
LON = ds_expats_opera['lon']

img2 = ax.pcolormesh(LON, 
                        LAT,
                        reflectivity,
                        transform=ccrs.PlateCarree(), 
                        cmap='turbo',
                        shading='auto',
                        vmin=Ze_min, 
                        vmax=50)

cbar = plt.colorbar(img2, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
cbar.ax.tick_params(labelsize=14)
cbar.set_label(f'Reflectivity (dBZ)', fontsize=14)
cbar_msg = plt.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, pad=0.02)
cbar_msg.ax.tick_params(labelsize=14)
cbar_msg.set_label(f'10.8 microns', fontsize=14)

ax.coastlines(resolution='10m', color='orangered', linewidth=3)
ax.add_feature(cfeature.BORDERS, color='orangered', linewidth=3)

plt.tight_layout()
fig.savefig(f"{out_dir_maps_expats}/msg_radar_plot_{date}.png",
            dpi=300, bbox_inches='tight')
plt.close(fig)
