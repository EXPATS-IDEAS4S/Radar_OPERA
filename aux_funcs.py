import os
import pdb
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
from datetime import datetime
from pyproj import Proj, Transformer
import wradlib as wrl


def open_with_wrl(fpath):
    f = wrxl.util.get_wradlib_data_file(fpath)
    fcontent = wrl.io.read_opera_hdf5(f)
    # which keyswords can be used to access the content?
    print(fcontent.keys())
    # print the entire content including values of data and metadata
    # (numpy arrays will not be entirely printed)
    #print(fcontent["dataset1/data1/data"])
    #fig = plt.figure(figsize=(10, 10))
    da = wrl.georef.create_xarray_dataarray(
        fcontent["dataset1/data1/data"]
    ).wrl.georef.georeference()
    #im = da.wrl.vis.plot(fig=fig, crs="cg")
    #fig.savefig(output_path+'prova.png')
    #plt.close()
    return da  # Return the xarray DataArray for further processing or visualization
#


def list_s3_files(s3, bucket, prefix):
    """List all files in an S3 bucket under a given prefix, handling pagination."""
    file_keys = []
    continuation_token = None

    try:
        while True:
            if continuation_token:
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=continuation_token)
            else:
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

            if 'Contents' in response:
                file_keys.extend([obj['Key'] for obj in response['Contents']])

            # Check if there are more objects to retrieve
            if response.get('IsTruncated'):  # More results exist
                continuation_token = response['NextContinuationToken']
            else:
                break  # No more results

    except ClientError as e:
        logging.error(e)
        return []

    return file_keys


def print_h5_structure(h5file, level=0):
    """ Recursively prints the structure of an HDF5 file """
    for key in h5file.keys():
        print("  " * level + f"- {key}")  # Indentation for hierarchy
        if isinstance(h5file[key], h5py.Group):  # If it's a group, recurse
            print_h5_structure(h5file[key], level + 1)

def reproject_to_wgs84(proj_def, LON, LAT):
    """
    Converts projected coordinates to WGS84 latitude and longitude.
    
    Args:
        proj_def (str): Projection definition string (PROJ.4 format).
        LON (ndarray): 2D array of longitudes in projected coordinates.
        LAT (ndarray): 2D array of latitudes in projected coordinates.

    Returns:
        tuple: (lon_corrected, lat_corrected) in WGS84
    """
    if proj_def:
        try:
            proj_laea = Proj(proj_def)  # Define source projection

            transformer = Transformer.from_proj(proj_laea, "epsg:4326", always_xy=True)  # Convert to lat/lon (WGS84)
            # Flatten the 2D lat/lon arrays to 1D
            lon_flat = LON.flatten()
            lat_flat = LAT.flatten()

            # Transform coordinates
            lon_transformed, lat_transformed = transformer.transform(lon_flat, lat_flat)

            # Reshape back to original grid shape
            lon_corrected = lon_transformed.reshape(LON.shape)
            lat_corrected = lat_transformed.reshape(LAT.shape)
        
            return lon_corrected, lat_corrected
        except Exception as e:
            logging.error(f"Projection transformation failed: {e}")
            return LON, LAT  # Return original values if transformation fails
    return LON, LAT  # If no projection, return as-is

def read_h5_from_s3(s3, bucket, file_key):
    """Reads an HDF5 file from S3 and returns an xarray dataset."""
    try:
        # Load file from S3
        obj = s3.get_object(Bucket=bucket, Key=file_key)
        file_stream = io.BytesIO(obj['Body'].read())  # Convert to a seekable file-like object

        with h5py.File(file_stream, 'r') as f:
            print("Reading HDF5 file...")
            
            # ---- Extract Data ----
            dataset1 = f['dataset1']
            dataset2 = f['dataset2']  # Quality index (QIND)
            what1 = dataset1['what']

            nodata_value = what1.attrs['nodata']
            undetect_value = what1.attrs['undetect']

            # Reflectivity data (assuming shape is [lat, lon])
            reflectivity = np.array(dataset1['data1/data'])

            # Set nodata & undetect values to NaN
            #reflectivity[reflectivity == nodata_value] = np.nan
            #reflectivity[reflectivity == undetect_value] = np.nan

            # ---- Extract Spatial Metadata ----
            where_data = f['where']

            # Grid size
            xsize = where_data.attrs['xsize']
            ysize = where_data.attrs['ysize']

            # Pixel scale (meters per pixel)
            xscale = where_data.attrs['xscale']  
            yscale = where_data.attrs['yscale']  

            # Read grid corners (geographic coordinates)
            ll_lon, ll_lat = where_data.attrs['LL_lon'], where_data.attrs['LL_lat']
            ul_lon, ul_lat = where_data.attrs['UL_lon'], where_data.attrs['UL_lat']
            ur_lon, ur_lat = where_data.attrs['UR_lon'], where_data.attrs['UR_lat']
            lr_lon, lr_lat = where_data.attrs['LR_lon'], where_data.attrs['LR_lat']
                        

            # ---- Generate Latitude & Longitude Grids ----
            lats = np.linspace(ul_lat, ll_lat, int(ysize))  # Latitude from upper to lower
            lons = np.linspace(ul_lon, ur_lon, int(xsize))  # Longitude from left to right
            #print(lats)

            # Create 2D meshgrid
            LAT, LON = np.meshgrid(lats, lons, indexing='ij')
            #print(LON)
            #print(LAT)

            # ---- Projection Definition ----
            proj_def = where_data.attrs.get('projdef', '').decode('utf-8') if isinstance(where_data.attrs.get('projdef', ''), bytes) else where_data.attrs.get('projdef', '')
            #print(proj_def)
            #exit()
            # ---- Convert to WGS84 if needed ----
            #lon_corrected, lat_corrected = reproject_to_wgs84(proj_def, LON, LAT)
            #print(lon_corrected)
            #print(lat_corrected)
            #exit()

            # ---- Extract Time ----
            what_data = f['what']  # Time-related metadata
            date_str = what_data.attrs['date'].decode('utf-8')  # Example: '20130401'
            time_str = what_data.attrs['time'].decode('utf-8')  # Example: '000000'

            # Convert to datetime format
            timestamp = np.datetime64(datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S"))

            ds = xr.Dataset(
                {
                    "reflectivity": (["lat", "lon"], reflectivity)  # Main data variable
                },
                coords={
                    "time": timestamp,  # Time coordinate
                    "lat": lats,  # Grid index for y-axis
                    "lon": lons,  # Grid index for x-axis
                },
                attrs={
                    "projection": proj_def,  # Store projection metadata
                    "nodata": nodata_value,  # Add nodata as an attribute
                    "undetect": undetect_value  # Add undetect as an attribute
                }
            )

            # Add lat/lon grids as auxiliary 2D variables
            #ds["latitude"] = (["y", "x"], LAT)
            #ds["longitude"] = (["y", "x"], LON)

            print("Successfully created xarray dataset!")
            return ds

    except Exception as e:
        logging.error(f"Error reading {file_key}: {e}")
        return None


def merge_and_save_netcdf(datasets, output_file):
    """ Merge multiple datasets and save as compressed NetCDF """
    merged_ds = xr.concat(datasets, dim='time')
    merged_ds.to_netcdf(output_file, engine='netcdf4', encoding={'reflectivity': {'zlib': True, 'complevel': 5}})
    print(f"Saved NetCDF: {output_file}")


def plot_reflectivity_histogram(dataset, output_dir, bins=50, log_scale=True):
    """Plots a histogram of reflectivity values and saves it as an image."""
    
    # Extract reflectivity values, flattening the array
    reflectivity_values = dataset['reflectivity'].values.flatten()

    reflectivity_values = reflectivity_values[reflectivity_values < -1000]
    print(np.unique(reflectivity_values))
    
    # Remove invalid values (assuming -9999000 is a nodata value)
    reflectivity_values = reflectivity_values[reflectivity_values > -9999000]

    # Extract timestamp for filename
    timestamp = str(dataset['time'].values)[:19].replace(":", "").replace("-", "").replace("T", "_")
    
    # Generate output filename
    output_path = os.path.join(output_dir, f"reflectivity_histogram_{timestamp}.png")

    # Create the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(reflectivity_values, bins=bins, color='blue', alpha=0.7, edgecolor='black')

    # Log scale option
    if log_scale:
        plt.yscale('log')

    # Labels and title
    plt.xlabel("Reflectivity")
    plt.ylabel("Frequency (log scale)" if log_scale else "Frequency")
    plt.title(f"Reflectivity Histogram - {timestamp}")

    # Save and close the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    print(f"Histogram saved to {output_path}")


def plot_data(dataset, output_dir):
    """Plots reflectivity data using Cartopy and saves the output with a timestamped filename."""
    
    # Extract timestamp for filename
    timestamp = dataset.time.values
    
    # Generate output filename
    output_path = os.path.join(output_dir, f"opera_reflectivity_{timestamp}.png")

    # build string from timestamp
    timestamp_str = str(timestamp)[:19].replace(":", "").replace("-", "").replace("T", "_")
    output_path = os.path.join(output_dir, f"{timestamp_str}_OPERA_reflectivity.png")  
    
    # Set up the figure
    #proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=55)
    proj = ccrs.PlateCarree()  # Use Plate Carree projection for simplicity

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj})

    # Get reflectivity data
    reflectivity = dataset['reflectivity']
    
    # mask reflectivity data smaller than Ze_min 
    Ze_min = -10  # Example threshold
    reflectivity = reflectivity.where(reflectivity >= Ze_min, np.nan)
    
    LAT = dataset['lat']
    LON = dataset['lon']
    
    mesh = ax.pcolormesh(LON, 
                         LAT,
                         reflectivity,
                         transform=ccrs.PlateCarree(), 
                         cmap='turbo',
                         shading='auto',
                         vmin=-10, 
                         vmax=50)

    # Add coastlines and borders
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label("Reflectivity (dBZ)")

    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='gray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Set lat/lon gridlines
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
    gl.top_labels = False   # Remove top labels
    gl.right_labels = False  # Remove right labels

    # Title and save
    ax.set_title(f"Reflectivity Data - {timestamp}", fontsize=12)
    
    # Save with tight layout
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    print(f"Plot saved to {output_path}")
    
    
    
    


def plot_msg(lons, lats, variable, label, cmap, cbar_title, domain, path_out, key, back_transparent, vmin, vmax):
    """
    plot map of the variable over the defined domain

    Args:
        lons (array): longitude values
        lats (array): latitudes 
        variable (matrix): variable to plot
        label (string): string for plot filename
        cmap (colormap): color map 
        cbar_title (string): title for color bar
        domain (array): minlon, maxlon, minlat, maxlat     

        path_out (string): path where to save the plot
        key (string): string possible "expats" or "dfg"
        back_transparent (boolean): True or False - true means transparent background
        vmin (float): value min in colorbar
        vmax (float): max value in colorbar
        
    Dependencies:
    plot_cities_expats, 
    plot_local_dfg
    
    """
    from mpl_style import plot_cities_expats
    
    # Plot the map of MSG channels vs lat/lon
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.spines["top"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.set_extent(domain)
    
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 14}
    gl.ylabel_style = {'fontsize': 14}
    
    

    var_levels = np.linspace(vmin, vmax, 20)
    # plot 10th percentile as filled contours
    mesh = ax.contourf(lons, 
                        lats, 
                        variable, 
                        cmap=cmap, 
                        transform=ccrs.PlateCarree(), 
                        levels=var_levels, 
                        vmin=vmin, # 230
                        vmax=vmax) # 270
    cbar = plt.colorbar(mesh, label=cbar_title, shrink=0.6)

    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)
    
     #vmin, vmax=220.,
    plt.tick_params(axis='both',which='major',labelsize=14)

    ax.set_xlabel('Latitude [$^{\circ}$]')
    ax.set_ylabel('Longitude [$^{\circ}$]')
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3)
    #cbar = plt.colorbar(pc,ax=ax,shrink=0.75)
    #cbar.set_label(label, fontsize=14)
    #cbar.ax.tick_params(labelsize=14)

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.5, color='black')
    #ax.add_feature(cfeature.STATES, linewidth=0.2)
    ax.add_feature(cfeature.BORDERS, linewidth=1.5, color='black')

    # Title and save date_str+'_IR108'
    ax.set_title(label, fontsize=12)
    
    if key =='expats':
        plot_cities_expats(ax, 'grey', 50)
    elif key == 'dfg':
        plot_local_dfg(ax, 'black', 50)
        
        
    plt.savefig(
        os.path.join(path_out, label+"_"+key+".png"),
        dpi=300,
        bbox_inches="tight",
        transparent=back_transparent,
        )

    plt.close()
#f"{i:03d}"

    
    
    
def gif_maker(image_folder, gif_name, gif_path, gif_duration, channel, png_string):
    """
    script to create animated gif from a folder containing images

    Args:
        image_folder (string): folder containing png images
        gif_name (string): string as filename for gif
        gif_path (string): path for gif file
        gif_duration (int): duration for gif (typical 250)
        channel(string): variable string for the gif
        png_string (string): string to filter images, e.g. "_IR_108_expats.png" or "_OPERA_reflectivity.png"

    """
    from PIL import Image
    import glob
    
    import matplotlib.pyplot as plt
    

    
    # read files into the fil elist
    image_array = []
        
    
    for file in sorted(glob.glob(image_folder+'*'+png_string)):
                
        image = Image.open(file)
        image_array.append(image)

    im = image_array[0]            
    im.save(gif_path+gif_name+".gif", 
            format='png',
            save_all=True, 
            append_images=image_array, 
            duration=gif_duration, 
            loop=0)
    

    
    return

def convert_gif_to_mp4(gif_path, mp4_path):
    """
    Converts a GIF file to MP4 format using ffmpeg.
    
    Args:
        gif_path (str): Path to the input GIF file.
        mp4_path (str): Path where the output MP4 file will be saved.
    """
    import os
    import subprocess

    # Root directory containing GIFs
    input_root = gif_path

    # Directory where MP4s will be saved
    output_root = mp4_path
    os.makedirs(output_root, exist_ok=True)

    # Output video settings
    output_fps = 10

    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if filename.lower().endswith(".gif"):
                gif_path = os.path.join(dirpath, filename)

                # Get relative path from input_root and use it to build output path
                rel_path = os.path.relpath(dirpath, input_root)
                output_dir = os.path.join(output_root, rel_path)
                os.makedirs(output_dir, exist_ok=True)

                mp4_filename = os.path.splitext(filename)[0] + ".mp4"
                mp4_path = os.path.join(output_dir, mp4_filename)

                if os.path.exists(mp4_path):
                    print(f"🔁 Skipping existing MP4: {mp4_path}")
                    continue

                print(f"🎞️ Converting: {gif_path} -> {mp4_path}")

                try:
                    subprocess.run([
                        "ffmpeg",
                        "-y",  # overwrite output if needed
                        "-i", gif_path,
                        "-vf", f"fps={output_fps},pad=ceil(iw/2)*2:ceil(ih/2)*2",  # even dimensions
                        "-c:v", "libx264",
                        "-preset", "slow",         # compression trade-off: slower = smaller file
                        "-crf", "32",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "faststart",
                        mp4_path
                    ], check=True)
                    print(f"✅ Saved MP4: {mp4_path}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ FFmpeg failed for {gif_path}: {e}")

