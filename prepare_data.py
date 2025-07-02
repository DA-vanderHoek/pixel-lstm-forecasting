# prepare_data.py

import os
import numpy as np
import xarray as xr
import cftime

# --------------- Configuration --------------- #
RAW_DIR = "raw_data"
OUT_DIR = "prepared_data"
os.makedirs(OUT_DIR, exist_ok=True)

# --------- Load Datasets --------- #
def load_dataset(filename, var=None):
    ds = xr.open_dataset(os.path.join(RAW_DIR, filename))
    return ds[var] if var else ds

evi = load_dataset('EVI_std_s_africa_l.nc')
spei1 = load_dataset('SPEI01_monthly_2000_2023_0_1_degree_s_africa_l.nc')
spei3 = load_dataset('SPEI03_monthly_2000_2023_0_1_degree_s_africa_l.nc')
precip = load_dataset('era5_total_precipitation_1950_2023_monthly_0_1_bil_regridded_s_africa_l.nc')['tp']
temp_raw = load_dataset('era5_2m_temperature_1950_2023_monthly_0_1_remapnn_s_africa_l.nc')
wtd = load_dataset('WTD_monthlymeans_2004_2014_regrid_0_1_degrees_s_africa_l.nc')['WTD'].mean('time')
sand = load_dataset('sandfraction01d_s_africa_l.nc')['GLDAS_soilfraction_sand'].isel(time=0)
elev = load_dataset('GMTED2010_15n015_regrid_0_1_degrees_s_africa_l.nc')['elevation']

# --------- Fix TEMP Time Axis --------- #
time_values = temp_raw['time'].values
units = temp_raw['time'].attrs['units']
calendar = temp_raw['time'].attrs.get('calendar', 'standard')
dates = cftime.num2date(time_values, units, calendar)
temp_raw['time'] = ('time', dates)
temp_raw['time'] = temp_raw.indexes['time'].to_datetimeindex()
temp = temp_raw['t2m']

# --------- Standardization Utilities --------- #
def standardize_monthly(xr_data):
    monthly_mean = xr_data.groupby('time.month').mean('time')
    monthly_std = xr_data.groupby('time.month').std('time')
    return xr.apply_ufunc(lambda x, m, s: (x - m) / s, xr_data.groupby('time.month'), monthly_mean, monthly_std)

def transform_precip(x): return np.log1p(x * 10000)
def transform_wtd(x): return np.log1p((x - 1) * -1)

# --------- Apply Preprocessing --------- #
PREC = standardize_monthly(xr.apply_ufunc(transform_precip, precip))
TEMP = standardize_monthly(temp)
SPEI1 = spei1.__xarray_dataarray_variable__
SPEI3 = spei3.__xarray_dataarray_variable__
EVI = evi.EVI
WTD = xr.apply_ufunc(transform_wtd, wtd)
evi_static = EVI.isel(time=0)

# --------- Coordinate Grid & Valid Pixels --------- #
lat = EVI['lat'].values
lon = EVI['lon'].values
lon_grid, lat_grid = np.meshgrid(lon, lat)
coords = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)

# Build X, y, static
X = np.stack([
    SPEI3.values.reshape(275, -1),
    SPEI1.values.reshape(275, -1),
    PREC.values.reshape(275, -1),
    TEMP.values.reshape(275, -1)
], axis=2)

y = EVI.values.reshape(275, -1)

static = np.stack([
    WTD.values.reshape(-1),
    sand.values.reshape(-1),
    elev.values.reshape(-1),
    evi_static.values.reshape(-1),
], axis=1)

# --------- Mask NaNs --------- #
nan_mask_X = np.any(np.isnan(X), axis=(0, 2))
nan_mask_static = np.any(np.isnan(static), axis=1)
nan_mask_y = np.any(np.isnan(y), axis=0)
valid_mask = ~(nan_mask_X | nan_mask_static | nan_mask_y)

X = X[:, valid_mask, :]
static = static[valid_mask]
y = y[:, valid_mask]
coords = coords[valid_mask]

# --------- Normalize Static --------- #
static = (static - static.mean(axis=0)) / static.std(axis=0)

# --------- Save Outputs --------- #
np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "static.npy"), static)
np.save(os.path.join(OUT_DIR, "y.npy"), y)
np.save(os.path.join(OUT_DIR, "coords_lan_lon.npy"), coords)

print("✅ Prepared data saved to:", OUT_DIR)
print(f"  X: {X.shape}, y: {y.shape}, static: {static.shape}, coords: {coords.shape}")