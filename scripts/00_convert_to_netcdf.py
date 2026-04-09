"""
Convert downloaded GeoTIFF stacks to compressed NetCDF files.

Run this script ONCE after downloading all GeoTIFFs from Google Drive.
Input:  data/monthly/{Country}_Monthly_{Var}.tif  (39 files, 312 bands each)
Output: data/netcdf/{Country}_ERA5Land_monthly.nc  (3 files, ~225 MB total)

Usage:
    python 00_convert_to_netcdf.py
"""

import os
import sys
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401 - registers .rio accessor

GEOTIFF_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'GEE_ERA5Land_Monthly')
NETCDF_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'netcdf')

ALL_VARS = [
    'P_mm', 'ET_mm', 'R_sro_mm', 'R_ssro_mm', 'R_mm', 'S_mm',
    'T_C', 'Td_C', 'Rn_sw', 'Rn_lw', 'Wind', 'Ts_C', 'SP'
]

COUNTRIES = ['Saudi', 'Italy', 'Bangladesh']


def convert_country(country):
    arrays = {}
    for var in ALL_VARS:
        fpath = os.path.join(GEOTIFF_DIR, f'{country}_Monthly_{var}.tif')
        if not os.path.exists(fpath):
            print(f'  WARNING: {fpath} not found, skipping.')
            continue

        da = rioxarray.open_rasterio(fpath, masked=True)  # (band, y, x)
        n_bands = da.sizes['band']
        times = pd.date_range('2000-01', periods=n_bands, freq='MS')
        da = (da.assign_coords(band=times)
                .rename({'band': 'time', 'x': 'lon', 'y': 'lat'}))
        da.name = var
        arrays[var] = da
        print(f'  {var}: {n_bands} bands loaded')

    if not arrays:
        print(f'  No files found for {country}, skipping.')
        return

    ds = xr.Dataset(arrays)
    ds.attrs['country'] = country
    ds.attrs['source']  = 'ERA5-Land ECMWF/ERA5_LAND/MONTHLY_AGGR'
    ds.attrs['period']  = '2000-01 to 2025-12'
    ds.attrs['crs']     = 'EPSG:4326'

    encoding = {
        var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
        for var in arrays
    }
    out_path = os.path.join(NETCDF_DIR, f'{country}_ERA5Land_monthly.nc')
    ds.to_netcdf(out_path, encoding=encoding)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  Saved: {out_path}  ({size_mb:.1f} MB)')


def main():
    os.makedirs(NETCDF_DIR, exist_ok=True)

    missing = [c for c in COUNTRIES
               if not any(os.path.exists(os.path.join(GEOTIFF_DIR, f'{c}_Monthly_{v}.tif'))
                          for v in ALL_VARS)]
    if missing:
        print(f'WARNING: No GeoTIFF files found for: {missing}')
        print(f'Expected files in: {os.path.abspath(GEOTIFF_DIR)}')
        print('Download GeoTIFFs from Google Drive first.')
        sys.exit(1)

    for country in COUNTRIES:
        print(f'\nProcessing {country}...')
        convert_country(country)

    print('\nDone. NetCDF files ready in data/netcdf/')
    print('Next step: run scripts/01_water_balance.py')


if __name__ == '__main__':
    main()
