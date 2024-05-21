import rioxarray
import rasterio
import datetime
import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import zeropad_strint

hj_base = '/gws/nopw/j04/hydro_jules/data/uk/'
data_dir = '/gws/nopw/j04/ceh_generic/netzero/downscaling/ceh-gear/'
sm_data_dir = hj_base + '/soil_moisture_map/output/netcdf/SM/'
home_data_dir = '/home/users/doran/data_dump/'

# open catchment boundaries
gdf = gpd.read_file(home_data_dir + '/nrfa/V12_NRFA_CatchmentsAll.gpkg')

def grab_catchment_rainfall(catch_id, start_date, end_date,
                            spatial=False, land_attrs=False):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    dates = pd.date_range(start_date, end_date, freq="H")

    # process desired date range
    uniq_years = np.unique(dates.year)
    uniq_months = {str(yy) : np.unique(dates.month[np.where(dates.year==yy)]) for yy in uniq_years}
    
    rf_list = []
    for yy in uniq_years:
        for mm in uniq_months[str(yy)]:
            # open specific month of hourly gear and grab rainfall element
            cds = rioxarray.open_rasterio(data_dir + f'/{yy}/CEH-GEAR-1hr-v2_{yy}{zeropad_strint(mm)}.nc', decode_coords="all")
            rfd = cds[2].drop_vars(['lat', 'lon', 'min_dist', 'stat_disag'])
            # write BNG crs
            rfd.rio.write_crs("epsg:27700", inplace=True)
            #print(rfd.rio.crs)
            del(cds)
            rf_list.append(rfd)

    catch_shape = gdf.loc[gdf['CatchID']==str(catch_id)].geometry.values
    xy_bounds = catch_shape.bounds
    x_inds = np.intersect1d(np.where(rf_list[0].x >= xy_bounds[0,0])[0], np.where(rf_list[0].x <= xy_bounds[0,2])[0])
    y_inds = np.intersect1d(np.where(rf_list[0].y >= xy_bounds[0,1])[0], np.where(rf_list[0].y <= xy_bounds[0,3])[0])
    for i in range(len(rf_list)):
        # trim down using the bounds first
        catch_dat = rf_list[i].isel(x=x_inds).isel(y=y_inds).copy()
        catch_dat = catch_dat.rio.clip(catch_shape, rf_list[i].rio.crs)
        rf_list[i] = catch_dat
    
    # join list on time dim
    rfd = xr.concat(rf_list, "time")

    # hack using numpy datetime to subset datetimes
    rfd['time'] = rfd.time.astype("datetime64[ns]")
    rfd = rfd.sel(time = dates)

    # mask fill values with nans
    rfd = rfd.where(rfd.rainfall_amount != -999)

    # calculate catchment average rainfall
    ca_rain = rfd.rainfall_amount.mean(dim=['x', 'y'])

    results = {}
    results['ca_timeseries'] = ca_rain

    if spatial:
        # either return the full spatial timeseries
        results['spatial_timeseries'] = rfd
        
        # or calculate a heat map of the rainfall over the period
        heatmap = rfd.rainfall_amount.sum(dim=['time'])
        results['heatmap'] = heatmap

        if land_attrs:
            # also pull out data on land cover, topography and soils
            pass

    return results
    
def grab_catchment_soilwetness(catch_id, start_date,
                               spatial=False, land_attrs=False):
    start_date = pd.to_datetime(start_date)
    prev_day = start_date - datetime.timedelta(days=1)
    this_day_weight = (start_date.hour + start_date.minute/60)/24
    
    vwc_bounds = rioxarray.open_rasterio(sm_data_dir + '/vwc_bounds.nc')
    
    this_sm = rioxarray.open_rasterio(sm_data_dir + f'/{start_date.year}/SM_{start_date.year}{zeropad_strint(start_date.month)}.nc')
    if not ((start_date.year==prev_day.year) and (start_date.month==prev_day.month)):
        prev_sm = rioxarray.open_rasterio(sm_data_dir + f'/{prev_day.year}/SM_{prev_day.year}{zeropad_strint(prev_day.month)}.nc')
        prev_sm = prev_sm.isel(time=prev_day.day-1).theta # zero-indexed so -1 from day
        this_sm = this_sm.isel(time=start_date.day-1).theta # zero-indexed so -1 from day
    else:
        prev_sm = this_sm.isel(time=prev_day.day-1).theta # zero-indexed so -1 from day
        this_sm = this_sm.isel(time=start_date.day-1).theta # zero-indexed so -1 from day
    
    this_sm.rio.write_crs("epsg:27700", inplace=True)
    prev_sm.rio.write_crs("epsg:27700", inplace=True)
    vwc_bounds.rio.write_crs("epsg:27700", inplace=True)
    
    catch_shape = gdf.loc[gdf['CatchID']==str(catch_id)].geometry.values
    xy_bounds = catch_shape.bounds
    x_inds = np.intersect1d(np.where(this_sm.x >= xy_bounds[0,0])[0], np.where(this_sm.x <= xy_bounds[0,2])[0])
    y_inds = np.intersect1d(np.where(this_sm.y >= xy_bounds[0,1])[0], np.where(this_sm.y <= xy_bounds[0,3])[0])
    
    # trim down using the bounds first
    this_sm = this_sm.isel(x=x_inds).isel(y=y_inds)
    this_sm = this_sm.rio.clip(catch_shape, this_sm.rio.crs)
    
    prev_sm = prev_sm.isel(x=x_inds).isel(y=y_inds)
    prev_sm = prev_sm.rio.clip(catch_shape, prev_sm.rio.crs)
    
    vwc_bounds = vwc_bounds.isel(x=x_inds).isel(y=y_inds)
    vwc_bounds = vwc_bounds.rio.clip(catch_shape, vwc_bounds.rio.crs)
    
    # do weighted average of previous and current day    
    antecedent_vwc = 0.5 * (this_sm * this_day_weight + prev_sm * (1-this_day_weight))
    antecedent_sw = antecedent_vwc / vwc_bounds.vwc_max    

    # calculate catchment average soil wetness and vwc
    ca_sw = antecedent_sw.mean(dim=['x', 'y'])
    ca_vwc = antecedent_sw.mean(dim=['x', 'y'])

    results = {}
    results['ca_soil_wetness'] = float(ca_sw.values)
    results['ca_vwc'] = float(ca_sw.values)

    if spatial:        
        results['heatmap_sw'] = antecedent_sw
        results['heatmap_vwc'] = antecedent_vwc

    return results   
    
def grab_catchment_topography(catch_id):
    
    topog = rioxarray.open_rasterio(hj_base + '/ancillaries/uk_ihdtm_topography+topoindex_1km.nc')
    topog = topog.assign_coords({'xx':topog.x*1000, 'yy':topog.y*1000})
    topog.coords['y'] = topog.yy
    topog.coords['x'] = topog.xx
    topog = topog.sel(y=slice(None, None, -1)) 
    
    topog = rioxarray.open_rasterio(home_data_dir + '/height_map/topography_bng_1km.nc')    
    topog.rio.write_crs("epsg:27700", inplace=True)
    
    topog.coords['y'] = topog.coords['y'] * 1000
    topog.coords['x'] = topog.coords['x'] * 1000
    #catch_shape = gdf.loc[gdf['CatchID']==str(catch_id)].geometry.values
    #xy_bounds = catch_shape.bounds
    #x_inds = np.intersect1d(np.where(topog.x >= xy_bounds[0,0])[0], np.where(topog.x <= xy_bounds[0,2])[0])
    #y_inds = np.intersect1d(np.where(topog.y >= xy_bounds[0,1])[0], np.where(topog.y <= xy_bounds[0,3])[0])
    
    # trim down using the bounds first
    topog_c = topog.isel(x=x_inds).isel(y=y_inds)
    topog_c = topog_c.rio.clip(catch_shape, topog_c.rio.crs)
    ''' doesn't work '''
    return results
    

def load_landcover():    
    fldr = hj_base+'/soil_moisture_map/ancillaries/land_cover_map/2015/data/'
    fnam = '/LCM2015_GB_1km_percent_cover_aggregate_class.tif'
    lcm = rasterio.open(fldr + fnam)    
    lcm = lcm.read()
    
    lcm1 = rioxarray.open_rasterio(fldr + fnam)
    lcm1.rio.write_crs("epsg:27700", inplace=True)
    
    catch_shape = gdf.loc[gdf['CatchID']==str(catch_id)].geometry.values
    xy_bounds = catch_shape.bounds
    x_inds = np.intersect1d(np.where(lcm1.x >= xy_bounds[0,0])[0], np.where(lcm1.x <= xy_bounds[0,2])[0])
    y_inds = np.intersect1d(np.where(lcm1.y >= xy_bounds[0,1])[0], np.where(lcm1.y <= xy_bounds[0,3])[0])
    
    # trim down using the bounds first
    lcm_c = lcm1.isel(x=x_inds).isel(y=y_inds)
    lcm_c = lcm_c.astype(np.float32).rio.clip(catch_shape, lcm_c.rio.crs)    
    lcm_c['dom_class'] = (('y', 'x'), lcm_c.values.argmax(axis=0).astype(np.float32))
    
    lcm_c.dom_class.plot(cmap='Pastel1'); plt.show()
    class_names = ['Broadleaf_woodland',
                   'Coniferous_woodland',
                   'Arable',
                   'Improved_grassland',
                   'Semi-natural_grassland',
                   'Mountain_heath_bog',
                   'Saltwater',
                   'Freshwater',
                   'Coastal',
                   'Urban']
    dominant_class = lcm.argmax(axis=0)
    l_wooded = lcm[:2,:,:].sum(axis=0)
    l_open = lcm[2:5,:,:].sum(axis=0) + lcm[6:9,:,:].sum(axis=0)
    l_high = lcm[5,:,:]
    l_urban = lcm[9,:,:]
    if False:
        fig, ax = plt.subplots(2,2)
        ax[0,0].imshow(l_wooded)
        ax[0,1].imshow(l_open)
        ax[1,0].imshow(l_high)
        ax[1,1].imshow(l_urban)
        plt.show()
    # access using lcm[:, lcm.shape[1]-1-chess_y, chess_x] # as indexes y-inverted compared to chess
    return np.stack([l_wooded, l_open, l_high, l_urban], axis=0).astype(np.float32)/100.
