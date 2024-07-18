import numpy as np
import pandas as pd
import xarray as xr
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle
import glob
import rioxarray
import datetime
from pathlib import Path
from shapely.geometry import box, Point, LineString

from river_reach_modelling.catchment_wrangling import wrangle_descriptors
from river_reach_modelling.river_class import River
from river_reach_modelling.river_utils import *
from river_reach_modelling.utils import zeropad_strint, trim_netcdf, calculate_soil_wetness, merge_two_soil_moisture_days


## load individual river reaches
hjflood_base = "/gws/nopw/j04/hydro_jules/data/uk/flood_events/"
reaches_dir = hjflood_base + "/reach_data/"
save_rivers_dir = reaches_dir + "/river_objects/"
segments = gpd.read_parquet(reaches_dir + "/segments_gb.parquet")

## load cumulative catchment descriptors and boundaries
draincells = pd.read_parquet(reaches_dir + "/CDs_draincells.parquet")
boundaries = gpd.read_file(reaches_dir + "/segments_gb_boundaries.shp")

## load the NRFA stations
catch_desc, desc_names = wrangle_descriptors()
nest_catch, unnest_catch = wrangle_catchment_nesting()
nrfa_station_metadata = pd.read_csv(hjflood_base + "/nrfa-station-metadata-2024-04-15.csv")
nrfa_station_metadata = nrfa_station_metadata[[
    'id', 'name', 'catchment-area', 'easting', 'northing',
    'latitude', 'longitude', 'river', 'location'
]]
nrfa_points = []
for i in range(nrfa_station_metadata.shape[0]):
    nrfa_points.append(
        Point(np.array([nrfa_station_metadata.iloc[i].easting,
                        nrfa_station_metadata.iloc[i].northing]))
    )
nrfa_station_metadata['geometry'] = nrfa_points
nrfa_station_metadata = gpd.GeoDataFrame(nrfa_station_metadata)
nrfa_station_metadata = nrfa_station_metadata.rename({'id':'nrfa_id'}, axis=1)

# find "endpoint" river reaches that drain into the sea
# and step backwards to build up river segments
end_points = segments[segments.tidal==True]

# along the way, calculate incremental catchment descriptors and boundaries
nonoverlap_desc = np.setdiff1d(draincells.columns, desc_names.PROPERTY_ITEM)
overlap_desc = np.intersect1d(draincells.columns, desc_names.PROPERTY_ITEM)

standard_descs = [ # that we can simply treat by multiplying by area
    'QB19', # BFIHOST19 [0,1]
    #'QALT', # Altitude (mean) ---- instead use mean slope
    'QUEX', # URBEXT1990 Urban extent, [0,1]
    'QUE2', # URBEXT2000 Urban extent, [0,1]
    'QFPX', # Mean flood plain extent, [0,1]
    'QFPD', # Mean flood plain depth in a 100-year event, >0 (questionable utility)
    'QPRW', # PROPWET Proportion of time soils are wet, [0,1]
    'QSPR', # Standard percentage runoff from HOST, % [0,100]
    #'QDPL', # DPLBAR Mean drainage path length ---- this needs to be calculated differently,
    # can calculate easily using distance between x,y of each point and drainage cell 
    'QDPS', # DPSBAR Mean drainage path slope, m/km ~ 0--400
    'QFAR'  # FARL Flood attenuation by reservoirs and lakes, [0,1]
]
print(desc_names[desc_names.PROPERTY_ITEM.isin(np.setdiff1d(overlap_desc, standard_descs))])
print(desc_names[desc_names.PROPERTY_ITEM.isin(np.intersect1d(overlap_desc, standard_descs))])

'''
calculated desc:
    ICAR: [0, ~20, but small number up to 250]
    REACH_LENGTH: [50, ~10000, but small number up to 40000]
    REACH_SLOPE: [0, ~35] (check for negatives due to rounding)
    CCAR: [3, ~2000, but some up to 10000]
    QDPB: to calculate
    DRAINAGE_DENSITY: to calculate
'''

hgb_n = [col for col in draincells if col.startswith('HGB')] # hydrogeology bedrock
hgs_n = [col for col in draincells if col.startswith('HGS')] # hydrogeology superficial
lc_n = [col for col in draincells if col.startswith('LC')] # landcover (aggregate these!)
desc_to_use = standard_descs + hgb_n + hgs_n + lc_n

# threshold ccar to simplify catchments
ccar_threshold = 3
end_points = end_points[end_points.ccar >= ccar_threshold]
small_segments = segments[segments.ccar < ccar_threshold]
segments = segments[segments.ccar >= ccar_threshold]
boundaries = boundaries[boundaries.ccar >= ccar_threshold]

Path(save_rivers_dir).mkdir(parents=True, exist_ok=True)

NEW_RIVERS = False
if NEW_RIVERS:
    build_all_rivers(end_points, segments, draincells, boundaries,
                     small_segments, desc_to_use, save_rivers_dir)


ADD_NRFA_STATIONS = False
if ADD_NRFA_STATIONS:
    add_nrfa_stations_to_rivers(nrfa_station_metadata, draincells, save_rivers_dir)
    
# river_id = 617 looks like a good simple test catchment with a single nrfa station
# river_id = 15340 a much more complex system with multiple stations

ADD_WEIGHT_MATRICES = False
if ADD_WEIGHT_MATRICES:
    # loop through rivers loading GEAR and SM grids for each catchment
    # save the calculated weight matrices
    pass

MATCH_GAUGES_TO_REACHES = False
if MATCH_GAUGES_TO_REACHES:
    ## create map from nrfa station to river id
    gauge_river_map = gpd.sjoin_nearest(
        nrfa_station_metadata,
        (boundaries[boundaries.id.isin(segments.query("tidal==True").id)]
            [['id', 'dc_id', 'xout', 'yout', 'east', 'north', 'geometry']]
        ),
        distance_col="distance"
    ).query('distance==0').drop('distance', axis=1)
    gauge_river_map = gauge_river_map[['nrfa_id', 'id',  'dc_id', 'name', 'river', 'location', 'xout', 'yout']]    

    # appending total basin size to the gauge mapping
    basin_ccar = []
    for init_id in gauge_river_map.id:
        paper_trail = [init_id]
        tidal = False
        nextid = segments.at[init_id, 'dwn_id']
        while not tidal:
            dwnid = segments.at[nextid, 'dwn_id']
            if dwnid == nextid:
                tidal = True
            nextid = dwnid
            if not tidal:
                paper_trail.append(nextid)
        basin_ccar.append(segments.at[paper_trail[-1], 'ccar'])
    
    gauge_river_map['basin_area'] = basin_ccar
    gauge_river_map.reset_index(drop=True).to_csv(hjflood_base + '/station_to_river_map.csv', index=False)



'''
##############################
######### TESTING ############
##############################
'''
if False:
    ## loading all rivers to find normalisations of all static quantities
    import os
    all_river_ids = next(os.walk(save_rivers_dir))[1]
    all_river_ids = [int(rid) for rid in all_river_ids]

    all_cds_increm = pd.DataFrame()
    all_cds_cumul = pd.DataFrame()
    for i,rid in enumerate(all_river_ids):
        print(i)
        river_obj = River()
        river_obj.load(save_rivers_dir, rid)
        all_cds_increm = pd.concat([all_cds_increm, river_obj.cds_increm], axis=0)
        all_cds_cumul = pd.concat([all_cds_cumul, river_obj.cds_cumul], axis=0)

    all_cds_increm.to_parquet(reaches_dir + '/all_increm_descriptors.parquet')
    all_cds_cumul.to_parquet(reaches_dir + '/all_cumul_descriptors.parquet')

    nan_inds = np.where(np.isnan(all_cds_increm.values))
    # we also have nans in cumulative descriptors...

    to_norm_cds = [
        'QDPS',
        'ICAR',
        'QDPL',
        'DRAINAGE_DENSITY',
        'REACH_LENGTH',
        'REACH_SLOPE'
    ]
    cd_dists = all_cds_increm.quantile(q=np.linspace(0,1,201))[to_norm_cds]
    cd_dists = pd.concat([cd_dists, all_cds_cumul[['CCAR']].quantile(q=np.linspace(0,1,201))], axis=1)
    cd_dists.index.name = "quantile"
    cd_dists.reset_index().to_csv(reaches_dir + '/catchment_descriptor_quantiles.csv', index=False)
    
    def moving_average(a, n=2):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    cd_dist_path = reaches_dir + '/descriptor_distributions/'
    Path(cd_dist_path).mkdir(exist_ok=True, parents=True)
    
    for name in to_norm_cds:
        # calculating pdf and cdf
        print(name)
        count, bins_count = np.histogram(all_cds_increm[name].dropna(), bins=201) 
        bin_centres = moving_average(bins_count, n=2)
        
        pdf = count / sum(count) 
        #plt.plot(bin_centres, pdf); plt.show()

        cdf = np.cumsum(pdf)
        #plt.plot(bin_centres, cdf); plt.show()
        
        (pd.DataFrame({'bin_centre':bin_centres, 'pdf':pdf, 'cdf':cdf})
            .to_csv(cd_dist_path + f'/{name}_dist.csv', index=False)
        )

    name = 'CCAR'
    count, bins_count = np.histogram(all_cds_cumul[name].dropna(), bins=201) 
    bin_centres = moving_average(bins_count, n=2)
    
    pdf = count / sum(count) 
    #plt.plot(bin_centres, pdf); plt.show()

    cdf = np.cumsum(pdf)
    #plt.plot(bin_centres, cdf); plt.show()
    
    (pd.DataFrame({'bin_centre':bin_centres, 'pdf':pdf, 'cdf':cdf})
        .to_csv(cd_dist_path + f'/{name}_dist.csv', index=False)
    )


if False:
    import os
    all_river_ids = next(os.walk(save_rivers_dir))[1]
    all_river_ids = [int(rid) for rid in all_river_ids]
    
    # find rivers wit whacky descriptors
    for i,rid in enumerate(all_river_ids):
        print(i)
        river_obj = River()
        river_obj.load(save_rivers_dir, rid)
        if river_obj.cds_increm.max().QB19>1 or river_obj.cds_increm.min().QB19<0:
            break
    
    badid = 107749 # 124122
    paper_trail = [badid]
    tidal = False
    nextid = segments.at[badid, 'dwn_id']
    while not tidal:
        dwnid = segments.at[nextid, 'dwn_id']
        if dwnid == nextid:
            tidal = True
        if not tidal:
            paper_trail.append(nextid)
        nextid = dwnid

    bad_rid = dwnid

if False:
    # plot line segments with flow as line thickness
    cm = plt.get_cmap('viridis')
    num_colours = river_obj.network.reach_level.max() + 1
    [plt.plot(*river_obj.network.iloc[i].geometry.xy,
              c=cm(river_obj.network.iloc[i].reach_level/num_colours),
              linewidth=2*np.log(1 + river_obj
                .flow_est[river_obj.network.id.iloc[i]]
                .at[date_range[5],'flow']
            ))
        for i in range(river_obj.network.shape[0])]

    plt.plot(
        river_obj.points_in_catchments[
            river_obj.points_in_catchments.nrfa_id.isin(list(river_obj.flow_data.keys()))
        ].easting,
        river_obj.points_in_catchments[
            river_obj.points_in_catchments.nrfa_id.isin(list(river_obj.flow_data.keys()))
        ].northing,
        'o'
    )
    # add parent cumulative catchment outline
    plt.plot(*river_obj.boundaries_cumul.loc[river_obj.river_id].exterior.xy,
             '--', c='k', linewidth=1.5, alpha=0.8)
    plt.show()



if False:
    # find a good test catchment
    sub_list_catch = end_points.sort_values('ccar')[['id', 'ccar']].iloc[2500:2700]
    for rid in sub_list_catch.id:
        river_obj = River()
        river_obj.load(save_rivers_dir, rid)
        if river_obj.points_in_catchments.shape[0]>0:
            print(rid)
            print(river_obj.points_in_catchments)

if False:
    ## stomp small catchments and join them to bigger downstream ones   
    area_threshold = 5e6 # 1km x 1km
    rid = 15340
    river_obj = River()
    river_obj.load(save_rivers_dir, rid)
    # broken, and also we have a problem at confluences!
    river_obj = stomp_small_catchments(river_obj, area_threshold=area_threshold)

    mask = [type(river_obj.network.geometry.iloc[i])==type(MultiLineString()) 
        for i in range(river_obj.network.shape[0])]

    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    [ax[0].plot(*river_obj.network[mask].explode().iloc[i].geometry.xy)
        for i in range(river_obj.network[mask].explode().shape[0])]
    [ax[1].plot(*multiline_to_single_line(river_obj.network[mask].loc[idx].geometry).xy)
        for idx in river_obj.network[mask].id]
    plt.show()

if False:
    # load a map for visualisation
    import xarray as xr
    hj_base = "/gws/nopw/j04/hydro_jules/data/uk/"
    xygrid = xr.load_dataset(hj_base + "/ancillaries/chess_landfrac.nc",
                             decode_coords="all")
    elev_map = xr.load_dataset(hj_base + "/ancillaries/uk_ihdtm_topography+topoindex_1km.nc",
                               decode_coords="all")
    xygrid['elev'] = (("y", "x"), elev_map.elev.data)
    xygrid.elev.values[np.isnan(xygrid.elev.values)] = 0
            
    # visualise river reaches on top of grid
    river_obj.plot_river(grid=xygrid.elev, stations=True)

if False:
    ## testing grid averaging function
    import rioxarray
    import time
    gear_dir = '/gws/nopw/j04/ceh_generic/netzero/downscaling/ceh-gear/'
    
    rid = 59387 # 15340
    river_obj = River()
    river_obj.load(save_rivers_dir, rid)

    date_range = pd.date_range(start="1995/04/08 11:45:00", end="1995/04/10 13:15:00", freq='15min')

    ## load flow observations
    river_obj.load_flow_data(date_range, flow_dir) # creates river_obj.flow_data

    # generate initial flow condition across entire river network
    river_obj.generate_teacher_forcing_flows() # creates river_obj.flow_est

    ## load precip
    river_obj.load_precip_data(date_range, gear_dir) # populates river_obj.precip_data
    # what about antecedent precip?

    ## load SM
    if vwc_quantiles is None:
        vwc_quantiles = rioxarray.open_rasterio(sm_data_dir + '/vwc_quantiles.nc')
        
    river_obj.load_soil_wetness(date_range, vwc_quantiles, sm_data_dir)  # populates river_obj.soil_wetness_data

    ## approximate land cover from 1990 and 2015 values
    river_obj.calculate_lc_for_event()

    # testing plotting catchment average/grid things
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from matplotlib.collections import PatchCollection


    # Plots a Polygon to pyplot `ax`
    def plot_polygon(ax, poly, **kwargs):
        path = Path.make_compound_path(
            Path(np.asarray(poly.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

        patch = PathPatch(path, **kwargs)
        collection = PatchCollection([patch], **kwargs)
        
        ax.add_collection(collection, autolim=True)
        ax.autoscale_view()
        return collection

    fig, ax = plt.subplots()
    cm = plt.get_cmap('viridis')
    num_colours = river_obj.network.reach_level.max() + 1
    
    [plt.plot(*river_obj.network.iloc[i].geometry.xy,
              c=cm(river_obj.network.iloc[i].reach_level/num_colours),
              linewidth=1.2) 
        for i in range(river_obj.network.shape[0])]

    uniq_days = date_range[(date_range.hour == 0) & (date_range.minute == 0)]
    [plot_polygon(ax, river_obj.boundaries_increm.loc[i],
                  facecolor=plt.cm.Blues(
                    river_obj.soil_wetness_data[i].at[uniq_days[0], 'soil_wetness'])
                 ) for i in river_obj.network.id]    
    
    # [plot_polygon(ax, river_obj.boundaries_increm.loc[i],
                  # facecolor=plt.cm.Blues(
                    # 1.5*river_obj.precip_data[i].at[date_range[-3], 'precip'])
                 # ) for i in river_obj.network.id]                 

    plt.plot(*river_obj.boundaries_cumul.loc[river_obj.river_id].exterior.xy,
             '--', c='k', linewidth=1.5, alpha=0.8)
    plt.show()


    # now load up the gridded version and plot the same thing
    if str(date_range[0])[14:16]=='00':
        start_date_hr = pd.to_datetime(str(date_range[0])[:13] + ':00:00', format="%Y-%m-%d %H:%M:%S")
    else:
        pushed_time = date_range[0] + pd.to_timedelta("1H")
        start_date_hr = pd.to_datetime(str(pushed_time)[:13] + ':00:00', format="%Y-%m-%d %H:%M:%S")
    if str(date_range[-1])[14:16]=='00':
        end_date_hr = pd.to_datetime(str(date_range[-1])[:13] + ':00:00', format="%Y-%m-%d %H:%M:%S")
    else:
        pushed_time = date_range[-1] + pd.to_timedelta("1H")
        end_date_hr = pd.to_datetime(str(pushed_time)[:13] + ':00:00', format="%Y-%m-%d %H:%M:%S")
        
    dates_hr = pd.date_range(start_date_hr, end_date_hr, freq="H")
    time_index = []    
    res = 1000 # 1km
    yy = date_range[-3].year
    mm = date_range[-3].month            
    cds = rioxarray.open_rasterio(gear_dir + f'/{yy}/CEH-GEAR-1hr-v2_{yy}{zeropad_strint(mm)}.nc',
                                  decode_coords="all")
    rfd = cds[2].drop_vars(['lat', 'lon', 'min_dist', 'stat_disag'])
    rfd['x'] = rfd.x + res/2
    rfd['y'] = rfd.y + res/2
    del(cds)
    
    rfd['time'] = rfd.time.astype("datetime64[ns]")
    rfd = rfd.sel(time = rfd.time.isin(dates_hr))
    these_times = pd.to_datetime(rfd['time'].values)
    if len(time_index)==0:
        time_index = these_times
    else:
        time_index = np.hstack([time_index, these_times])
    
    if len(river_obj.precip_parent_bounds.keys())==0:                    
        big_xy_bounds = river_obj.boundaries_cumul.loc[river_obj.river_id].bounds # (minx, miny, maxx, maxy)
        bx_inds = np.intersect1d(np.where((rfd.x + res/2) >= big_xy_bounds[0]),
                                 np.where((rfd.x - res/2) <= big_xy_bounds[2]))
        by_inds = np.intersect1d(np.where((rfd.y + res/2) >= big_xy_bounds[1]),
                                 np.where((rfd.y - res/2) <= big_xy_bounds[3]))
        river_obj.precip_parent_bounds['x_inds'] = bx_inds
        river_obj.precip_parent_bounds['y_inds'] = by_inds
    
    rfd_trim = rfd.isel(x=river_obj.precip_parent_bounds['x_inds'],
                        y=river_obj.precip_parent_bounds['y_inds'])
                        
                        
    rfd_trim.rainfall_amount[-1,:,:].plot(alpha=0.6, cmap='Blues')
    cm = plt.get_cmap('viridis')
    num_colours = river_obj.network.reach_level.max() + 1
    [plt.plot(*river_obj.network.iloc[i].geometry.xy,
              c=cm(river_obj.network.iloc[i].reach_level/num_colours),
              linewidth=1.2) 
        for i in range(river_obj.network.shape[0])]
    [plt.plot(*river_obj.boundaries_increm.loc[i].exterior.xy, ':',
              c=cm(river_obj.network.iloc[idx].reach_level/num_colours),
              linewidth=0.7, alpha=0.9) 
        for idx, i in enumerate(river_obj.network.id)]
    # add parent cumulative catchment outline
    plt.plot(*river_obj.boundaries_cumul.loc[river_obj.river_id].exterior.xy,
             '--', c='k', linewidth=1.5, alpha=0.8)
    plt.show()

    
    # ## Method 3: parallel computation across sub catchments?
    # import multiprocessing
    # def run_chains_parallel(j_objs, ncpu=1):
        # # chunk site list into ncpu length chunks
        # chunk_size = ncpu
        # n_chunks = len(j_objs) // chunk_size
        # leftover = len(j_objs) - chunk_size * n_chunks
        # chunk_sizes = np.repeat(chunk_size, n_chunks)
        # if leftover>0:
            # chunk_sizes = np.hstack([chunk_sizes, leftover])
        
        # csum = np.hstack([0, np.cumsum(chunk_sizes)])
        # chain_nums = np.arange(len(j_objs))
        
        # for chunk_num in range(len(chunk_sizes)):
            # start_time = time.perf_counter()
            # processes = []
            
            # # takes subset of chains to work with, length <= ncpu
            # chain_slice = chain_nums[csum[chunk_num]:csum[chunk_num+1]]

            # # Creates chunk_sizes[chunk_num] processes then starts them
            # for i in range(chunk_sizes[chunk_num]):
                # print(f'Running chain {chain_slice[i]} as task {i}')
                # p = multiprocessing.Process(target = j_objs[chain_slice[i]].run_all_sites)
                # p.start()
                # processes.append(p)
            
            # # Joins all the processes 
            # for p in processes:
                # p.join()
         
        # finish_time = time.perf_counter()
        # print(f"Finished chain slice in {(finish_time-start_time)/60.} minutes")


