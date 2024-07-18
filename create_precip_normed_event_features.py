import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray
import time
import multiprocessing
import argparse

from pathlib import Path
from types import SimpleNamespace

from river_reach_modelling.event_wrangling import load_statistics
from river_reach_modelling.catchment_wrangling import wrangle_descriptors
from river_reach_modelling.river_class import (
    River, load_event_data, sm_data_dir, hj_base, save_rivers_dir
)
from river_reach_modelling.utils import zeropad_strint
from river_reach_modelling.funcs_precip_normed_event_features import (
    run_events_parallel, prepare_constant_catchment_values, process_one_event
)

if __name__=="__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("job_index", help = "index of set of basins")
        parser.add_argument("njobs", help = "number of jobs we are splitting into")
        parser.add_argument("ncpu", help = "number of cpu cores per job")
        parser.add_argument("max_basins", help = "number of river basins")
        parser.add_argument("max_events_per_basin", help = "number of events per basin")
        
        args = parser.parse_args()

        job_index = int(args.job_index)
        njobs = int(args.njobs)
        ncpu = int(args.ncpu)
        max_basins = int(args.max_basins)
        max_events_per_basin = int(args.max_events_per_basin)
        
    except:
        print('Not running as script, using default params')              
        ## define run vars
        job_index = 0
        njobs = 1
        ncpu = 1
        max_basins = 0
        max_events_per_basin = 0

    run_as = 'par'
    np.random.seed(42)
    outpath = hj_base + '/flood_events/event_encoding/'
    
    gauge_river_map = pd.read_csv(hj_base + '/flood_events/station_to_river_map.csv')
    flood_event_df = load_statistics()
    station_descriptors, desc_names = wrangle_descriptors()
    vwc_quantiles = rioxarray.open_rasterio(sm_data_dir + '/vwc_quantiles.nc')

    station_descriptors = station_descriptors.reset_index()[['Station', 'CCAR']]
    station_descriptors.columns = ['Station', 'station_ccar']
    flood_event_df = pd.merge(station_descriptors, flood_event_df, how='right', on='Station')

    # constrain by event length
    lower_bound = 24*4 # one day
    upper_bound = 30*24*4 # one month
    flood_event_df = flood_event_df[
        (flood_event_df.EventDuration_15min >= lower_bound) &
        (flood_event_df.EventDuration_15min <= upper_bound)
    ]

    ######################################
    # import seaborn as sns
    # subdat = flood_event_df[flood_event_df.PeakFlow < flood_event_df.PeakFlow.quantile(0.95)]
    # subdat = subdat[subdat.TotalRain > 0.0005]
    
    # x = subdat.PeakRain
    # y = subdat.PeakFlow
    # sns.jointplot(x=x, y=y, kind="hex", color="#4CB391", gridsize=250)
    # plt.show()
    
    # f, ax = plt.subplots(figsize=(6, 6))
    # sns.scatterplot(x=x, y=y, s=5, color=".15")
    # sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    # sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
    # plt.show()
    
    
    ######################################

    # link events to tidal river segment IDs
    flood_event_df = (flood_event_df.rename({'Station':'nrfa_id'}, axis=1)
        [['nrfa_id', 'station_ccar', 'Event', 'FlowStartDate', 'FlowEndDate']]
        .merge(gauge_river_map, on='nrfa_id')
        .dropna()
    )
    #flood_event_df = flood_event_df.sort_values(['basin_area','id'])
    flood_event_df = flood_event_df.sort_values(['station_ccar','id'])
    
    ######################################
    
    use_nrfa_ids = (flood_event_df[['nrfa_id', 'station_ccar']]
        .drop_duplicates()
        .query('station_ccar < 2000') # < 1500
        .nrfa_id.values
    )
    np.random.shuffle(use_nrfa_ids)    
    
    if max_basins>0:
        use_nrfa_ids = use_nrfa_ids[:max_basins]
        
    chunk_size = int(len(use_nrfa_ids) / njobs)
    n_chunks = len(use_nrfa_ids) // chunk_size
    leftover = len(use_nrfa_ids) - chunk_size * n_chunks
    chunk_sizes = np.repeat(chunk_size, n_chunks)
    if leftover>0:
        chunk_sizes = np.hstack([chunk_sizes, leftover])    
    csum = np.hstack([0, np.cumsum(chunk_sizes)])
    nid_slice = use_nrfa_ids[csum[job_index]:csum[job_index+1]]
    
    for nrfa_id in nid_slice:
        gauge_events = flood_event_df[flood_event_df.nrfa_id==nrfa_id]
        rid = gauge_events.id.values[0]
        station_ccar = gauge_events.station_ccar.values[0]
        all_events = gauge_events.sample(n=gauge_events.shape[0], replace=False)
        if max_events_per_basin>0:
            all_events = all_events.iloc[:max_events_per_basin]

        # create river and calculate in-stream data once and refer back to it for each event    
        river_obj = River()
        river_obj.load(save_rivers_dir, rid)
        gauge_station_data = {}
        average_catchment_data = {}
        gauge_station_data, average_catchment_data = prepare_constant_catchment_values(
            river_obj,
            nrfa_id,
            gauge_station_data,
            average_catchment_data
        )
        gauge_station_data[nrfa_id]['station_ccar'] = station_ccar
        
        if run_as=='serial':
            all_event_data = {}
            start_time = time.perf_counter()
            for ii in range(all_events.shape[0]): 
                event = all_events.iloc[ii]
                feature_timeseries = process_one_event(
                    event,
                    gauge_station_data,
                    average_catchment_data,
                    vwc_quantiles,
                    verbose=True
                )            
                all_event_data[f'{event.nrfa_id}_{event.Event}'] = feature_timeseries
            finish_time = time.perf_counter()
            print(f"Finished event slice in {(finish_time-start_time)/60.} minutes")        
        elif run_as=='par':
            all_event_data = run_events_parallel(
                all_events,
                gauge_station_data,
                average_catchment_data,
                vwc_quantiles,
                ncpu=10
            )
        
        np.savez(outpath + f'{nrfa_id}_{rid}.npz', **all_event_data)    

