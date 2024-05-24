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
import argparse
import time

from catchment_wrangling import *
from river_class import River
from river_utils import *
from utils import zeropad_strint, trim_netcdf, calculate_soil_wetness, merge_two_soil_moisture_days

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_num", help = "chunk of end_points to do, 0-indexed")
    args = parser.parse_args()
    chunk_num = int(args.chunk_num)
    #chunk_num = 218
    time.sleep(chunk_num / 2.)

    ## load individual river reaches
    hjflood_base = "/gws/nopw/j04/hydro_jules/data/uk/flood_events/"
    reaches_dir = hjflood_base + "/reach_data/"
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
        #'QFPD', # Mean flood plain depth in a 100-year event, >0 --- (questionable utility)
        'QPRW', # PROPWET Proportion of time soils are wet, [0,1]
        'QSPR', # Standard percentage runoff from HOST, % [0,100]
        #'QDPB', # DPLBAR Mean drainage path length ---- this needs to be calculated differently,
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
    small_segments = segments[segments.ccar < ccar_threshold] # for calculating drainage density
    segments = segments[segments.ccar >= ccar_threshold]
    boundaries = boundaries[boundaries.ccar >= ccar_threshold]

    save_rivers_dir = reaches_dir + "/river_objects/"
    Path(save_rivers_dir).mkdir(parents=True, exist_ok=True)

    of_n_chunks = 250
    n_total = len(end_points)
    chunk_size = min(25, max(5, n_total // of_n_chunks))    
    chunk_sizes = np.repeat(chunk_size, of_n_chunks)
    leftover = n_total - chunk_size*of_n_chunks
    if leftover>0:
        chunk_sizes[-leftover:] += 1
    elif leftover<0:
        csum = np.cumsum(chunk_sizes)
        pind = np.where(csum > n_total)[0][0]
        diff = csum[pind] - n_total
        chunk_sizes[pind] -= diff
        chunk_sizes[(pind+1):] = 0
       
    csum = np.hstack([0, np.cumsum(chunk_sizes)])
    run_ids = end_points.iloc[csum[chunk_num]:csum[chunk_num+1]]

    build_all_rivers(run_ids, segments, draincells, boundaries,
                     small_segments, desc_to_use, save_rivers_dir)
