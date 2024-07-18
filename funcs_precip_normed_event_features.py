import numpy as np
import pandas as pd
import geopandas as gpd
import copy
import pkbar
import time
import multiprocessing
from pathlib import Path

from river_reach_modelling.river_class import (
    River, load_event_data, sm_data_dir, hj_base, save_rivers_dir
)
from river_reach_modelling.utils import zeropad_strint

static_features = dict(
    c_feats = [
        'QB19', # BFIHOST19
        'QUEX', # urban extent, merging QUEX and QUE2
        'QDPS', # drainage path slope
        'ICAR', # incremental catchment area
        'QDPL', # mean drainage path length
        'DRAINAGE_DENSITY' # channel length / catchment area
    ],
    mc_feats = [
        'HGS', # superficial hydrogeology 
        'HGB', # bedrock hydrogeology
        'LC' # land cover classes, merging LC1990 and LC2015
    ],
    r_feats = [
        'REACH_LENGTH',
        'REACH_SLOPE',
        'CCAR' # cumulative catchment area, as some proxy of cross-sectional area of channel?
    ]
)


def find_upstream_catchment_ids(river_obj, nrfa_id):
    end_ids = np.setdiff1d(river_obj.network.id, river_obj.network.dwn_id)    
    trails = {}
    for startid in end_ids:
        trails[startid] = [startid]
        tidal = False
        nextid = river_obj.network.set_index('id').at[startid, 'dwn_id']
        while not tidal:
            dwnid = river_obj.network.set_index('id').at[nextid, 'dwn_id']            
            trails[startid].append(nextid)
            if dwnid == nextid:
                tidal = True
            nextid = dwnid
    
    this_station = river_obj.points_in_catchments.loc[river_obj.points_in_catchments.nrfa_id==nrfa_id]
    st_rid = this_station.id.values[0]
    norm_dist_upstream = this_station.norm_dist_upstream.values[0]
    
    upstream_ids = []
    for k in trails.keys():
        if st_rid in trails[k]:
            upstream_ids += trails[k][:trails[k].index(st_rid)]
    upstream_ids = list(set(upstream_ids))
    return (st_rid, norm_dist_upstream, 
            upstream_ids.remove(river_obj.river_id) if st_rid in upstream_ids else upstream_ids)

def find_downstream_path_to_target(river_obj, start_id, target_id):
    trail = []
    at_target = False
    next_id = river_obj.network.set_index('id').at[start_id, 'dwn_id']
    if next_id == target_id:
        trail.append(next_id)
        at_target = True
    while not at_target:
        dwn_id = river_obj.network.set_index('id').at[next_id, 'dwn_id']
        if dwn_id == target_id:
            at_target = True
        trail.append(next_id)
        next_id = dwn_id
    return trail

# do LC aggregation for cumulative catchments
def aggregate_cumul_lc_classes(river_obj):
    # my own hydrological aggregations
    c_agg = {
        '1':[1], # Broadleaf woodland
        '2':[2], # Coniferous woodland # merge both woodlands?
        '3':[3], # Arable
        '4':[4,5,6,7], # Grassland and pasture
        '5':[8,11,19], # Wetlands
        '6':[9,10], # Shrubs
        '7':[12,15,16,17,18], # Bare soil and rock
        '8':[13], # Saltwater
        '9':[14], # Freshwater # merge both water classes?    
        '10':[20,21] # Built-up areas and gardens
    }
    agg_1990 = []
    agg_2015 = []
    for k in c_agg.keys():
        agg_1990.append(river_obj.cds_cumul[[f'LC1990_{zeropad_strint(n)}' for n in c_agg[k]]].sum(axis=1))
        agg_2015.append(river_obj.cds_cumul[[f'LC2015_{zeropad_strint(n)}' for n in c_agg[k]]].sum(axis=1))
        
    agg_1990 = pd.concat(agg_1990, axis=1)
    agg_1990.columns = [f'LC1990_{zeropad_strint(int(n))}' for n in c_agg.keys()]
    agg_2015 = pd.concat(agg_2015, axis=1)
    agg_2015.columns = [f'LC2015_{zeropad_strint(int(n))}' for n in c_agg.keys()]
    
    # remove negatives (they are small) and renormalize to 1
    agg_2015 = agg_2015.clip(lower=0)
    agg_2015 = (agg_2015 / agg_2015.sum(axis=1).values[...,None])
    agg_1990 = agg_1990.clip(lower=0)
    agg_1990 = (agg_1990 / agg_1990.sum(axis=1).values[...,None])
    
    river_obj.cds_cumul = river_obj.cds_cumul.drop(
        [col for col in river_obj.cds_cumul if col.startswith('LC')], axis=1
    )
    
    indexname = river_obj.cds_cumul.index.name
    if indexname is None:
        indexname = 'index'
    
    river_obj.cds_cumul = (river_obj.cds_cumul.reset_index()
        .merge(agg_1990.reset_index(), on=indexname)
        .merge(agg_2015.reset_index(), on=indexname)
        .set_index(indexname)
    )
    river_obj.cds_cumul.index.name = 'id'
    return river_obj

    
def calculate_cumul_lc_for_event(river_obj):        
    mean_year = river_obj.precip_data[river_obj.river_id].index.mean().year
    if mean_year > 2015:
        mean_year = 2015
    elif mean_year < 1990:
        mean_year = 1990
    river_obj.mean_lc_cumul = (
        (1-(mean_year - 1990)/(2015-1990)) * river_obj.cds_cumul.loc[:, river_obj.lc_1990_names].values + 
        (1-(2015 - mean_year)/(2015-1990)) * river_obj.cds_cumul.loc[:, river_obj.lc_2015_names].values
    )
    river_obj.mean_lc_cumul = pd.DataFrame(river_obj.mean_lc_cumul, index=river_obj.cds_cumul.index)        
    river_obj.mean_lc_cumul.columns = river_obj.lc_names
    
    if mean_year > 2000:
        mean_year = 2000
    elif mean_year < 1990:
        mean_year = 1990
    river_obj.mean_urbext_cumul = (
        (1-(mean_year - 1990)/(2000-1990)) * river_obj.cds_cumul.loc[:, ["QUEX"]].values + 
        (1-(2000 - mean_year)/(2000-1990)) * river_obj.cds_cumul.loc[:, ["QUE2"]].values
    )
    river_obj.mean_urbext_cumul = pd.DataFrame({'QUEX':river_obj.mean_urbext_cumul.flatten()},
                                    index=river_obj.cds_cumul.index)
    return river_obj

def replace_cds_with_event_average(river_obj):
    # incremental
    river_obj.cds_increm = (river_obj.cds_increm.drop(['QUEX', 'QUE2'], axis=1)
        .reset_index()
        .merge(river_obj.mean_urbext.reset_index(), on='id')
        .set_index('id')
    )
    river_obj.cds_increm = (river_obj.cds_increm.drop(
            [s for s in river_obj.cds_increm.columns if s.startswith("LC")], axis=1
        )
        .reset_index()
        .merge(river_obj.mean_lc.reset_index(), on='id')
        .set_index('id')
    )
    
    # cumulative
    river_obj.cds_cumul = (river_obj.cds_cumul.drop(['QUEX', 'QUE2'], axis=1)
        .reset_index()
        .merge(river_obj.mean_urbext_cumul.reset_index(), on='id')
        .set_index('id')
    )
    river_obj.cds_cumul = (river_obj.cds_cumul.drop(
            [s for s in river_obj.cds_cumul.columns if s.startswith("LC")], axis=1
        )
        .reset_index()
        .merge(river_obj.mean_lc_cumul.reset_index(), on='id')
        .set_index('id')
    )
    return river_obj

def calc_cumulative_catchment_average(river_obj, var, upstream_ids, station_rid, norm_dist_upstream):
    upstream_area_scaled = (river_obj.cds_increm.loc[upstream_ids, var] * river_obj.cds_increm.loc[upstream_ids, 'ICAR']).sum()
    station_reach_area_scaled = river_obj.cds_increm.loc[station_rid, var] * (1-norm_dist_upstream) * river_obj.cds_increm.loc[station_rid, 'ICAR']
    total_area = river_obj.cds_increm.loc[upstream_ids, 'ICAR'].sum() + (1-norm_dist_upstream) * river_obj.cds_increm.loc[station_rid, 'ICAR']
    return (upstream_area_scaled + station_reach_area_scaled) / total_area

def calc_cumulative_catchment_reach_attrs(river_obj, upstream_ids, station_rid, norm_dist_upstream):
    reach_lengths_area_scaled = []
    reach_slope_area_scaled = []
    for j, uid in enumerate(upstream_ids):
        downstream_path = find_downstream_path_to_target(river_obj, uid, station_rid)                    
        instream_dpl = 0
        instream_dh = 0
        for ds_id in downstream_path:
            instream_dpl += river_obj.cds_increm.loc[ds_id].REACH_LENGTH
            instream_dh += river_obj.cds_increm.loc[ds_id].REACH_LENGTH * np.tan(np.deg2rad(river_obj.cds_increm.loc[ds_id].REACH_SLOPE))
        instream_dpl += (1-norm_dist_upstream)*river_obj.cds_increm.loc[station_rid].REACH_LENGTH
        instream_dh += ((1-norm_dist_upstream)*river_obj.cds_increm.loc[station_rid].REACH_LENGTH * 
            np.tan(np.deg2rad(river_obj.cds_increm.loc[station_rid].REACH_SLOPE))
        )
        instream_slope = np.rad2deg(np.arctan(instream_dh / instream_dpl))
        
        reach_lengths_area_scaled.append(instream_dpl * river_obj.cds_increm.loc[uid, 'ICAR'])
        reach_slope_area_scaled.append(instream_slope * river_obj.cds_increm.loc[uid, 'ICAR'])
    
    reach_lengths_area_scaled.append((1-norm_dist_upstream)*river_obj.cds_increm.loc[station_rid].REACH_LENGTH * 
        (1-norm_dist_upstream) * river_obj.cds_increm.loc[station_rid, 'ICAR'])
    reach_slope_area_scaled.append(river_obj.cds_increm.loc[station_rid].REACH_SLOPE * 
        (1-norm_dist_upstream) * river_obj.cds_increm.loc[station_rid, 'ICAR'])
    total_area = river_obj.cds_increm.loc[upstream_ids, 'ICAR'].sum() + (1-norm_dist_upstream)*river_obj.cds_increm.loc[station_rid,'ICAR']
    av_reachlen = sum(reach_lengths_area_scaled) / total_area
    av_reachslope = sum(reach_slope_area_scaled) / total_area
    return av_reachlen, av_reachslope

def calc_cumulative_catchment_soilwetness(river_obj, upstream_ids, station_rid, norm_dist_upstream, date_range, tt):
    if date_range[tt].day==date_range[0].day:
        sw_u = (river_obj.antecedent_soil_wetness.loc[upstream_ids, 'soil_wetness'] * river_obj.cds_increm.loc[upstream_ids, 'ICAR']).sum()
        sw_s = river_obj.antecedent_soil_wetness.loc[station_rid, 'soil_wetness'] * (1-norm_dist_upstream) * river_obj.cds_increm.loc[station_rid, 'ICAR']
        sw = (sw_u + sw_s) / (river_obj.cds_increm.loc[upstream_ids, 'ICAR'].sum() + (1-norm_dist_upstream) * river_obj.cds_increm.loc[station_rid, 'ICAR'])
    else:
        sw_ar = 0
        for uid in upstream_ids:
            sw_ar += river_obj.soil_wetness_data[uid].loc[
                pd.to_datetime(date_range[tt].date(), format='%Y-%m-%d')
            ].soil_wetness * river_obj.cds_increm.loc[uid, 'ICAR']
        sw_ar += river_obj.soil_wetness_data[station_rid].loc[
                pd.to_datetime(date_range[tt].date(), format='%Y-%m-%d')
            ].soil_wetness * (1-norm_dist_upstream)*river_obj.cds_increm.loc[station_rid, 'ICAR']
        sw = sw_ar / (river_obj.cds_increm.loc[upstream_ids, 'ICAR'].sum() + (1-norm_dist_upstream) * river_obj.cds_increm.loc[station_rid, 'ICAR'])
    return sw

def features_scaled_by_precip(river_obj, names, uprid, date_range, tt, cfeats,
                              nrfa_id, gauge_station_data):    
    station_rid = gauge_station_data[nrfa_id]['station_rid']
    norm_dist_upstream = gauge_station_data[nrfa_id]['norm_dist_upstream']
    upstream_ids = gauge_station_data[nrfa_id]['upstream_ids']
    
    p = ((river_obj.precip_data[uprid].iloc[tt].values[0] * 1e-3) * 
        (river_obj.cds_increm.loc[uprid, 'ICAR'] * 1e3 * 1e3)  / (15 * 60)
    ) # precip in m3/s
    
    if uprid==station_rid:
        p *= (1 - norm_dist_upstream) # assume precip uniform over catchment containing station    
    
    # grab day average of soil wetness
    if date_range[tt].day==date_range[0].day:
        sw = river_obj.antecedent_soil_wetness.at[uprid,'soil_wetness']                    
    else:
        sw = river_obj.soil_wetness_data[uprid].loc[
            pd.to_datetime(date_range[tt].date(), format='%Y-%m-%d')
        ].soil_wetness
    
    # accumulate in-stream flow info (reach length, slope)
    if uprid==station_rid:
        instream_dpl = (1 - norm_dist_upstream) * river_obj.cds_increm.loc[station_rid].REACH_LENGTH
        instream_dh = ((1 - norm_dist_upstream) * river_obj.cds_increm.loc[station_rid].REACH_LENGTH * 
            np.tan(np.deg2rad(river_obj.cds_increm.loc[station_rid].REACH_SLOPE))
        )
        instream_slope = np.rad2deg(np.arctan(instream_dh / instream_dpl)) # degrees
    else:
        instream_dpl = gauge_station_data[nrfa_id]['instream_data'].at[uprid, 'instream_pathlength']
        instream_slope = gauge_station_data[nrfa_id]['instream_data'].at[uprid, 'instream_slope']                
    
    these_cfeats = (river_obj.cds_increm[cfeats]
        .assign(SOIL_WETNESS = sw,
                REACH_LENGTH = instream_dpl,
                REACH_SLOPE = instream_slope)
    )    
    these_cfeats = these_cfeats.loc[uprid, [c for c in names if c in these_cfeats.columns]].values
    
    return p, these_cfeats

def features_zero_precip_case(river_obj, names, date_range, tt,
                              station_cds, nrfa_id,
                              average_catchment_data, gauge_station_data):
    ovlp_names = [c for c in names if (c in river_obj.cds_increm.columns and c in river_obj.cds_cumul.columns)]
    
    station_rid = gauge_station_data[nrfa_id]['station_rid']
    norm_dist_upstream = gauge_station_data[nrfa_id]['norm_dist_upstream']
    upstream_ids = gauge_station_data[nrfa_id]['upstream_ids']  
    
    av_sw = calc_cumulative_catchment_soilwetness(river_obj, upstream_ids, station_rid, norm_dist_upstream, date_range, tt)
    
    # calculate cumulative descriptors for whole basin upstream of gauging station
    partial_feats = (station_cds['these_cds'][ovlp_names] * station_cds['full_area'] - 
        norm_dist_upstream * station_cds['these_cds_i'].ICAR * station_cds['these_cds_i'][ovlp_names]) / station_cds['partial_area']
    partial_feats['QDPL'] = average_catchment_data[nrfa_id].average_drainage_path_length.values[0]
    partial_feats['QDPS'] = average_catchment_data[nrfa_id].average_drainage_path_slope.values[0]
    partial_feats['DRAINAGE_DENSITY'] = average_catchment_data[nrfa_id].average_drainage_density.values[0]
    partial_feats['REACH_SLOPE'] = average_catchment_data[nrfa_id].average_reach_slope.values[0]
    partial_feats['REACH_LENGTH'] = average_catchment_data[nrfa_id].average_reach_length.values[0]
    #partial_feats['CCAR'] = station_cds['partial_area']
    partial_feats['SOIL_WETNESS'] = av_sw
    partial_feats['ICAR'] = 0 # "rained on" area as fraction of total area
    #partial_feats['total_precip'] = 0
    return partial_feats[[n for n in names if n in partial_feats.index]].values.flatten()

def process_one_timepoint(river_obj, date_range, tt, cfeats,
                          station_cds, nrfa_id,
                          average_catchment_data, gauge_station_data):

        names = ['total_precip'] + cfeats + ['SOIL_WETNESS', 'REACH_LENGTH', 'REACH_SLOPE', 'CCAR']
        cfeat_names = cfeats + ['SOIL_WETNESS', 'REACH_LENGTH', 'REACH_SLOPE']
        icar_idx = cfeat_names.index('ICAR')
        num_incr_catchments = len(gauge_station_data[nrfa_id]['upstream_ids'] + [gauge_station_data[nrfa_id]['station_rid']])
        incr_descriptors = np.zeros((num_incr_catchments, len(names)-2), dtype=np.float32)
        incr_rainfall = np.zeros((num_incr_catchments, 1), dtype=np.float32)
        
        ## deal with situation of zero precip by falling back on cumulative catchment descriptors
        #(mm * km^2 in m3) / (15 minutes in seconds)
        total_precip = sum([(river_obj.precip_data[k].iloc[tt].precip * 1e-3) * 
            (river_obj.cds_increm.at[k,'ICAR'] * 1e3 * 1e3) / (15 * 60) 
            for k in gauge_station_data[nrfa_id]['upstream_ids']])
        total_precip += ((river_obj.precip_data[gauge_station_data[nrfa_id]['station_rid']].iloc[tt].precip * 1e-3) * 
            (1 - gauge_station_data[nrfa_id]['norm_dist_upstream']) * 
            (river_obj.cds_increm.at[gauge_station_data[nrfa_id]['station_rid'],'ICAR'] * 1e3 * 1e3) / 
            (15 * 60) 
        ) # precip in m3/s
        if total_precip==0:
            normed_cds = features_zero_precip_case(
                river_obj, names, date_range, tt,
                station_cds, nrfa_id,
                average_catchment_data, gauge_station_data
            )
        else:
            for j, uprid in enumerate(gauge_station_data[nrfa_id]['upstream_ids'] + [gauge_station_data[nrfa_id]['station_rid']]):
                this_precip, these_cfeats = features_scaled_by_precip(
                    river_obj, names, uprid, date_range, tt, cfeats,
                    nrfa_id, gauge_station_data
                )
                
                incr_descriptors[j,:] = these_cfeats
                incr_rainfall[j,:] = this_precip
            
            # calculate rained-on area as fraction of total area
            rained_on_area = (incr_descriptors[:,icar_idx] * (incr_rainfall[:,0]>0)).sum()
            
            # we could also do something to capture the spread/variability
            # in the incr_descriptors values rather than just take the weighted mean?
            # but might be too complicated for now...            
            normed_cds = incr_descriptors * incr_rainfall / incr_rainfall.sum()
            normed_cds = normed_cds.sum(axis=0)
            normed_cds[icar_idx] = rained_on_area

        return total_precip, normed_cds    

def prepare_constant_catchment_values(river_obj, nrfa_id, gauge_station_data, average_catchment_data):
    gauge_station_data[nrfa_id] = {}
    station_rid, norm_dist_upstream, upstream_ids = find_upstream_catchment_ids(river_obj, nrfa_id)
    gauge_station_data[nrfa_id]['station_rid'] = station_rid
    gauge_station_data[nrfa_id]['norm_dist_upstream'] = norm_dist_upstream
    gauge_station_data[nrfa_id]['upstream_ids'] = upstream_ids
    
    pathlengths = []        
    slopes = []
    upstream_ids = []
    for uprid in gauge_station_data[nrfa_id]['upstream_ids']:            
        downstream_path = find_downstream_path_to_target(river_obj, uprid, station_rid)
        instream_dpl = 0
        instream_dh = 0
        for ds_id in downstream_path:
            instream_dpl += river_obj.cds_increm.loc[ds_id].REACH_LENGTH
            instream_dh += river_obj.cds_increm.loc[ds_id].REACH_LENGTH * np.tan(np.deg2rad(river_obj.cds_increm.loc[ds_id].REACH_SLOPE))
        instream_dpl += (1 - norm_dist_upstream) * river_obj.cds_increm.loc[station_rid].REACH_LENGTH
        instream_dh += ((1 - norm_dist_upstream) * river_obj.cds_increm.loc[station_rid].REACH_LENGTH * 
            np.tan(np.deg2rad(river_obj.cds_increm.loc[station_rid].REACH_SLOPE))
        )
        instream_slope = max(0, np.rad2deg(np.arctan(instream_dh / instream_dpl)))
        upstream_ids.append(uprid)            
        slopes.append(instream_slope)
        pathlengths.append(instream_dpl)
    gauge_station_data[nrfa_id]['instream_data'] = pd.DataFrame(
        {'instream_pathlength' : pathlengths,
         'instream_slope' : slopes
        }, index = upstream_ids
    )
            
    # find average QDPL, QDPS, DRAINAGE_DENSITY, REACH_SLOPE and REACH_LENGTH for when there is no precip           
    av_qdpl = calc_cumulative_catchment_average(river_obj, 'QDPL', upstream_ids, station_rid, norm_dist_upstream)
    av_qdps = calc_cumulative_catchment_average(river_obj, 'QDPS', upstream_ids, station_rid, norm_dist_upstream)
    av_draindens = calc_cumulative_catchment_average(river_obj, 'DRAINAGE_DENSITY', upstream_ids, station_rid, norm_dist_upstream)                
    av_reachlen, av_reachslope = calc_cumulative_catchment_reach_attrs(river_obj, upstream_ids, station_rid, norm_dist_upstream)
    average_catchment_data[nrfa_id] = pd.DataFrame({
        'average_drainage_path_length':av_qdpl,
        'average_drainage_path_slope':av_qdps,
        'average_drainage_density':av_draindens,
        'average_reach_length':av_reachlen,
        'average_reach_slope':av_reachslope
    }, index = [0])
    return gauge_station_data, average_catchment_data

def prepare_constant_event_values(event, gauge_station_data, vwc_quantiles):
    date_range = pd.date_range(start=event.FlowStartDate, end=event.FlowEndDate, freq='15min')
    
    river_obj = load_event_data(event.id, date_range, vwc_quantiles=vwc_quantiles)
    river_obj = replace_cds_with_event_average(river_obj)

    cfeats = static_features['c_feats'] + [s for s in river_obj.cds_increm.columns 
        for mcf in static_features['mc_feats'] if s.startswith(mcf)]
    
    these_cds = river_obj.cds_cumul.loc[gauge_station_data[event.nrfa_id]['station_rid']]
    these_cds_i = river_obj.cds_increm.loc[gauge_station_data[event.nrfa_id]['station_rid']]
    full_area = these_cds.CCAR
    #partial_area = these_cds.CCAR - gauge_station_data[event.nrfa_id]['norm_dist_upstream'] * these_cds_i.ICAR
    partial_area = gauge_station_data[event.nrfa_id]['station_ccar'] # area of catchment down to gauge
    station_data = {
        'these_cds':these_cds,
        'these_cds_i':these_cds_i,
        'full_area':full_area,
        'partial_area':partial_area
    }
    return river_obj, date_range, cfeats, station_data

def process_one_event(event, gauge_station_data, average_catchment_data,
                      vwc_quantiles, verbose=False):
    river_obj, date_range, cfeats, station_cds = prepare_constant_event_values(
        event, gauge_station_data, vwc_quantiles)
        
    feature_timeseries = []
    precip_timeseries = []
    for tt in range(len(date_range)):
        if verbose:
            kbar = pkbar.Pbar(name = f' Station_{event.nrfa_id} {event.Event}', target=len(date_range), width=15)
        
        total_precip, normed_cds = process_one_timepoint(
            river_obj, date_range, tt, cfeats,
            station_cds, event.nrfa_id,
            average_catchment_data, gauge_station_data
        )
        
        # add cumulative catchment area as feature
        feature_timeseries.append(np.hstack([normed_cds, station_cds['partial_area']]))
        precip_timeseries.append(total_precip)
        
        if verbose:
            kbar.update(tt)
    
    feature_timeseries = np.stack(feature_timeseries, axis=0)
    precip_timeseries = np.stack(precip_timeseries, axis=0)
    
    # add flow and precip at index 0 and 1
    feature_timeseries = np.concatenate([
        river_obj.flow_data[event.nrfa_id].values.flatten()[...,None],
        precip_timeseries[...,None],
        feature_timeseries
    ], axis=-1)
    
    return feature_timeseries
    
def grab_names(filepath, event=None):
    if Path(filepath).exists():
        names = np.load(filepath)
    elif event is not None:
        date_range = pd.date_range(start=event.FlowStartDate, end=event.FlowEndDate, freq='15min')
        
        river_obj = load_event_data(event.id, date_range, vwc_quantiles=None)
        river_obj = aggregate_cumul_lc_classes(river_obj)
        river_obj = calculate_cumul_lc_for_event(river_obj)
        river_obj = replace_cds_with_event_average(river_obj)

        cfeats = static_features['c_feats'] + [s for s in river_obj.cds_increm.columns 
            for mcf in static_features['mc_feats'] if s.startswith(mcf)]    
        names = ['FLOW', 'PRECIP'] + cfeats + ['SOIL_WETNESS', 'REACH_LENGTH', 'REACH_SLOPE', 'CCAR']
        names = np.array(names)
        np.save(filepath, np.array(names))
    else:
        return 1
    return names
    
def par_worker(ii, event, gauge_station_data, average_catchment_data, vwc_quantiles, return_dict):
    feature_timeseries = process_one_event(
        event, gauge_station_data, average_catchment_data, vwc_quantiles, verbose=False
    )
    return_dict[f'{event.nrfa_id}_{event.Event}'] = feature_timeseries

def run_events_parallel(all_events, gauge_station_data, average_catchment_data, vwc_quantiles, ncpu=1):
    overall_start_time = time.perf_counter()
    # chunk site list into ncpu length chunks    
    chunk_size = ncpu
    event_array = np.array(range(all_events.shape[0]))
    n_chunks = all_events.shape[0] // chunk_size
    leftover = all_events.shape[0] - chunk_size * n_chunks
    chunk_sizes = np.repeat(chunk_size, n_chunks)
    if leftover>0:
        chunk_sizes = np.hstack([chunk_sizes, leftover])
    csum = np.hstack([0, np.cumsum(chunk_sizes)])
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    for chunk_num in range(len(chunk_sizes)):
        start_time = time.perf_counter()
        processes = []
        
        # takes subset of event list to work with, length <= ncpu
        event_slice = event_array[csum[chunk_num]:csum[chunk_num+1]]

        # Creates chunk_sizes[chunk_num] processes then starts them
        for i in range(chunk_sizes[chunk_num]):
            print(f'Running {event_slice[i]} as task {i}')
            p = multiprocessing.Process(target = par_worker, args=(
                    event_slice[i], all_events.iloc[event_slice[i]],
                    gauge_station_data, average_catchment_data, vwc_quantiles,
                    return_dict                    
                )
            )
            p.start()
            processes.append(p)
        
        # Joins all the processes 
        for p in processes:
            p.join()
     
        finish_time = time.perf_counter()
        print(f"Finished event slice in {(finish_time - start_time)/60.} minutes")
    overall_end_time = time.perf_counter()
    print(f"Finished all events in {(overall_end_time - overall_start_time)/60.} minutes")
    return return_dict
