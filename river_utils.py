import numpy as np
import pandas as pd
import xarray as xr
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle
import glob
from pathlib import Path
from shapely.geometry import box, Point, LineString

from river_class import River

### functions for building rivers
def calculate_incremental_catchment_descriptors(draincells, segments, parent_idx, desc_to_use):
    # this currently only deals with descriptors that can be easily un-aggregated through area multiplication
    this_segment = segments[segments.id==parent_idx]
    big_catch = draincells[draincells.DCid.isin(this_segment.dc_id)].reset_index(drop=True)
    full_cds = (big_catch.CCAR.values[0] * big_catch[desc_to_use]) # area scaled
            
    small_catchs = [draincells[draincells.DCid==thisid] 
        for thisid in segments[(segments.dwn_id==parent_idx) & (segments.id!=parent_idx)].dc_id]
    upstream_cds = sum(
        [(small_catchs[i].CCAR.values[0] * small_catchs[i][desc_to_use]).reset_index(drop=True) 
            for i in range(len(small_catchs))]
    ) # area scaled
    upstream_ccar = sum([s.CCAR.values[0] for s in small_catchs])
    
    ccar_increment = big_catch.CCAR.values[0] - upstream_ccar
    cds_increment = (full_cds - upstream_cds) / ccar_increment
    
    # length of river reach
    reach_length = this_segment.geometry.iloc[0].length
    
    # calculate average reach slope using drainage cell heights
    if len(small_catchs)>0:
        upstream_height = sum([s.HGHT.values[0] for s in small_catchs]) / float(len(small_catchs))
        reach_slope = np.rad2deg(np.arcsin(
            (upstream_height - big_catch.HGHT.values[0]) / float(reach_length)
        ))
        # check units of reach_length amd HGHT are the same!
    else:
        # approximate slope using mean drainage path slope
        reach_slope = np.rad2deg(np.arctan(big_catch.QDPS.values[0] / 1000.)) # as QDPS reported in m/km
        # is this right or does m/km mean following the hypotenuse for 1km rather than
        # the "horizontal" adjacent side of the triangle?
    return {
        'cumul':big_catch[desc_to_use + ['CCAR']],
        'increm':cds_increment.assign(
            ICAR = ccar_increment,
            REACH_LENGTH = reach_length,
            REACH_SLOPE = reach_slope
        )
    }
    
def calculate_incremental_catchment_boundary(boundaries, parent_idx):
    sub_boundaries = boundaries[boundaries.dwn_id == parent_idx]
    parent_boundary = boundaries[boundaries.id==parent_idx].iloc[0].geometry
    other_boundaries = sub_boundaries.loc[sub_boundaries.id!=parent_idx]
    if other_boundaries.shape[0]>0:
        for i in range(other_boundaries.shape[0]):
            if i==0:
                upstream_boundary = other_boundaries.iloc[0].geometry
            else:
                upstream_boundary = upstream_boundary.union(other_boundaries.iloc[i].geometry)
        incremental_boundary = parent_boundary.difference(upstream_boundary)
    else:
        incremental_boundary = parent_boundary
    return {
        'cumul':parent_boundary,
        'increm':incremental_boundary
    }
    
def head_upstream(segments, river_obj, sub_river, draincells,
                  desc_to_use, boundaries, reach_level):
    for k in sub_river.id:
        river_branch = segments[segments.dwn_id==k].assign(reach_level = reach_level + 1)
        
        # add new network members and new incremental catchment descs
        river_obj.network = pd.concat([river_obj.network, river_branch])            
        river_obj.catchment_descriptors[k] = calculate_incremental_catchment_descriptors(
            draincells, segments, k, desc_to_use
        )
        river_obj.catchment_boundaries[k] = calculate_incremental_catchment_boundary(boundaries, k)
        
        # keep going down branch
        river_branch = river_branch[river_branch.id != river_branch.dwn_id]
        river_obj = head_upstream(segments, river_obj, river_branch,
                                  draincells, desc_to_use, boundaries,
                                  reach_level + 1)
    return river_obj

def build_all_rivers(end_points, segments, draincells, boundaries,
                     desc_to_use, save_rivers_dir):    
    for ii in end_points.id:
        print(ii)
        reach_level = 0
        cur_river = pd.concat([
            segments[segments.id==ii].assign(reach_level = reach_level),
            segments[(segments.dwn_id==ii) & (segments.id!=ii)].assign(reach_level = reach_level + 1)],
            axis = 0
        )
        sub_river = cur_river[cur_river.id != cur_river.dwn_id]

        this_river_obj = River(ii, segments[segments.id==ii].dc_id.values[0])
        this_river_obj.network = cur_river.copy()
        this_river_obj.catchment_descriptors[ii] = calculate_incremental_catchment_descriptors(
            draincells,
            segments,
            ii,
            desc_to_use
        )
        this_river_obj.catchment_boundaries[ii] = calculate_incremental_catchment_boundary(boundaries, ii)
            
        this_river_obj = head_upstream(
            segments,
            this_river_obj,
            sub_river,
            draincells,
            desc_to_use,
            boundaries,
            reach_level+1
        )
        
        this_river_obj.join_dicts()
        
        this_river_obj.save(save_rivers_dir)

def add_nrfa_stations_to_rivers(nrfa_station_metadata, draincells, save_rivers_dir):
    river_ids = glob.glob(save_rivers_dir + "*")
    river_ids = [int(s.replace(save_rivers_dir, "")) for s in river_ids]
    for rid in river_ids:
        print(rid)
        new_river = River()
        new_river.load(save_rivers_dir, rid)
        new_river.find_stations_in_catchments(nrfa_station_metadata, draincells)
        new_river.save_flow_stations(save_rivers_dir)

def merge_incremental_catchment_descriptors(river_obj, stomp_id, absorb_id, desc_to_use):
    # assumes boundaries and reach linestrings have already been merged into absorb_id
    # note: we lose stomp_id from river_obj.cds_increm
    cds_u = river_obj.cds_increm.loc[stomp_id][desc_to_use] * river_obj.cds_increm.loc[stomp_id]['CCAR']
    cds_d = river_obj.cds_increm.loc[absorb_id][desc_to_use] * river_obj.cds_increm.loc[absorb_id]['CCAR']
    new_cds = (cds_u + cds_d) / (river_obj.cds_increm.loc[stomp_id]['CCAR'] + river_obj.cds_increm.loc[absorb_id]['CCAR'])
    
    ccar_increment = river_obj.boundaries_increm.at[absorb_id].area
    reach_length = new_network.at[absorb_id, 'geometry'].length
    reach_slope = np.rad2deg(np.arcsin(
        (river_obj.cds_increm.at[stomp_id, 'REACH_LENGTH'] * 
            np.sin(np.deg2rad(river_obj.cds_increm.at[stomp_id, 'REACH_SLOPE'])) + 
        river_obj.cds_increm.at[absorb_id, 'REACH_LENGTH'] * 
            np.sin(np.deg2rad(river_obj.cds_increm.at[absorb_id, 'REACH_SLOPE']))) /
        new_network.at[absorb_id, 'geometry'].length
    ))
    new_cds = pd.concat([new_cds, 
        pd.Series({'ICAR': ccar_increment,
                   'REACH_LENGTH': reach_length,
                   'REACH_SLOPE': reach_slope})]
    )
    new_cds.name = absorb_id
    new_cds_increm = river_obj.cds_increm.copy()
    new_cds_increm.loc[absorb_id] = new_cds
    new_cds_increm = new_cds_increm.drop(stomp_id)
    return new_cds_increm



def multiline_to_single_line(geometry):
    if isinstance(geometry, LineString):
        return geometry
    coords = list(map(lambda part: list(part.coords), geometry.geoms))
    flat_coords = [Point(*point) for segment in coords for point in segment]
    return LineString(flat_coords)

def stomp_small_catchments(river_obj, area_threshold=1e6):
    # broken!! and also we have a problem at confluences!
    small_areas = (river_obj.calculate_areas()
        .drop(river_obj.river_id)
        .sort_values('area')
        .query(f'area < {area_threshold}')
    )
    small_catchments_present = True if small_areas.shape[0]>0 else False
    while small_catchments_present:
        stomp_id = small_areas.index[0]
        print('stomping %d' % stomp_id)
        river_obj.network = river_obj.network.set_index('id', drop=False)
        # find the catchment that will absorb the small one
        absorb_id = river_obj.network.loc[river_obj.network.id==stomp_id, 'dwn_id'].values
        # find the upstream catchments that drain into the small one
        upstream_ids = river_obj.network.loc[river_obj.network.dwn_id==stomp_id, 'id'].values
        
        if len(absorb_id)==1:
            absorb_id = absorb_id[0] # take as integer
            print('absorbing into %d' % absorb_id)
            
            # merge linestrings for reaches of stomp_id and absorb_id        
            new_reach_line = (river_obj.network.at[stomp_id, 'geometry']
                .union(river_obj.network.at[absorb_id, 'geometry'])
            )
            new_network = river_obj.network.copy()
            new_network.at[absorb_id, 'geometry'] = multiline_to_single_line(new_reach_line)
            
            # update river.network so upstream_ids drain into absorb_id
            for iu in upstream_ids:
                new_network.at[iu, 'dwn_id'] = absorb_id
            new_network = new_network.drop(stomp_id)
            river_obj.network = new_network
            
            # merge incremental catchment boundaries of absorb_id and stomp_id
            new_boundary = (river_obj.boundaries_increm.at[stomp_id]
                .union(river_obj.boundaries_increm.at[absorb_id])
            )
            river_obj.boundaries_increm.at[absorb_id] = new_boundary
            river_obj.boundaries_increm = river_obj.boundaries_increm.drop(stomp_id)
            
            # re-calculate incremental catchment descriptors for new merged catchment
            river_obj.cds_increm = merge_incremental_catchment_descriptors(
                river_obj, stomp_id, absorb_id, desc_to_use
            )
        else:
            # no catchment downstream to absorb small catchment,
            # either leave as is or merge onto an upstream one?
            # have to do something or loop will go on forever
            # maybe filter out tidal catchments from the areas beforehand
            pass
        
        small_areas = (river_obj.calculate_areas()
            .drop(river_obj.river_id)
            .sort_values('area')
            .query(f'area < {area_threshold}')
        )
        small_catchments_present = True if small_areas.shape[0]>0 else False
    return river_obj
