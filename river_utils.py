import numpy as np
import pandas as pd
import xarray as xr
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle
import glob
import networkx as nx

from shapely.geometry import box, Point, LineString
from shapely.ops import transform
import shapely.wkt
from sklearn.neighbors import NearestNeighbors
from matplotlib.path import Path as mpl_path

from river_reach_modelling.river_class import River

### functions for building rivers

def integerize_coordinates(geom):
    
   def _int_coords(x, y):
      x = int(x)
      y = int(y)

      return [c for c in (x, y) if c is not None]
   
   return transform(_int_coords, geom)
   
def extract_edges(knn_indices, dists=None, dist_thresh=None):
    edges = []
    for i in range(knn_indices.shape[0]):
        for j in range(1, knn_indices.shape[1]):
            if dist_thresh is None:
                edges.append([knn_indices[i,0], knn_indices[i,j]])
            else:
                if dists[i,j] < dist_thresh:
                    edges.append([knn_indices[i,0], knn_indices[i,j]])

    # remove repeats
    for k in range(len(edges)):
        edges[k] = tuple(np.sort(edges[k]))
    edges = list(set(edges))
    return edges

def calculate_drainage_density(this_river_obj, small_segments):
    '''
    Approximates drainage density by accumulating the length of all the 
    river segments present within an incremental catchment boundary,
    including the main reach and those segments removed by the ccar threshold
    '''
    polygons_to_test = (gpd.GeoDataFrame(this_river_obj.boundaries_increm)
        .reset_index()
        .rename({'index':'id'}, axis=1)
    )
    reaches_in_catchments = gpd.sjoin_nearest(
            small_segments[['xout', 'yout', 'geometry']],
            polygons_to_test,
            distance_col="distance"
    ).query('distance==0').drop('distance', axis=1)
    
    drainage_density = []
    for rid in this_river_obj.network.id:
        channel_length = this_river_obj.network.loc[rid].geometry.length
        for sm_segs in reaches_in_catchments[reaches_in_catchments.id == rid].geometry:
            channel_length += sm_segs.length
        this_drainage_density = channel_length/1000 / this_river_obj.cds_increm.loc[rid].ICAR # km / km^2
        drainage_density.append(this_drainage_density)
    drainage_density = pd.DataFrame({'DRAINAGE_DENSITY':drainage_density},
                                    index=this_river_obj.network.id)
    
    this_river_obj.cds_increm = (this_river_obj.cds_increm.reset_index()
        .rename({'index':'id'}, axis=1)
        .merge(drainage_density, on='id', how='left')
        .set_index('id')
    )
    return this_river_obj
    
def calculate_mean_drainage_path_length(this_segment, increm_boundary):
    '''
    Approximates mean drainage path length by calculating shortest path
    to drainage cell on a simply connected network representing a coarse
    grid that overlays the catchment boundary 
    '''
    npts_in_ctch = 0    
    dpl_res = 100    
    while npts_in_ctch==0:        
        # find points within the incremental catchment boundary
        pts_ext = np.asarray(increm_boundary.exterior.coords)[:, :2]
        xvec = np.arange(pts_ext.min(axis=0)[0], pts_ext.max(axis=0)[0], dpl_res)
        yvec = np.arange(pts_ext.min(axis=0)[1], pts_ext.max(axis=0)[1], dpl_res)
        x_grid, y_grid = np.meshgrid(xvec, yvec)
        x_grid, y_grid = x_grid.flatten(), y_grid.flatten()
        pts_grid = np.vstack((x_grid, y_grid)).T    
        pts_mask = mpl_path(pts_ext).contains_points(pts_grid)
        pts_ctch = pts_grid[pts_mask]
        npts_in_ctch = pts_ctch.shape[0]
        if npts_in_ctch==0 and dpl_res>30:
            dpl_res -= 10
            continue
        
        if npts_in_ctch==0 and dpl_res==30:
            # approximate
            return np.sqrt(increm_boundary.area) / 2.
        
        # add in the drainage cell location at index 0
        dc_loc = this_segment[['xout', 'yout']].values
        dc_in_ptsctch = np.intersect1d(
            np.where(pts_grid[pts_mask][:,0]==dc_loc[0,0])[0],
            np.where(pts_grid[pts_mask][:,1]==dc_loc[0,1])[0]
        )
        if len(dc_in_ptsctch)==0:
            pts_ctch = np.vstack([dc_loc, pts_ctch])
        else:
            pts_ctch = np.vstack([dc_loc, np.delete(pts_ctch, dc_in_ptsctch, axis=0)])
        
        # use a nearest neighbour graph to define shortest path to drainage cell
        nbrs = NearestNeighbors(n_neighbors=min(8, pts_ctch.shape[0]), metric='euclidean').fit(pts_ctch)
        dists, knn_graph = nbrs.kneighbors(pts_ctch, return_distance=True)
        knn_graph = knn_graph.astype(np.int32)
        nodes = list(knn_graph[:,0])
        
        # only use one-hop neighbours, including diagonal hops        
        edges = extract_edges(knn_graph)
        
        G = nx.Graph()
        G.add_nodes_from(nodes) # simply a list
        G.add_edges_from(edges)
                    
        dist_attrs = {}
        for e in edges:
            if e[1] in knn_graph[e[0],:]:
                dist_attrs[e] = dists[e[0], np.where(knn_graph[e[0],:]==e[1])[0][0]]
            else:
                dist_attrs[e] = dists[e[1], np.where(knn_graph[e[1],:]==e[0])[0][0]]
        nx.set_edge_attributes(G, dist_attrs, "distance")
        
        drainage_path_lengths = []
        for src in range(1,len(nodes)):
            try:
                drainage_path_lengths.append(
                    nx.shortest_path_length(
                        G, source=src, target=0, method='dijkstra', weight="distance"
                    )
                )
            except:
                # ignore disconnected arms of catchment...
                continue
        
        return np.mean(drainage_path_lengths)

def approx_increm_cd(cds_increment, big_catch, small_catchs, name):
    cds_increment.loc[0,name] = (big_catch[name].values[0] + sum(
                [(small_catchs[i][name]).values[0]  for i in range(len(small_catchs))]
            ) / len(small_catchs)) / 2
    return cds_increment

def calculate_incremental_catchment_descriptors(draincells, segments, parent_idx,
                                                desc_to_use, increm_boundary):
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
    
    # calculate the distance of all points within the incremental boundary
    # to the drainage cell to represent mean drainage path length 
    dpl_bar = calculate_mean_drainage_path_length(this_segment, increm_boundary)

    # calculate average reach slope using drainage cell heights
    if len(small_catchs)>0:
        upstream_height = sum([s.HGHT.values[0] for s in small_catchs]) / float(len(small_catchs))
        reach_slope = np.rad2deg(np.arcsin(
            (upstream_height - big_catch.HGHT.values[0]) / float(reach_length)
        ))
        # units of reach_length (m) and HGHT (m) should be the same!
    else:
        # approximate slope using mean drainage path slope
        reach_slope = np.rad2deg(np.arctan(big_catch.QDPS.values[0] / 1000.)) # as QDPS reported in m/km
        # is this right or does m/km mean following the hypotenuse for 1km rather than
        # the "horizontal" adjacent side of the triangle?

    if reach_slope<0: reach_slope = 0

    # check values of each descriptor against estimated ranges: (errors from rounding descriptor values)
    # if outside the range and ICAR is small, use the average of the upstream values
    '''    
    standard_descs = [ # that we can simply treat by multiplying by area
        'QB19', # BFIHOST19 [0,1]
        'QUEX', # URBEXT1990 Urban extent, [0,1]
        'QUE2', # URBEXT2000 Urban extent, [0,1]
        #'QFPX', # Mean flood plain extent, [0,1]
        #'QPRW', # PROPWET Proportion of time soils are wet, [0,1]
        #'QSPR', # Standard percentage runoff from HOST, % [0,100]        
        'QDPS', # DPSBAR Mean drainage path slope, m/km ~ 0--400
        #'QFAR'  # FARL Flood attenuation by reservoirs and lakes, [0,1]
    '''
    if len(small_catchs)>0:
        if (cds_increment.QB19.values[0]<0) or (cds_increment.QB19.values[0]>1):
            cds_increment = approx_increm_cd(cds_increment, big_catch, small_catchs, "QB19")
        
        if (cds_increment.QUEX.values[0]<0) or (cds_increment.QUEX.values[0]>1):
            cds_increment = approx_increm_cd(cds_increment, big_catch, small_catchs, "QUEX")
            
        if (cds_increment.QUE2.values[0]<0) or (cds_increment.QUE2.values[0]>1):
            cds_increment = approx_increm_cd(cds_increment, big_catch, small_catchs, "QUE2")
            
        if (cds_increment.QFPX.values[0]<0) or (cds_increment.QFPX.values[0]>1):
            cds_increment = approx_increm_cd(cds_increment, big_catch, small_catchs, "QFPX")
        
        if (cds_increment.QPRW.values[0]<0) or (cds_increment.QPRW.values[0]>1):
            cds_increment = approx_increm_cd(cds_increment, big_catch, small_catchs, "QPRW")
            
        if (cds_increment.QSPR.values[0]<0) or (cds_increment.QSPR.values[0]>100):
            cds_increment = approx_increm_cd(cds_increment, big_catch, small_catchs, "QSPR")
            
        if (cds_increment.QDPS.values[0]<0) or (cds_increment.QDPS.values[0]>750):
            cds_increment = approx_increm_cd(cds_increment, big_catch, small_catchs, "QDPS")
            
        if (cds_increment.QFAR.values[0]<0) or (cds_increment.QFAR.values[0]>1):
            cds_increment = approx_increm_cd(cds_increment, big_catch, small_catchs, "QFAR")            

    return {
        'cumul':big_catch[desc_to_use + ['CCAR']],
        'increm':cds_increment.assign(
            ICAR = ccar_increment,
            QDPL = dpl_bar,
            REACH_LENGTH = reach_length,
            REACH_SLOPE = reach_slope # in degrees
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
        print(f'-- {k}')
        river_branch = segments[segments.dwn_id==k].assign(reach_level = reach_level + 1)
        
        # add new network members and new incremental catchment descs
        river_obj.network = pd.concat([river_obj.network, river_branch])            
        
        river_obj.catchment_boundaries[k] = calculate_incremental_catchment_boundary(boundaries, k)
        
        river_obj.catchment_descriptors[k] = calculate_incremental_catchment_descriptors(
            draincells,
            segments,
            k,
            desc_to_use,
            river_obj.catchment_boundaries[k]['increm']
        )
                
        # keep going down branch
        river_branch = river_branch[river_branch.id != river_branch.dwn_id]
        river_obj = head_upstream(segments, river_obj, river_branch,
                                  draincells, desc_to_use, boundaries,
                                  reach_level + 1)
    return river_obj

def build_all_rivers(end_points, segments, draincells, boundaries,
                     small_segments, desc_to_use, save_rivers_dir):    
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
        
        this_river_obj.catchment_boundaries[ii] = calculate_incremental_catchment_boundary(boundaries, ii)
        
        this_river_obj.catchment_descriptors[ii] = calculate_incremental_catchment_descriptors(
            draincells,
            segments,
            ii,
            desc_to_use,
            this_river_obj.catchment_boundaries[ii]['increm']
        )
            
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
        
        this_river_obj = calculate_drainage_density(this_river_obj, small_segments)
        
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


