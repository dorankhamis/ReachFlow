import pickle
import rioxarray
import glob

import numpy as np
import pandas as pd
import xarray as xr
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd

from pathlib import Path
from shapely.geometry import box, Point
from sklearn.neighbors import NearestNeighbors

from utils import (zeropad_strint, trim_netcdf, calculate_soil_wetness,
                   merge_two_soil_moisture_days,  normalise_df_simplex_subset)

hj_base = "/gws/nopw/j04/hydro_jules/data/uk/"
gear_dir = "/gws/nopw/j04/ceh_generic/netzero/downscaling/ceh-gear/"
sm_data_dir = hj_base + "/soil_moisture_map/output/netcdf/SM/"
flow_dir = hj_base + "/flood_events/15min_flow_data/"
reaches_dir = hj_base + "/flood_events/reach_data/"
save_rivers_dir = reaches_dir + "/river_objects/"

class River:
    def __init__(self, river_id=None, parent_drainage_id=None):
        self.river_id = river_id
        self.final_drainage_cell_id = parent_drainage_id
        self.network = pd.DataFrame()
        self.catchment_descriptors = {} # temporary
        self.catchment_boundaries = {} # temporary
        self.cds_cumul = pd.DataFrame() # cumulative catchment descriptors
        self.cds_increm = pd.DataFrame() # incremental catchment descriptors
        self.boundaries_cumul = pd.DataFrame() # cumulative catchment boundaries
        self.boundaries_increm = pd.DataFrame() # incremental catchment boundaries
        self.points_in_catchments = None # flow gauge locations 
        self.precip_parent_bounds = {} # for trimming precip netcdf to watershed
        self.soilmoisture_parent_bounds = {} # for trimming soil wetness to watershed
        self.weight_matrices = {} # for catchment-averaging netcdfs
        self.flow_data = {} # 15 minute flow at NRFA stations
        self.precip_data = {} # hourly catchment average rainfall        
        
    def save(self, parent_path):
        # make directory to house all river data
        this_dir = parent_path + f'/{self.river_id}/'
        Path(this_dir).mkdir(exist_ok=True, parents=True)
        
        # save river network and id/dc_id
        self.network.to_parquet(this_dir + 'network.parquet', index=False)
        pd.to_pickle({'river_id':self.river_id,
                      'final_drainage_cell_id':self.final_drainage_cell_id},
                     this_dir + 'river_id_numbers.pickle')
        
        # save catchment descriptors
        self.cds_increm.to_parquet(this_dir + 'descriptors_increm.parquet')
        self.cds_cumul.to_parquet(this_dir + 'descriptors_cumul.parquet')
        
        # save catchment boundaries
        gpd.GeoDataFrame(self.boundaries_cumul).reset_index().to_file(this_dir + 'boundaries_cumul.shp')
        gpd.GeoDataFrame(self.boundaries_increm).reset_index().to_file(this_dir + 'boundaries_increm.shp')
        
        # save weighting matrices for averaging gridded data
        if len(self.weight_matrices.keys())>0:
            self.save_weight_matrices(parent_path)
        
        # save flow gauging station metadata in subcatchments
        if not self.points_in_catchments is None:
            self.save_flow_stations(parent_path)
        
    def load(self, parent_path, river_id, aggregate_landcover=True):
        this_dir = parent_path + f'/{river_id}/'
        
        self.network = gpd.read_parquet(this_dir + 'network.parquet')
        
        river_ids = pd.read_pickle(this_dir + 'river_id_numbers.pickle')
        self.river_id = river_ids['river_id']
        self.final_drainage_cell_id = river_ids['final_drainage_cell_id']
        
        self.cds_increm = pd.read_parquet(this_dir + 'descriptors_increm.parquet')
        self.cds_cumul = pd.read_parquet(this_dir + 'descriptors_cumul.parquet')
        
        # rename CCAR to ICAR in cds_increm
        if "CCAR" in self.cds_increm.columns:
            self.cds_increm = self.cds_increm.rename({"CCAR":"ICAR"}, axis=1)
        
        # fix negatives in hydrogeography and land cover simplexes
        self.cds_increm = normalise_df_simplex_subset(self.cds_increm, "HGB")
        self.cds_increm = normalise_df_simplex_subset(self.cds_increm, "HGS")
        self.cds_increm = normalise_df_simplex_subset(self.cds_increm, "LC1990")
        self.cds_increm = normalise_df_simplex_subset(self.cds_increm, "LC2015")
                
        self.boundaries_cumul = gpd.read_file(this_dir + 'boundaries_cumul.shp').set_index('index')['geometry']
        self.boundaries_increm = gpd.read_file(this_dir + 'boundaries_increm.shp').set_index('index')['geometry']
        
        try:
            self.load_flow_stations(parent_path)
        except:
            pass
        
        try:
            self.load_weight_matrices(river_obj_dir)
        except:
            pass
            
        if aggregate_landcover:
            self.aggregate_landcover_classes()    
        
        # set names for multi-class descriptors
        self.hgb_names = [col for col in self.cds_increm if col.startswith('HGB')]
        self.hgs_names = [col for col in self.cds_increm if col.startswith('HGS')]
        self.lc_1990_names = [col for col in self.cds_increm if col.startswith('LC1990')]
        self.lc_2015_names = [col for col in self.cds_increm if col.startswith('LC2015')]
    
    def save_flow_stations(self, river_obj_dir):
        self.points_in_catchments.to_parquet(river_obj_dir + f'/{self.river_id}/flow_stations.parquet')
        
    def load_flow_stations(self, river_obj_dir):
        self.points_in_catchments = pd.read_parquet(river_obj_dir + f'/{self.river_id}/flow_stations.parquet')
        
    def save_weight_matrices(self, river_obj_dir):
        pd.to_pickle(self.weight_matrices, river_obj_dir + f'/{self.river_id}/weight_matrices.pickle')
        
    def load_weight_matrices(self, river_obj_dir):
        self.weight_matrices = pd.read_pickle(river_obj_dir + f'/{self.river_id}/weight_matrices.pickle')    
    
    def join_dicts(self):
        # join the catchment descriptor/boundary dicts into single dfs for cumulative and incremental
        self.cds_cumul = (pd.concat([
            self.catchment_descriptors[k].pop('cumul') 
                for k in self.catchment_descriptors.keys()], axis=0)
            .set_index(np.array(list(self.catchment_descriptors.keys())))
        )        
        self.cds_increm = (pd.concat([
            self.catchment_descriptors[k].pop('increm') 
                for k in self.catchment_descriptors.keys()], axis=0)
            .set_index(np.array(list(self.catchment_descriptors.keys())))
        )
        del(self.catchment_descriptors)
        
        self.boundaries_cumul = pd.DataFrame({'geometry':[
                self.catchment_boundaries[k].pop('cumul') 
                for k in self.catchment_boundaries.keys()]
            }, index=list(self.catchment_boundaries.keys())
        )['geometry'] # Series
        self.boundaries_increm = pd.DataFrame({'geometry':[
                self.catchment_boundaries[k].pop('increm') 
                for k in self.catchment_boundaries.keys()]
            }, index=list(self.catchment_boundaries.keys())
        )['geometry'] # Series
        del(self.catchment_boundaries)
        
    def catchment_average_netcdf(self, catchment_id, nc_dat, var_name,
                                 res=1000, plot=False, nc_type='gear'):
        got_wm = False
        if catchment_id in self.weight_matrices.keys():
            if nc_type in self.weight_matrices[catchment_id].keys():
                got_wm = True
            else:
                self.weight_matrices[catchment_id][nc_type] = {}
        else:
            self.weight_matrices[catchment_id] = {}
            self.weight_matrices[catchment_id][nc_type] = {}

        nc_dat['trimmed_'+var_name] = nc_dat[var_name].copy()
        nc_dat['trimmed_'+var_name].values[:,:] = np.nan
        
        the_dims = list(nc_dat[var_name].dims)
        y_pos = the_dims.index('y')
        x_pos = the_dims.index('x')
        t_pos = None
        if len(the_dims)>2:
            if 'time' in the_dims:
                t_pos = the_dims.index('time')
            elif 't' in the_dims:
                t_pos = the_dims.index('t')
    
        if got_wm:
            # re-use existing weight matrix and XY coords
            weight_matrix = self.weight_matrices[catchment_id][nc_type]['weight_matrix']
            XY = self.weight_matrices[catchment_id][nc_type]['XY']
            
            # trim the values to those within the catchment bounds rectangle            
            nc_dat = trim_netcdf(nc_dat, XY, var_name, the_dims,
                                 t_pos=t_pos, y_pos=y_pos, x_pos=x_pos)            
        else:
            # extract xy points and create within-boundary mask
            catchment_geom = self.boundaries_increm.loc[catchment_id]
            xy_bounds = catchment_geom.bounds # (minx, miny, maxx, maxy)

            if nc_dat.x.values[0]%res==0 and nc_dat.y.values[0]%res==0:
                # coordinates at bottom left of pixel
                # move them to the pixel centre
                nc_dat['x'] = nc_dat.x + res/2
                nc_dat['y'] = nc_dat.y + res/2

            # find easting/northing grid indices for gridded dataset       
            x_inds = np.intersect1d(np.where((nc_dat.x + res/2) >= xy_bounds[0]),
                                    np.where((nc_dat.x - res/2) <= xy_bounds[2]))
            y_inds = np.intersect1d(np.where((nc_dat.y + res/2) >= xy_bounds[1]),
                                    np.where((nc_dat.y - res/2) <= xy_bounds[3]))
            XY = np.meshgrid(x_inds, y_inds)
            
            # trim the values to those within the catchment bounds rectangle            
            nc_dat = trim_netcdf(nc_dat, XY, var_name, the_dims,
                                 t_pos=t_pos, y_pos=y_pos, x_pos=x_pos)
            
            # generate weight matrix representing fractional coverage for calculating spatial means        
            weight_matrix = XY[0].copy().astype(np.float32) * 0
            catch_area = catchment_geom.area
            grid_cells = []
            for j, xi in enumerate(x_inds):
                xc = nc_dat.x.values[xi]
                xmax = xc + res/2
                xmin = xc - res/2
                for i, yi in enumerate(y_inds):
                    yc = nc_dat.y.values[yi]
                    ymax = yc + res/2
                    ymin = yc - res/2
                    thisbox = box(xmin, ymin, xmax, ymax)
                    grid_cells.append(thisbox)
                    weight_matrix[i,j] = catchment_geom.intersection(thisbox).area / catch_area
            
            # save for re-use
            self.weight_matrices[catchment_id][nc_type]['weight_matrix'] = weight_matrix
            self.weight_matrices[catchment_id][nc_type]['XY'] = XY

        if plot:
            self.plot_river(grid=nc_dat['trimmed_'+var_name])
        
        # use weight matrix to calculate weighted catchment aggregation. Check if the X/Y ordering and W.T makes sense
        if len(the_dims)>2:  
            if t_pos==0 and y_pos==1 and x_pos==2:
                catchment_average = np.sum(nc_dat['trimmed_'+var_name].values[:,XY[1],XY[0]] * weight_matrix, axis=(1,2)) / weight_matrix.sum()
            elif t_pos==0 and y_pos==2 and x_pos==1:
                catchment_average = np.sum(nc_dat['trimmed_'+var_name].values[:,XY[0],XY[1]] * weight_matrix.T, axis=(1,2)) / weight_matrix.sum()
            if t_pos==2 and y_pos==0 and x_pos==1:
                catchment_average = np.sum(nc_dat['trimmed_'+var_name].values[XY[1],XY[0],:] * weight_matrix, axis=(0,1)) / weight_matrix.sum()
            elif t_pos==2 and y_pos==1 and x_pos==0:
                catchment_average = np.sum(nc_dat['trimmed_'+var_name].values[XY[0],XY[1],:] * weight_matrix.T, axis=(0,1)) / weight_matrix.sum()
        else:
            catchment_average = np.sum(nc_dat['trimmed_'+var_name].values[XY[1],XY[0]] * weight_matrix) / weight_matrix.sum()
        
        return catchment_average
        
    def plot_river(self, reaches=True, boundaries=True, grid=None, stations=False):
        if not grid is None:
            grid.plot(alpha=0.6, vmin=grid.min().values, vmax=float(grid.max().values), cmap='Spectral')
        cm = plt.get_cmap('viridis')
        num_colours = self.network.reach_level.max() + 1
        if reaches:
            [plt.plot(*self.network.iloc[i].geometry.xy,
                      c=cm(self.network.iloc[i].reach_level/num_colours),
                      linewidth=1.2) 
                for i in range(self.network.shape[0])]
        if boundaries:
            [plt.plot(*self.boundaries_increm.loc[i].exterior.xy, ':',
                      c=cm(self.network.iloc[idx].reach_level/num_colours),
                      linewidth=0.7, alpha=0.9) 
                for idx, i in enumerate(self.network.id)]
        if stations:
            plt.plot(self.points_in_catchments.easting, self.points_in_catchments.northing, 'o')
        # add parent cumulative catchment outline
        plt.plot(*self.boundaries_cumul.loc[self.river_id].exterior.xy,
                 '--', c='k', linewidth=1.5, alpha=0.8)
        plt.show()

    def plot_flows(self, flow_dict, date, scaler=2.5, add_min=0.1):
        cm = plt.get_cmap('viridis')
        num_colours = self.network.reach_level.max() + 1
        [plt.plot(*self.network.iloc[i].geometry.xy,
                  c=cm(self.network.iloc[i].reach_level/num_colours),
                  linewidth = add_min + scaler*np.log(1 + flow_dict
                    [self.network.id.iloc[i]]
                    .at[date, 'flow']
                ))
            for i in range(self.network.shape[0])]

        plt.plot(
            self.points_in_catchments[
                self.points_in_catchments.nrfa_id.isin(list(self.flow_data.keys()))
            ].easting,
            self.points_in_catchments[
                self.points_in_catchments.nrfa_id.isin(list(self.flow_data.keys()))
            ].northing,
            'o', color='orange'
        )
        
        # add parent cumulative catchment outline
        plt.plot(*self.boundaries_cumul.loc[self.river_id].exterior.xy,
                 '--', c='k', linewidth=1.5, alpha=0.8)
        plt.show()

    def calculate_areas(self, catchment_type='increm'):
        if catchment_type=='increm':        
            return pd.DataFrame({'area': [self.boundaries_increm.loc[i].area 
                for i in self.network.id]}, index=list(self.network.id))
        else:
            return pd.DataFrame({'area': [self.boundaries_cumul.loc[i].area 
                for i in self.network.id]}, index=list(self.network.id))
    
    def calculate_lengths(self):
        return pd.DataFrame({'length': [self.network.geometry.iloc[i].length 
            for i in range(self.network.shape[0])]}, index=list(self.network.id))
    
    def find_stations_in_catchments(self, station_metadata, draincells):
        ''' station_metadata must be GeoDataFram with column "geometry"
        with Points of station locations in easting, northing '''
        # find which points lie within incremental catchment bounds
        polygons_to_test = (gpd.GeoDataFrame(self.boundaries_increm)
            .reset_index()
            .rename({'index':'id'}, axis=1)
        )
        points_in_catchments = gpd.sjoin_nearest(
            station_metadata, polygons_to_test, distance_col="distance"
        ).query('distance==0').drop('distance', axis=1)
        
        # calculate how far upstream each point is from the catchment drainage cell 
        norm_dist_upstream = []
        for k in range(points_in_catchments.shape[0]):
            cid = points_in_catchments.iloc[k].id
            line = self.network.loc[self.network.id==cid, 'geometry'].values[0]
            p = points_in_catchments.iloc[k].geometry
            p_normdist = line.project(p, normalized=True)
            
            # check whether distance is from upstream point or drainage cell    
            dc_point = (draincells
                .loc[draincells.DCid==self.network.loc[
                     self.network.id==cid, 'dc_id'].values[0]]
            )
            dc_point = Point(np.array([dc_point.x, dc_point.y]))
            dc_normdist = line.project(dc_point, normalized=True)
            if dc_normdist>0.5: # what about for very small reaches??
                # distance is downstream, so invert to find distance upstream
                if p_normdist > dc_normdist:
                    p_normdist = 0 # at drainage cell
                else:
                    p_normdist = 1 - p_normdist # upstream of drainage cell
            else:
                # distance is upstream
                if p_normdist<dc_normdist:
                    p_normdist = 0 # at drainage cell
            norm_dist_upstream.append(p_normdist)
        points_in_catchments['norm_dist_upstream'] = norm_dist_upstream
        self.points_in_catchments = points_in_catchments
    
    def load_flow_data(self, date_range, data_dir):
        # open netcdf files of 15 minute flow    
        for nrfa_st in self.points_in_catchments.nrfa_id:
            try:
                self.flow_data[nrfa_st] = xr.open_dataset(data_dir + '/%06d.nc' % (nrfa_st,))
            except:
                # some stations are missing
                continue
        
        # subset times to desired date range
        time_template = pd.DataFrame({'DATE_TIME':date_range})
        for k in list(self.flow_data.keys()):
            self.flow_data[k] = self.flow_data[k].sel(time = self.flow_data[k].time.isin(date_range))
            if len(self.flow_data[k].time)>0:
                self.flow_data[k] = self.flow_data[k].load()                
                self.flow_data[k] = (self.flow_data[k].to_dataframe()
                    .reset_index()[['time', 'flow_value']]
                    .rename({'time':'DATE_TIME', 'flow_value':'flow'}, axis=1)
                    .merge(time_template, how='right', on='DATE_TIME')
                    .set_index('DATE_TIME')
                )        
                if self.flow_data[k].dropna().shape[0]==0:
                    self.flow_data.pop(k)
            else:
                self.flow_data.pop(k)

    def generate_teacher_forcing_flows(self):
        ''' Generate estimated flows by scaling the observations by
        the cumulative catchment area of each sub catchment '''
        gauges = self.points_in_catchments[self.points_in_catchments.nrfa_id.isin(self.flow_data.keys())]
        
        X1 = gauges[['catchment-area', 'easting', 'northing']].values
        X2 = self.network[['ccar', 'xout', 'yout']].values

        X1[:,0] = X1[:,0] / X2[:,0].max()
        X2[:,0] = X2[:,0] / X2[:,0].max()
        X1[:,1:] = (X1[:,1:] - X2[:,1:].mean(axis=0)) / X2[:,1:].std(axis=0)
        X2[:,1:] = (X2[:,1:] - X2[:,1:].mean(axis=0)) / X2[:,1:].std(axis=0)
        
        nbrs = NearestNeighbors(n_neighbors=X1.shape[0], metric='euclidean').fit(X1)
        dists, knn = nbrs.kneighbors(X2, return_distance=True)
        one_ov_distssq = 1 / (dists**3)        

        self.flow_est = {}
        for i, rid in enumerate(self.network.id):
            nids = gauges.nrfa_id.values[knn[i,:]]
            normed_flows = np.stack([self.flow_data[nid].flow.values / gauges.loc[gauges.nrfa_id==nid, 'catchment-area'].values[0] for nid in nids], axis=1)
            
            weight_mask = np.isfinite(normed_flows)
            gauge_weights = weight_mask * np.repeat(one_ov_distssq[i,:][None,...], weight_mask.shape[0], axis=0)
            gauge_weights = gauge_weights / gauge_weights.sum(axis=1, keepdims=True)    
            
            allnans_mask = np.isnan(normed_flows).all(axis=1)
            this_flow = np.nansum(gauge_weights * normed_flows, axis=1)
            this_flow[allnans_mask] = np.nan
            this_ccar = self.network.loc[self.network.id==rid, 'ccar'].values[0]
            
            self.flow_est[rid] = self.flow_data[nids[0]].assign(flow = this_flow*this_ccar).fillna(method='ffill').fillna(method='bfill')

        ### OLD:
        # matched_gauges = pd.merge_asof(
            # self.cds_cumul.reset_index().sort_values('CCAR')[['index','CCAR']],
            # gauges[['nrfa_id', 'catchment-area']].sort_values('catchment-area'),
            # left_on="CCAR", right_on="catchment-area",
            # allow_exact_matches=True, direction="nearest"
        # ).set_index('index')
        # self.flow_est = {}
        # for rid in matched_gauges.index:
            # mg = matched_gauges.loc[rid]
            # nid = mg['nrfa_id']
            # ccar_norm = mg['catchment-area']
            # ccar_this = mg['CCAR']
            # self.flow_est[rid] = ccar_this * self.flow_data[nid] / ccar_norm
            # self.flow_est[rid] = self.flow_est[rid].fillna(method='ffill').fillna(method='bfill')

    def load_precip_data(self, date_range, data_dir):
        # accumulation of gear rainfall is foward to the hour label, 
        # so 11:00 is "rain from 10:01 to 11:00"        
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
        uniq_years = np.unique(dates_hr.year)
        uniq_months = {str(yy) : np.unique(dates_hr.month[np.where(dates_hr.year==yy)]) for yy in uniq_years}

        time_index = []
        self.precip_data = {} # reset dict
        res = 1000 # 1km
        for yy in uniq_years:
            for mm in uniq_months[str(yy)]:        
                cds = rioxarray.open_rasterio(data_dir + f'/{yy}/CEH-GEAR-1hr-v2_{yy}{zeropad_strint(mm)}.nc',
                                              decode_coords="all")
                rfd = cds[2].drop_vars(['lat', 'lon', 'min_dist', 'stat_disag'])
                rfd['x'] = rfd.x + res/2
                rfd['y'] = rfd.y + res/2
                del(cds)
                
                # hack using numpy datetime to subset datetimes
                rfd['time'] = rfd.time.astype("datetime64[ns]")
                rfd = rfd.sel(time = rfd.time.isin(dates_hr))
                these_times = pd.to_datetime(rfd['time'].values)
                if len(time_index)==0:
                    time_index = these_times
                else:
                    time_index = np.hstack([time_index, these_times])
                
                if len(self.precip_parent_bounds.keys())==0:                    
                    big_xy_bounds = self.boundaries_cumul.loc[self.river_id].bounds # (minx, miny, maxx, maxy)
                    bx_inds = np.intersect1d(np.where((rfd.x + res/2) >= big_xy_bounds[0]),
                                             np.where((rfd.x - res/2) <= big_xy_bounds[2]))
                    by_inds = np.intersect1d(np.where((rfd.y + res/2) >= big_xy_bounds[1]),
                                             np.where((rfd.y - res/2) <= big_xy_bounds[3]))
                    self.precip_parent_bounds['x_inds'] = bx_inds
                    self.precip_parent_bounds['y_inds'] = by_inds
                
                rfd_trim = rfd.isel(x=self.precip_parent_bounds['x_inds'],
                                    y=self.precip_parent_bounds['y_inds'])
                
                # do catchment averaging        
                for rid in self.network.id:
                    cav = self.catchment_average_netcdf(
                        rid,
                        rfd_trim,
                        'rainfall_amount',
                        res=res,
                        plot=False,
                        nc_type='gear'
                    )
                    if rid in self.precip_data.keys():
                        self.precip_data[rid] = np.hstack([self.precip_data[rid], cav])
                    else:
                        self.precip_data[rid] = cav

        ## create 15 min data frames
        time_index = pd.DatetimeIndex(time_index)
        time_index.name = 'DATE_TIME'
        time_template = pd.DataFrame({'dummy':1, 'DATE_TIME':date_range})
        for k in self.precip_data.keys():
            rain = pd.DataFrame({'precip':self.precip_data[k], 'DATE_TIME':time_index})
            rain = time_template.merge(rain, on='DATE_TIME', how='outer').sort_values('DATE_TIME')
            rain[['precip']] = rain[['precip']].fillna(method='bfill') / 4 # backfill hourly to 15 min
            rain = (rain.drop('dummy', axis=1)
                .set_index('DATE_TIME')
                .loc[date_range]
            )
            self.precip_data[k] = rain        

    def load_soil_wetness(self, date_range, vwc_quantiles, sm_data_dir):
        # do this for the antecedent time point, but then also for every
        # time where we click over into a new day?
        # "assimilate" the soil wetness into the state vector at start of each day?
        res = 1000
        
        uniq_days = date_range[(date_range.hour == 0) & (date_range.minute == 0)]
        uniq_years = np.unique(date_range.year)
        uniq_months = {str(yy) : np.unique(date_range.month[np.where(uniq_days.year==yy)]) for yy in uniq_years}
        self.soil_wetness_data = {} # reset dict
        for iy, yy in enumerate(uniq_years):
            for im, mm in enumerate(uniq_months[str(yy)]):
                this_sm_obj = rioxarray.open_rasterio(sm_data_dir + f'/{yy}/SM_{yy}{zeropad_strint(mm)}.nc')

                if len(self.soilmoisture_parent_bounds.keys())==0:
                    big_xy_bounds = self.boundaries_cumul.loc[self.river_id].bounds # (minx, miny, maxx, maxy)
                    bx_inds = np.intersect1d(np.where((this_sm_obj.x + res/2) >= big_xy_bounds[0]),
                                             np.where((this_sm_obj.x - res/2) <= big_xy_bounds[2]))
                    by_inds = np.intersect1d(np.where((this_sm_obj.y + res/2) >= big_xy_bounds[1]),
                                             np.where((this_sm_obj.y - res/2) <= big_xy_bounds[3]))
                    self.soilmoisture_parent_bounds['x_inds'] = bx_inds
                    self.soilmoisture_parent_bounds['y_inds'] = by_inds

                if iy==0 and im==0:
                    # do antecedent weighted soil moisture
                    prev_day = date_range[0] - pd.to_timedelta("1D")
                    merged_sm = merge_two_soil_moisture_days(
                        this_sm_obj,
                        date_range[0],
                        prev_day,
                        sm_data_dir,
                        self.soilmoisture_parent_bounds['x_inds'],
                        self.soilmoisture_parent_bounds['y_inds']
                    )
                    
                    # do this only once
                    vwc_quantiles_trim = vwc_quantiles.isel(x=self.soilmoisture_parent_bounds['x_inds'],
                                                            y=self.soilmoisture_parent_bounds['y_inds'])
                    
                    merged_sm = calculate_soil_wetness(merged_sm, vwc_quantiles_trim, incl_time=False)
                    
                    antecedent_soil_wetness = {}
                    for rid in self.network.id:
                        cav = self.catchment_average_netcdf(
                            rid,
                            merged_sm,
                            'soil_wetness',
                            res=res,
                            plot=False,
                            nc_type='soil_moisture'
                        )
                        antecedent_soil_wetness[rid] = cav    
                    self.antecedent_soil_wetness = pd.DataFrame(antecedent_soil_wetness, index=['soil_wetness']).T

                this_sm_trim = this_sm_obj.isel(x=self.soilmoisture_parent_bounds['x_inds'],
                                                y=self.soilmoisture_parent_bounds['y_inds'])
                # hack using numpy datetime to subset datetimes
                this_sm_trim['time'] = this_sm_trim.time.astype("datetime64[ns]")
                this_sm_trim = this_sm_trim.sel(time = this_sm_trim.time.isin(uniq_days))

                this_sm_trim = calculate_soil_wetness(this_sm_trim, vwc_quantiles_trim)
                    
                soil_wetness_dict = {}
                for rid in self.network.id:
                    cav = self.catchment_average_netcdf(
                        rid,
                        this_sm_trim,
                        'soil_wetness',
                        res=res,
                        plot=False,
                        nc_type='soil_moisture'
                    )
                    if rid in soil_wetness_dict.keys():
                        soil_wetness_dict[rid] = np.hstack([soil_wetness_dict[rid], cav])
                    else:
                        soil_wetness_dict[rid] = cav
                        
        for k in soil_wetness_dict.keys():
            soil_wetness = pd.DataFrame({'soil_wetness':soil_wetness_dict[k], 'DATE_TIME':uniq_days})
            self.soil_wetness_data[k] = soil_wetness.set_index('DATE_TIME')

    def aggregate_landcover_classes(self):
        # raw classes
        '''
        Broadleaved woodland    1
        ‘Coniferous Woodland’   2
        ‘Arable and Horticulture’   3
        ‘Improved Grassland’    4
        ‘Neutral Grassland’ 5
        ‘Calcareous Grassland’  6
        Acid grassland  7
        ‘Fen, Marsh and Swamp’  8
        Heather 9
        Heather grassland   10
        ‘Bog’   11
        ‘Inland Rock’   12
        Saltwater   13
        Freshwater  14
        ‘Supra-littoral Rock’   15
        ‘Supra-littoral Sediment’   16
        ‘Littoral Rock’ 17
        Littoral sediment   18
        Saltmarsh   19
        Urban   20
        Suburban    21
        '''
        # # from landcover map aggregations
        # c_agg = {
            # '1':[1], # Broadleaf woodland
            # '2':[2], # Coniferous woodland
            # '3':[3], # Aarable
            # '4':[4], # Improved grassland
            # '5':[5,6,7,8], # Semi-natural grassland
            # '6':[9,10,11,12], # Mountain, heath and bog
            # '7':[13], # Saltwater
            # '8':[14], # Freshwater
            # '9':[15,16,17,19], # Coastal
            # '10':[20,21] # Built-up areas and gardens
        # }
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
            agg_1990.append(self.cds_increm[[f'LC1990_{zeropad_strint(n)}' for n in c_agg[k]]].sum(axis=1))
            agg_2015.append(self.cds_increm[[f'LC2015_{zeropad_strint(n)}' for n in c_agg[k]]].sum(axis=1))
            
        agg_1990 = pd.concat(agg_1990, axis=1)
        agg_1990.columns = [f'LC1990_{zeropad_strint(int(n))}' for n in c_agg.keys()]
        agg_2015 = pd.concat(agg_2015, axis=1)
        agg_2015.columns = [f'LC2015_{zeropad_strint(int(n))}' for n in c_agg.keys()]
        
        # remove negatives (they are small) and renormalize to 1
        agg_2015 = agg_2015.clip(lower=0)
        agg_2015 = (agg_2015 / agg_2015.sum(axis=1).values[...,None])
        agg_1990 = agg_1990.clip(lower=0)
        agg_1990 = (agg_1990 / agg_1990.sum(axis=1).values[...,None])
        
        self.cds_increm = self.cds_increm.drop(
            [col for col in self.cds_increm if col.startswith('LC')], axis=1
        )
        
        indexname = self.cds_increm.index.name
        if indexname is None:
            indexname = 'index'
        
        self.cds_increm = (self.cds_increm.reset_index()
            .merge(agg_1990.reset_index(), on=indexname)
            .merge(agg_2015.reset_index(), on=indexname)
            .set_index(indexname)
        )
    
    def calculate_lc_for_event(self):        
        mean_year = self.precip_data[self.river_id].index.mean().year
        if mean_year > 2015:
            mean_year = 2015
        elif mean_year < 1990:
            mean_year = 1990
        self.mean_lc = (
            (1-(mean_year - 1990)/(2015-1990)) * self.cds_increm.loc[:, self.lc_1990_names].values + 
            (1-(2015 - mean_year)/(2015-1990)) * self.cds_increm.loc[:, self.lc_2015_names].values
        )
        self.mean_lc = pd.DataFrame(self.mean_lc, index=self.cds_increm.index)
        self.lc_names = [f'LC_{zeropad_strint(int(n))}' for n in range(1, self.mean_lc.shape[1]+1)]
        self.mean_lc.columns = self.lc_names        
        
        if mean_year > 2000:
            mean_year = 2000
        elif mean_year < 1990:
            mean_year = 1990
        self.mean_urbext = (
            (1-(mean_year - 1990)/(2000-1990)) * self.cds_increm.loc[:, ["QUEX"]].values + 
            (1-(2000 - mean_year)/(2000-1990)) * self.cds_increm.loc[:, ["QUE2"]].values
        )
        self.mean_urbext = pd.DataFrame({'QUEX':self.mean_urbext.flatten()},
                                        index=self.cds_increm.index)
    

def load_event_data(rid, date_range, vwc_quantiles=None):
    ## choose catchment and load river object
    river_obj = River()
    river_obj.load(save_rivers_dir, rid)

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
    
    return river_obj
