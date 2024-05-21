import numpy as np
import pandas as pd

## define the path to the data
basedir = '/gws/nopw/j04/hydro_jules/data/uk/flood_events/'

def wrangle_descriptors():
    ## catchment descriptors
    catch_desc = pd.read_csv(basedir + '/feh-catchment-desc.csv')

    # remove duplicated rows
    catch_desc = catch_desc.drop_duplicates()

    # extract descriptor name/id map
    desc_names = catch_desc[['PROPERTY_ITEM', 'TITLE']].drop_duplicates()

    # reorganise descriptors into row vector table
    catch_desc = catch_desc.rename({'STATION':'Station'}, axis=1)
    catch_desc = (catch_desc[['Station', 'PROPERTY_ITEM', 'PROPERTY_VALUE']]
        .pivot_table(columns='PROPERTY_ITEM', index='Station', values='PROPERTY_VALUE'))
        
    return catch_desc, desc_names
    
def wrangle_catchment_nesting():
    ## nested catchments
    catch_relations = pd.read_csv(basedir + '/nestedCatchments.csv')

    # separate parents from single Parent column
    parents = catch_relations['Parent'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    parents = parents.str.split(' ', expand=True)
    parents = parents.replace(to_replace=np.nan, value=None)
    par_nums = 'parent_' + np.array(range(parents.shape[1]), dtype=str).astype(object)
    parents.columns = par_nums

    # join to station IDs
    catch_relations = catch_relations.drop('Parent', axis=1)
    nest_catch = pd.concat([catch_relations, parents], axis=1)

    # collapse table
    nest_catch = (nest_catch.melt(id_vars=['ID', 'Nested'])
            .rename({'value':'Parent_ID', 'variable':'Parent_num'}, axis=1)
            .dropna())
    nest_catch['Parent_ID'] = nest_catch['Parent_ID'].astype(int)

    # and extract the un-nested catchments, either tiny separate catchments
    # or larger parent catchments
    unnest_catch = catch_relations[catch_relations['Nested']=='NO']

    return (nest_catch.rename({'ID':'Station'}, axis=1),
            unnest_catch.rename({'ID':'Station'}, axis=1))
    
if __name__=="__main__":

    # load and wrangle the data
    catch_desc, desc_names = wrangle_descriptors()
    nest_catch, unnest_catch = wrangle_catchment_nesting()
    
    # then we can pull out a vector of descriptors for a station using
    STATION_ID = 2001
    print(catch_desc.loc[STATION_ID])

    # and we can find the largest parent catchment for any nested catchment with
    STATION_ID = 39108 # an ID from nest_catch.Station
    print(
        unnest_catch[unnest_catch['Station'].isin(
            np.asarray(nest_catch[nest_catch['Station']==STATION_ID]['Parent_ID'])
        )]
    )

    # we can also find all the catchments nested within a larger un-nested one with
    STATION_ID = 39001 # an ID from unnest_catch.ID
    print(nest_catch[nest_catch['Parent_ID']==STATION_ID])
    
    # can we find the "tree" of dependent catchments, going down from 
    # the largest and branching off the to the smallest subcatchments?
    STATION_ID = 39001 # an ID from unnest_catch.ID
    parent_areas = catch_desc.reset_index()[['Station', 'CCAR']].rename({'Station':'Parent_ID', 'CCAR':'parent_area'}, axis=1)
    subcatch_areas = catch_desc.reset_index()[['Station', 'CCAR']].rename({'CCAR':'subcatch_area'}, axis=1)
    all_sub_catches = nest_catch[nest_catch['Parent_ID']==STATION_ID][['Station']]
    all_connections = pd.DataFrame()
    for substat in all_sub_catches.Station:
        this_connex = (nest_catch[nest_catch.Station==substat].merge(parent_areas, on='Parent_ID')
            .merge(subcatch_areas, on='Station')
            .sort_values('parent_area')
        )
        all_connections = pd.concat([all_connections, this_connex], axis=0)
    # to find the bottom of the tree, find the stations 
    # that are not in the parent column
    roots = np.setdiff1d(all_connections.Station, all_connections.Parent_ID)
    
    ## or look at the individual river reaches!

    if False:
        # and we can pull out the 1km pixels associated with each 
        # sub-catchment using the catchment boundaries
        catch_id = 39001
        
        import rioxarray
        import rasterio
        import datetime
        import time
        import geopandas as gpd
        import xarray as xr
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from shapely.geometry import Point, Polygon, MultiPoint
        from shapely.strtree import STRtree

        from utils import zeropad_strint

        hj_base = '/gws/nopw/j04/hydro_jules/data/uk/'
        home_data_dir = '/home/users/doran/data_dump/'

        # open catchment boundaries
        gdf = gpd.read_file(home_data_dir + '/nrfa/V12_NRFA_CatchmentsAll.gpkg')

        # subset to specific catchment boundary
        catch_shape = gdf.loc[gdf['CatchID']==str(catch_id)].geometry.values
        
        # extract xy points and create within-boundary mask
        xy_bounds = catch_shape.bounds

        # find easting/northing grid values for catchment using GEAR data
        data_dir = '/gws/nopw/j04/ceh_generic/netzero/downscaling/ceh-gear/'
        cds = rioxarray.open_rasterio(data_dir + f'/2010/CEH-GEAR-1hr-v2_201001.nc', decode_coords="all")
        rfd = cds[2].drop_vars(['lat', 'lon', 'min_dist', 'stat_disag']).isel(time=0)
        rfd.rio.write_crs("epsg:27700", inplace=True)
        del(cds)
        # pull out rectangular grid of catchment
        # assuming coordinate is at bottom left of pixel
        x_inds = np.intersect1d(np.where(rfd.x >= xy_bounds[0,0])[0],
                                np.where(rfd.x < (xy_bounds[0,2] + 1000))[0])
        y_inds = np.intersect1d(np.where(rfd.y >= xy_bounds[0,1])[0],
                                np.where(rfd.y < (xy_bounds[0,3] + 1000))[0])    
        catch_dat = rfd.isel(x=x_inds).isel(y=y_inds).copy()
        
        # clip rectangular grid to specific catchment boundary
        catch_dat = catch_dat.rio.clip(catch_shape, rfd.rio.crs)
        yi, xi = np.where(catch_dat.rainfall_amount.values >= 0)
        
        # define the pairs of easting and northing values for future use
        y_vals = catch_dat.y.values[yi] 
        x_vals = catch_dat.x.values[xi]
        
        
        ## how to clip a catchment using a grid of easting/northing values
        xygrid = xr.load_dataset(hj_base + '/ancillaries/chess_landfrac.nc',
                                 decode_coords="all")
        # pull out rectangular grid of catchment
        # assuming coordinate is at centre of pixel
        x_inds2 = np.intersect1d(np.where((xygrid.x - 500) >= xy_bounds[0,0])[0],
                                 np.where((xygrid.x - 500) < (xy_bounds[0,2] + 1000))[0])
        y_inds2 = np.intersect1d(np.where((xygrid.y - 500) >= xy_bounds[0,1])[0],
                                 np.where((xygrid.y - 500) < (xy_bounds[0,3] + 1000))[0])    
        latlon_catchment = xygrid.isel(x=x_inds2).isel(y=y_inds2)
        
        # create easting northing grid to index, moving origin to bottom left of pixel 
        easting_northing_grid = np.meshgrid(latlon_catchment.x.values - 500,
                                            latlon_catchment.y.values - 500)
        mask = sum([(easting_northing_grid[0]==x_vals[i]) * (easting_northing_grid[1]==y_vals[i]) for i in range(len(x_vals))])
        mask = mask.astype(float)
        mask[mask==0] = np.nan
        plt.imshow(latlon_catchment.landfrac.values * mask); plt.show()
        plt.imshow(latlon_catchment.lat.values * mask); plt.show()
        plt.imshow(latlon_catchment.lon.values * mask); plt.show()
        
        # pull out the lat/lon vals too (these will be pixel centre)
        lat_vals = (latlon_catchment.lat.values * mask).flatten()
        lat_vals = lat_vals[~np.isnan(lat_vals)]
        lon_vals = (latlon_catchment.lon.values * mask).flatten()
        lon_vals = lon_vals[~np.isnan(lon_vals)]

        
        ## use the catchment boundary or easting/northing values or lat/lon values
        ## to subset topography / land cover / soil data / meteorology etc
        
        ## then do this for all sub-catchments within the larger catchment?
        ## ideally for each river reach or 1km river pixel so we can build
        ## up the "slices" of extra catchment that drain into each piece of river
