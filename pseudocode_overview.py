### Pseudo code for flow model

""" Set up a watershed by loading a river object and data """

## choose catchment and load river object
'''
'the river object contains:
'   - connectivity of reaches within the watershed in river_obj.network (id/dwn_id)
'   - cumulative catchment descriptors in river_obj.cds_cumul
'   - incremental catchment descriptors in river_obj.cds_increm
'   - cumulative catchment boundaries in river_obj.boundaries_cumul
'   - incremental catchment boundaries in river_obj.boundaries_increm
'   - reach line strings in river_obj.network (geometry)
'   - gauging stations that lie in the catchment and the id of the reach
'       they belong to, as well as the distance upstream of the drainage
'       cell of that reach, in river_obj.points_in_catchments
'''
rid = 59387 # 15340
river_obj = River()
river_obj.load(save_rivers_dir, rid)

## define date range
'''
'this could be taken from a particular flood event from the flood event archive
'''
date_range = pd.date_range(start="1995/04/08 11:45:00", end="1995/04/10 13:15:00", freq='15min')

## load flow observations
'''
'this is NRFA 15 minute flow data. Not all gauging stations are present in the dataset
'''
river_obj.load_flow_data(date_range, flow_dir) # creates river_obj.flow_data

## generate initial flow condition across entire river network
'''
'this is a hack to get a start value for F by scaling the observed flow 
'at all NRFA stations by the cumulative catchment area for every reach in the catchment
'''
river_obj.generate_initial_flows(date_range) # creates river_obj.init_flows

## load precip
'''
'this loads GEAR hourly rainfall and spreads it to a 15 minute time
'template to match the flow date range 
'''
river_obj.load_precip_data(date_range, gear_dir) # populates river_obj.precip_data

## load SM
'''
'this loads modelled daily soil moisture and scales by upper and lower
'95% quantiles to get soil wetness. We calculate the antecedent soil
'wetness for each sub catchment by taking the weighted average of the 
'first day of date_range and the previous day. We also calculate the 
'soil wetness for each "new day" tick within date_range so we can
'assimilate the soil wetness each day into the catchment storage calculation
'''
vwc_quantiles = rioxarray.open_rasterio(sm_data_dir + '/vwc_quantiles.nc')
river_obj.load_soil_wetness(date_range, vwc_quantiles, sm_data_dir)  # populates river_obj.soil_wetness_data and river_obj.antecedent_soil_wetness

## approximate land cover from 1990 and 2015 values
'''
'for a given date range, do weighted average of 1990 and 2015 land cover 
'''
river_obj.calculate_lc_for_event() # creates river_obj.mean_lc


""" Define the model structure """

'''
'There are 3 quantities that we want to track for each reach:
'   - F, the flow at the drainage cell;
'   - L, the lateral in-flow from the land into the reach;
'   - S, the storage state of the catchment.
'Each variable (F, L and S) will be calculated as functions of a combination 
'of time-dependent inputs and static features. We will also input encodings
'of the time of the year and time of the day to act as seasonal indicators
'as well as local positional encodings for the attention calculations:
'   - sin(day_of_year / 365 * 2*PI), cos(day_of_year / 365 * 2*PI)
'   - sin((hour + minute/60) / 24 * 2*PI), cos((hour + minute/60) / 24 * 2*PI)

'At the beginning of each event timeseries, we need to calculate: 
'   - the initial storage state, S0, for each sub-catchment from the
'       antecedent soil wetness and catchment descriptors
'   - an estimate of the initial flow state, potentially by scaling the
'       first NRFA flow observation by the cumulative catchment area 
'       and assuming homeostatic flow across the whole watershed
'''

## calculate antecedent storage state, S(t=t0)
S0 = f_1(antecedent_soil_wetness, catchment_descriptors)

## calculate the lateral in-flow, L, and the new storage state, S, for the first timepoint
'''
'This is done using two steps: creation of an embedding vector LS_latent
'that looks back over the history of the precip and storage timeseries, and
'then point-wise transformations of that embedding into values of the 
'lateral in-flow and storage.
'''
LS_latent = f_2(precip(t=t0), S0, catchment_descriptors, time_encodings(t=t0))
L(t=t0) = f_3(LS_latent)
S(t=t0+1) = f_4(LS_latent)

## approximate the flow at t=t0 using cumulative catchment area scaling
'''
'Divide the first NRFA flow observation by the NRFA station cumulative 
'catchment area and then scale up by the current sub-catchment cumulative
'catchment area (not the incremental catchment area!)
'''
F(t=t0) = NRFA_flow(t=t0) / NRFA_CCAR   *  CCAR 

'''
'Now we have everything at the first timepoint, we can start the "proper"
'algorithm to run through the rest of the time series we want to predict
'''
for t_i in (t0+1, tN):
    for reach from headwaters to river mouth:
        ## if t_i is start of a new day (00:00:00) then we assimilate new
        ## daily soil wetness data to calculate S(t=t_i) and take average of values
        if t_i is day start:
            S_assim = f_1(soil_wetness(previous day), catchment_descriptors)
            S(t=t_i) = (S(t=t_i) + S_assim) / 2
        
        ## calculate the lateral in-flow, L, and the new storage state, S, for the first timepoint
        LS_latent = f_2(precip(t<=t_i), S(t<=t_i), catchment_descriptors, time_encodings(t<=t_i))
        L(t=t_i) = f_3(LS_latent)
        S(t=t_i+1) = f_4(LS_latent)

        ## calculate upstream flow contribution
        if in headwaters:
            F_u(t=t_i) = 0
        else:
            Fup_sum = sum(F from upstream_cells)            
            Fu_latent = f_5(Fup_sum(t<=t_i), reach_features, time_encodings(t<=t_i))
            F_u(t=t_i) = f_6(Fu_latent)

        ## calculate flow contribution from lateral in-flow        
        Fl_latent = f_7(L(t<=t_i), reach_features, time_encodings(t<=t_i))
        F_l(t=t_i) = f_6(Fl_latent)
        
        ## calculate the attenuation of the current flow ("history effect")        
        Fh_latent = f_8(F((t_i-lookback)<=t<t_i), flow_history_features, time_encodings((t_i-lookback)<=t<t_i))
        F_h(t=t_i) = f_6(Fh_latent)
        
        ## sum the flow contributions to get the flow at current timestep
        F(t=t_i) = F_u(t=t_i) + F_l(t=t_i) + F_h(t=t_i)
    
'''
'The "features" are static characteristics:
'   - catchment_descriptors are 
'        QB19, # BFIHOST19
'        QUEX, # URBEXT1990 Urban extent
'        QUE2, # URBEXT2000 Urban extent
'        QFPX, # Mean flood plain extent
'        QFPD, # Mean flood plain depth
'        QPRW, # PROPWET Proportion of time soils are wet
'        QSPR, # Standard percentage runoff from HOST        
'        QDPS, # DPSBAR Mean drainage path slope
'        QFAR,  # FARL Flood attenuation by reservoirs and lakes
'        LC_XX, # Aggregated land cover
'        HGB_XX and HGS_XX, # Geology
'        and potentially incremental catchment area
'    - reach_features are reach length and reach slope
'    - flow_history_features are reach slope and cumulative catchment area
'
'The f_X functions are neural networks:
'    - f_1, f_3, f_4 and f_6 are simple linear MLP-type networks
'    - f_2, f_5 and f_7 are attention-based transformer-type encoders
'    - f_8 is a 1D convolutional network
'    
'The premise is that we are working as much as possible in "physical space"
'where L, S, F and its contributions F_u, F_l and F_h are all scalar
'quantities with direct physical meaning. That means we aren't passing
'large embedding vectors around, we will be working with light-weight
'networks that will hopefully allow successful training on a very
'unconstrained system.
'''
        
        
        
        
        
