import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
import rioxarray
import copy
import pkbar

from torch.optim import Adam
from types import SimpleNamespace
from pathlib import Path

from architectures.river_reach_model import ReachFlow
from utils import setup_checkpoint, prepare_run, update_checkpoint, save_checkpoint
from event_wrangling import load_statistics
from river_class import River, load_event_data, sm_data_dir, hj_base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

static_features = dict(
    c_feats = [
        'QB19', # BFIHOST19
        'QUEX', # urban extent, merging QUEX and QUE2
        'QDPS', # drainage path slope
        'ICAR', # incremental catchment area
        'QDPL', # mean drainage path length
        'DRAINAGE_DENSITY', # channel length / catchment area
        'HGS_XX', # superficial hydrogeology 
        'HGB_XX', # bedrock hydrogeology
        'LC_XX' # land cover classes, merging LC1990 and LC2015
    ],
    r_feats = [
        'REACH_LENGTH',
        'REACH_SLOPE',
        'CCAR' # cumulative catchment area, as some proxy of cross-sectional area of channel?
    ]
)
c_feat_length = [s if "_XX" in s else 1 for s in static_features['c_feats']]
for i in range(len(c_feat_length)):
    if c_feat_length[i]=='HGS_XX' or c_feat_length[i]=='HGB_XX':
        c_feat_length[i] = 5 # each have 5 hydrogeology classes
    if c_feat_length[i]=='LC_XX':
        c_feat_length[i] = 10 # aggregated to 10 land cover classes
static_features['c_feat_length'] = sum(c_feat_length)
static_features['r_feat_length'] = 3

norm_dict = dict(
    QDPS = 100, # [0.022517, 738.293548], med = 78.3
    ICAR = 25, # [0.002500, 264.972500], med = 2.725000
    REACH_LENGTH = 1000, # [50 , 44808.178182], med = 1257.106781
    REACH_SLOPE = 5, # [0, 34.779979], med = 0.527106
    CCAR = 100, # [3.00000, 9971.32250], med = 12.28125
    QDPL = 1000, # [7.071068, 18064.327412], med = 1519.811097
    DRAINAGE_DENSITY = 2, # [0.002611, 28.284271], med = 0.943428
    PRECIP = 100,
    FLOW = 50
)

cfg = SimpleNamespace(
    # model params
    dropout = 0.1,
    d_model = 16,
    d_ff = 32,
    N_h = 2,
    N_l = 2,
    d_s_in = 1 + static_features['c_feat_length'],
    d_ls_src = 7 + static_features['c_feat_length'],
    d_ls_trg = 6 + static_features['c_feat_length'],
    d_instream_src = 8 + static_features['r_feat_length'],
    d_instream_trg = 6 + static_features['r_feat_length'],
        
    # training params
    max_batch_size = 10,
    lr = 1e-4,
    max_epochs = 100,
    train_len = 15,
    val_len = 8,
    feature_names = static_features,
    norm_dict = norm_dict
)

def parse_event_string(event_string):
    event = pd.Series({c[0].lstrip():c[-1].lstrip() 
        for c in [s.lstrip().split("   ") for s in event_string.split('\n')]})
    event.nrfa_id = int(event.nrfa_id)
    event.id = int(event.id)
    event.dc_id = int(event.dc_id)
    event.xout = float(event.xout)
    event.yout = float(event.yout)
    event.basin_area = float(event.basin_area)
    event.FlowStartDate = pd.to_datetime(event.FlowStartDate)
    event.FlowEndDate = pd.to_datetime(event.FlowEndDate)
    return event

def sample_event(flood_event_df, vwc_quantiles=None,
                 event=None, rid=None, date_range=None):    
    got_working_event = False    
    while not got_working_event:
        # do the event selection
        if event is not None:
            rid = event.id
            date_range = pd.date_range(start=event.FlowStartDate, end=event.FlowEndDate, freq='15min')
        elif (date_range is not None) and (rid is not None):
            pass
        else:
            event = flood_event_df.sample(1).iloc[0]
            rid = event.id
            date_range = pd.date_range(start=event.FlowStartDate, end=event.FlowEndDate, freq='15min')
        
        # attempt to load event data
        try:
            print(rid)
            print(event)
            print(date_range[0])
            print(date_range[-1])
            river_obj = load_event_data(rid, date_range, vwc_quantiles=vwc_quantiles)
            # add checks for NaNs in soil wetness and precip data?
            
            got_working_event = True
            print("Got event!")
        
        except: # error with loading corrupt/missing/broken flow data 
            got_working_event = False
            print("Failed to load event...")
            
        if not got_working_event:
            event = None
            rid = None
            date_range = None
            print("Failed to load event...")
        
    return river_obj, rid, date_range, event

def fit(model, cfg, flood_events_train, flood_events_val, vwc_quantiles,
        nt_opt, teacher_forcing, outdir='/logs/', checkpoint=None, device=None):
    ## setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_best = False
    losses, val_losses, curr_epoch, best_loss = prepare_run(checkpoint)    
    
    ## train
    for epoch in range(curr_epoch, cfg.max_epochs):
        # optimisation
        model.train()
        kbar = pkbar.Kbar(target=cfg.train_len, epoch=epoch,
                          num_epochs=cfg.max_epochs,
                          width=15, always_stateful=False)        
        running_ls = []
        opt_ls = []        
        for bidx in range(1, cfg.train_len+1):                                    
            river_obj, rid, date_range, event = sample_event(flood_events_train, vwc_quantiles)
            event_bar = pkbar.Kbar(
                target=len(date_range),                
                width=15, always_stateful=False
            )
            running_event_ls = []       
            for tstep in range(len(date_range)):
                river_obj, loss_tstep, loss_opt, opt_counter = model.forward(
                    river_obj, date_range, tstep, device, nt_opt=nt_opt,
                    teacher_forcing=teacher_forcing, train=True
                )
                if loss_tstep is not None:                    
                    running_event_ls.append(loss_tstep)
                    event_bar.update(tstep, values=[('loss_tstep', loss_tstep)])
                if opt_counter==0:
                    if loss_opt is not None:
                        opt_ls.append(loss_opt)
            running_ls += running_event_ls
            print_values = [('loss_tstep', np.mean(running_event_ls)),
                            ('opt_loss', opt_ls[-1])]
            kbar.update(bidx, values=print_values)
        losses.append(np.mean(running_ls)) # append epoch average loss
        
        # validation
        with torch.no_grad():
            kbarv = pkbar.Kbar(target=cfg.val_len, epoch=epoch,
                               num_epochs=cfg.max_epochs,
                               width=15, always_stateful=False)
            model.eval()
            running_ls = []            
            for bidx in range(1, cfg.val_len+1):
                river_obj, rid, date_range, event = sample_event(flood_events_val, vwc_quantiles)
                running_event_ls = []
                event_bar = pkbar.Kbar(
                    target=len(date_range),                
                    width=15, always_stateful=False
                )
                for tstep in range(len(date_range)):
                    river_obj, loss_tstep, _, _ = model.forward(
                        river_obj, date_range, tstep, device, nt_opt=nt_opt,
                        teacher_forcing=teacher_forcing, train=False
                    )
                    if loss_tstep is not None:                    
                        running_event_ls.append(loss_tstep)
                        event_bar.update(tstep, values=[('loss_tstep', loss_tstep)])                    
                running_ls += running_event_ls               
                
                print_values = [('loss', running_ls[-1])]
                kbarv.update(bidx, values=print_values)
                
            val_losses.append(np.mean(running_ls))
            kbar.add(1, values=[("val_loss", val_losses[-1])])
            
            is_best = bool(val_losses[-1] < best_loss)
            best_loss = min(val_losses[-1], best_loss)
            checkpoint = update_checkpoint(epoch, model, best_loss, losses, val_losses)
            save_checkpoint(checkpoint, is_best, outdir)
        print("Done epoch %d" % (epoch+1))
    return model, losses, val_losses

if __name__=="__main__":
    ## file paths
    log_dir = './logs/'
    model_name = f'reach_flow_model'
    model_outdir = f'{log_dir}/{model_name}/'
    Path(model_outdir).mkdir(parents=True, exist_ok=True)

    # training flags
    load_prev_chkpnt = True
    specify_chkpnt = f'{model_name}/checkpoint.pth' # if None, load best, otherwise "modelname/checkpoint.pth"
    reset_chkpnt = False

    ## create model
    model = ReachFlow(cfg)
    print("Number of trainable parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.to(device)
    model.setup_optimizer(cfg)

    ## load checkpoint
    model, checkpoint = setup_checkpoint(
        model, device, load_prev_chkpnt,
        model_outdir, log_dir,
        specify_chkpnt=specify_chkpnt,
        reset_chkpnt=reset_chkpnt
    )

    ## load events and link nrfa stations to river IDs    
    gauge_river_map = pd.read_csv(hj_base + '/flood_events/station_to_river_map.csv')
    flood_event_df = load_statistics()

    # constrain by event length
    lower_bound = 24*4 # one day
    upper_bound = 5*24*4 # five days #28**2 * 2 - 1 # something else?  # 30*24*4 # one month
    flood_event_df = flood_event_df[
        (flood_event_df.EventDuration_15min >= lower_bound) &
        (flood_event_df.EventDuration_15min <= upper_bound)
    ]

    # link events to tidal river segment IDs
    flood_event_df = (flood_event_df.rename({'Station':'nrfa_id'}, axis=1)
        [['nrfa_id', 'Event', 'FlowStartDate', 'FlowEndDate']]
        .merge(gauge_river_map, on='nrfa_id')
        .dropna()
    )
    
    # constrain by basin cumulative catchment area to not initially 
    # train on huge rivers as this is very slow
    large_river_events = flood_event_df[flood_event_df.basin_area >= 4000]
    flood_event_df = flood_event_df[flood_event_df.basin_area < 4000]
    
    # split into training, validation and testing set
    uniq_ids, ev_cnts = np.unique(flood_event_df.id, return_counts=True)
    holdout_every = 10
    val_ratio = 0.3
    holdout_ids = uniq_ids[np.argsort(ev_cnts)[::-1][3::holdout_every]]
    leftover_ids = np.setdiff1d(uniq_ids, holdout_ids)
    test_events = flood_event_df[flood_event_df.id.isin(holdout_ids)]
    fit_events = flood_event_df[~flood_event_df.id.isin(holdout_ids)]
    val_events = fit_events.sample(frac=val_ratio, replace=False, random_state=22)
    train_events = pd.concat([fit_events, val_events], axis=0).drop_duplicates(keep=False)
    del(fit_events)

    # load vwc_quantiles only once
    vwc_quantiles = rioxarray.open_rasterio(sm_data_dir + '/vwc_quantiles.nc')

    if False:
        ## test sampling an event and load up the river / data
        river_obj, rid, date_range, event = sample_event(flood_event_df, vwc_quantiles=vwc_quantiles)
        river_obj.plot_river(stations=True)
        river_obj.plot_flows(river_obj.flow_est, date_range[1])

    ## train model
    teacher_forcing = True
    nt_opt = 1
    
    # should eventually turn teacher forcing off and increase nt_opt gradually
    model, losses, val_losses = fit(
        model,
        cfg,
        train_events,
        val_events,
        vwc_quantiles,
        nt_opt,
        teacher_forcing,
        outdir=model_outdir,
        checkpoint=checkpoint,
        device=device
    )


    if False:
        #################
        ## tests
        #################
        
        event_string = """nrfa_id                              106003
            Event                              Event_33
            FlowStartDate           2013-02-12 23:00:00
            FlowEndDate             2013-02-15 09:00:00
            id                                    17560
            dc_id                             118470523
            name             Abhainn Roag at Mill Croft
            river                          Abhainn Roag
            location                         Mill Croft
            xout                                75500.0
            yout                               836150.0
            basin_area                          52.5625"""
        event = parse_event_string(event_string)
        
        river_obj, rid, date_range, event = sample_event(train_events, vwc_quantiles, event=event)
        
        for tstep in range(3):
            river_obj, loss_tstep, loss_opt, opt_counter = model.forward(
                river_obj, date_range, tstep, device, nt_opt=nt_opt,
                teacher_forcing=teacher_forcing, train=True
            )
            if loss_tstep is not None:
                running_ls.append(loss_tstep)
            if opt_counter==0:
                if loss_opt is not None:
                    opt_ls.append(loss_opt)
        
        
        
        # plot flows and NRFA stations
        river_obj.plot_flows(river_obj.flow_est, date_range[1], scaler=1.8, add_min=0.1)
        
        # plot catchments on a 1km grid
        import xarray as xr        
        xygrid = xr.load_dataset(hj_base + "/ancillaries/chess_landfrac.nc",
                                 decode_coords="all")
        elev_map = xr.load_dataset(hj_base + "/ancillaries/uk_ihdtm_topography+topoindex_1km.nc",
                                   decode_coords="all")
        xygrid['elev'] = (("y", "x"), elev_map.elev.data)
        xygrid.elev.values[np.isnan(xygrid.elev.values)] = 0
        
        river_obj.plot_river(grid=xygrid.elev, stations=False)
    
    
    
    
        ############################
        ## generate pre-primed event objects
        # rid == 66978 has lots of stations in!
        train_events.sort_values('basin_area')
        outdir = '/home/users/doran/data_dump/catchment_reaches/event_data/'
        for i in range(50):
            print(i)
            river_obj, rid, date_range, event = sample_event(train_events, vwc_quantiles)
            
            savedir = outdir + f'/{rid}_{i}/'
            Path(savedir).mkdir(exist_ok = True, parents = True)

            river_obj.save_event_data(event, savedir)
            
            
            
        # # then to load an event:
        rid = int(savedir.split('/')[-2].split('_')[0])
        river_obj = River()
        river_obj.load(save_rivers_dir, rid)
        river_obj.load_event_data(savedir)
