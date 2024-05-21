import numpy as np
import pandas as pd
import geopandas as gpd
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

cfg = SimpleNamespace(
    # model params
    dropout = 0.1,
    d_model = 16,
    d_ff = 32,
    N_h = 2,
    N_l = 2,
    d_s_in = 31,
    d_ls_src = 37,
    d_ls_trg = 36,    
    d_instream_src = 10,
    d_instream_trg = 8,
    #d_fh_src = 7,
    #d_fh_trg = 6
    
    # training params
    max_batch_size = 10,
    lr = 1e-4,
    max_epochs = 100,
    train_len = 50,
    val_len = 25
)

def sample_event(flood_event_df, vwc_quantiles=None):
    event = flood_event_df.sample(1).iloc[0]
    rid = event.id
    date_range = pd.date_range(start=event.FlowStartDate, end=event.FlowEndDate, freq='15min')
    river_obj = load_event_data(rid, date_range, vwc_quantiles=vwc_quantiles)
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
            for tstep in range(len(date_range)):
                river_obj, loss_tstep, loss_opt, opt_counter = model.forward(
                    river_obj, date_range, tstep, device, nt_opt=nt_opt,
                    teacher_forcing=teacher_forcing, train=True
                )
                running_ls.append(loss_tstep)
                if opt_counter==0:
                    opt_ls.append(loss_opt)
            
            print_values = [('loss_tstep', running_ls[-1]), ('opt_loss', opt_ls[-1])]
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
                for tstep in range(len(date_range)):
                    river_obj, loss_tstep, _, _ = model.forward(
                        river_obj, date_range, tstep, device, nt_opt=nt_opt,
                        teacher_forcing=teacher_forcing, train=False
                    )
                    running_ls.append(loss_tstep)                    
                
                print_values = [('loss', running_ls[-1])]
                kbarv.update(bidx, values=print_values)
                
            val_losses.append(np.mean(running_ls))
            kbar.add(1, values=[("val_loss", val_losses[-1])])
            
            is_best = bool(val_losses[-1] < best_loss)
            best_loss = min(val_losses[-1], best_loss)
            checkpoint = update_checkpoint(epoch, model, optimizer,
                                           best_loss, losses, val_losses)
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
    upper_bound = 28**2 * 2 - 1 # 30*24*4 # one month
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
