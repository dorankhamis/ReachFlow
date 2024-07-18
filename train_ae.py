import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray
import time
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.special import erf as torch_erf
from pathlib import Path
from types import SimpleNamespace
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from river_reach_modelling.event_wrangling import load_statistics
from river_reach_modelling.river_class import (
    River, load_event_data, sm_data_dir, hj_base, save_rivers_dir
)
from river_reach_modelling.utils import zeropad_strint, setup_checkpoint
from river_reach_modelling.funcs_precip_normed_event_features import grab_names
from river_reach_modelling.architectures.autoencoder import FlowAE, fit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rseed = 42
np.random.seed(rseed)

cfg = SimpleNamespace(
    # model params
    d_in = 31,
    d_model = 64,
    d_ff = 128,
    dropout = 0.15, # this might be making it hard to train    
        
    # training params    
    lr = 1e-4,
    lrs_gamma = 0.995,
    max_epochs = 1000,
    train_len = 2000,
    val_len = 1000
)

norm_dict = dict(
    QDPS = 100,
    ICAR = 25,
    REACH_LENGTH = 10000,
    REACH_SLOPE = 5,
    CCAR = 100,
    QDPL = 1000,
    DRAINAGE_DENSITY = 2,
    PRECIP = 500,
    FLOW = 50
)
norm_df = pd.DataFrame(norm_dict, index=[0])

'''
    QDPS = 100, # [0.022517, 738.293548], med = 78.3
    ICAR = 25, # [0.002500, 264.972500], med = 2.725000
    REACH_LENGTH = 1000, # [50 , 44808.178182], med = 1257.106781
    REACH_SLOPE = 5, # [0, 34.779979], med = 0.527106
    CCAR = 100, # [3.00000, 9971.32250], med = 12.28125
    QDPL = 1000, # [7.071068, 18064.327412], med = 1519.811097
    DRAINAGE_DENSITY = 2, # [0.002611, 28.284271], med = 0.943428
    PRECIP = 100,
    FLOW = 50
'''



if __name__=="__main__":
    # set file paths and model tags
    event_dat_path = hj_base + '/flood_events/event_encoding/'
    log_dir = './logs/'
    model_name = f'precip_map_ae_constBf'    
    model_outdir = f'{log_dir}/{model_name}/'
    Path(model_outdir).mkdir(parents=True, exist_ok=True)
    
    load_prev_chkpnt = True
    reset_chkpnt = False
    #specify_chkpnt = None #f'{model_name}/checkpoint.pth' # if None, load best, otherwise "modelname/checkpoint.pth"        
    specify_chkpnt = f'{model_name}/checkpoint.pth'


    # get feature order from grab_names()
    names_path = event_dat_path + '/feature_names.npy'
    nevent = None
    if not Path(names_path).exists():
        flood_event_df = load_statistics()
        gauge_river_map = pd.read_csv(hj_base + '/flood_events/station_to_river_map.csv')
        flood_event_df = (flood_event_df.rename({'Station':'nrfa_id'}, axis=1)
            [['nrfa_id', 'Event', 'FlowStartDate', 'FlowEndDate']]
            .merge(gauge_river_map, on='nrfa_id')
            .dropna()
        )
        flood_event_df = flood_event_df.sort_values(['basin_area','id'])
        nevent = flood_event_df.iloc[0]
    names = grab_names(names_path, event=nevent)


    # split into train and validation set
    catchment_file_list = glob.glob(event_dat_path + '*.npz')
    n_events_per_file = [len(np.load(ff)) for ff in catchment_file_list]
    event_df = pd.DataFrame({'file':catchment_file_list, 'n_events':n_events_per_file})
    event_df = event_df.query('n_events > 0')    
    
    fit_frac = 0.85
    train_frac = 0.75
    unseen_frac = 0.05
    unseen_df = event_df.sample(frac=unseen_frac, random_state=rseed)
    event_df = event_df[~event_df.index.isin(unseen_df.index)]
    train_df = event_df.sample(frac=fit_frac, random_state=rseed)
    train_df = train_df.assign(train_val_ind_split = lambda x: (x.n_events * train_frac).astype(np.int32))
    val_df = event_df[~event_df.index.isin(train_df.index)].assign(train_val_ind_split = 0)
    val_df = pd.concat([train_df, val_df], axis=0)
        
    train_df = (train_df
        .assign(log_p = lambda x: np.log(x.train_val_ind_split) / np.log(x.train_val_ind_split).sum())
        .assign(p = lambda x: x.log_p / x.log_p.sum())
    )
    val_df = (val_df
        .assign(log_p = lambda x: np.log(x.n_events - x.train_val_ind_split) / np.log(x.n_events - x.train_val_ind_split).sum())
        .assign(p = lambda x: x.log_p / x.log_p.sum())
    )
    
    ## create model, optimizer, lr_scheduler
    model = FlowAE(cfg)
    model = model.to(device)
    print("Number of trainable parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    LR_scheduler = ExponentialLR(optimizer, gamma=cfg.lrs_gamma)

    model, optimizer, checkpoint = setup_checkpoint(
        model, optimizer, device, load_prev_chkpnt,
        model_outdir, log_dir,
        specify_chkpnt=specify_chkpnt,
        reset_chkpnt=reset_chkpnt
    )
    

    out = fit(model, optimizer, cfg, train_df, val_df, names, norm_df,                             
              LR_scheduler=LR_scheduler, outdir=model_outdir,
              checkpoint=checkpoint, device=device)


    if False:                
        #checkpoint = torch.load(model_outdir + 'best_model.pth', map_location=torch.device('cpu'))
        plt.plot(checkpoint['losses'])
        plt.plot(checkpoint['val_losses'])
        plt.show()
        
        # test
        from autoencoder import get_batch, generate_flow_prediction
        model.eval()
        batch = get_batch(train_df, names, norm_df, device, sample_type='train')
        
        with torch.no_grad():
            loss, log_lik, flow_pred, flow_params = model(batch)

        plt.plot(batch['flow_true'].detach().numpy()[0,:,0])
        plt.plot(flow_pred.detach().numpy()[0,:])
        plt.plot(batch['precip'].detach().numpy()[0,:,0])
        plt.show()

    '''
    TODO: 
        - add extra loss component on MAX flow to incentivise capturing the
            rarer peak flow values
        - perhaps split into "local catchment rain tokens" and
            "cumulative catchment timeseries", where the base flow aspect
            is from the cumulative catchment timeseries?
        - or even just give the initial flow value as base flow and 
            don't infer it? just to focus on the response to precip?
        - write test code to examine effect on flow / params of sweeping
            across input variable ranges
        - change/check definition of some of the derived catchment descriptors:
            drainage path length and drainage path slope, how do we average 
            these across the wetted area? what does ICAR really mean?
            Are we just adding noise to the system with some of the quantities?
    '''
    param_ranges = dict(
        
    )
    
    from autoencoder import get_batch, generate_flow_prediction
    model.eval()
    # grab representative token as baseline    
    batch = get_batch(train_df, names, norm_df, device, sample_type='train')
    ind = torch.argmax(batch['inputs'][0,:,0])
    base_token = batch['inputs'][:,ind:(ind+1),:].clone()
    ts_length = 250
    zeros = torch.zeros((1, ts_length-1, base_token.shape[2])).to(torch.float32)
    batch['flow_true'] = torch.zeros((1, ts_length, 1)).to(torch.float32)
    num_par_vals = 50
    #for nn in names[2:]:
        
    nn = 'LC_01'
    n_ind = np.where(names[1:]==nn)[0][0]
    par_values = np.linspace(0, 1, num_par_vals)
    flows_out = np.ones((num_par_vals, ts_length), dtype=np.float32) * np.nan
    params_out = np.ones((num_par_vals, 4), dtype=np.float32) * np.nan
    for ii, val in enumerate(par_values):
        token = base_token.clone()
        token[:,:,n_ind] = val
        
        # renorm unity sums?
        # e.g. sinds = [np.where(names==nn)[0][0] for nn in [n for n in names if n.startswith('HGB')]]
        
        to_run = torch.cat([token, zeros], dim = 1)
        batch['inputs'] = to_run
        batch['precip'] = to_run[:,:,0:1]
        with torch.no_grad():
            loss, log_lik, flow_pred, flow_params = model(batch)
            flow_pred = flow_pred - flow_params['Bf']
        flows_out[ii,:] = flow_pred.detach().numpy().squeeze()
        params_out[ii,0] = flow_params['xi'][0,0]
        params_out[ii,1] = flow_params['omega'][0,0]
        params_out[ii,2] = flow_params['alpha'][0,0]
        params_out[ii,3] = flow_params['rho'][0,0]
    
    fig, axes = plt.subplots(2, 2, sharex=True)
    axes[0,0].plot(par_values, params_out[:,0])
    axes[0,0].set_title('Delay')
    axes[0,1].plot(par_values, params_out[:,1])
    axes[0,1].set_title('Spread')
    axes[1,0].plot(par_values, params_out[:,2])
    axes[1,0].set_title('Skew')
    axes[1,0].set_xlabel(nn)
    axes[1,1].plot(par_values, params_out[:,3])
    axes[1,1].set_title('Multiplicative scale')
    axes[1,1].set_xlabel(nn)
    
    fig2, ax2 = plt.subplots()
    ax2.plot(flows_out[0,:], 'k')
    ax2.plot(flows_out[num_par_vals//4,:], 'b')
    ax2.plot(flows_out[num_par_vals//2,:], 'g')
    ax2.plot(flows_out[3*num_par_vals//4,:], 'orange')
    ax2.plot(flows_out[-1,:], 'r')
    plt.show()

    ## should look at magnitude of change in delay, spread etc  
    ## across parameters as way to compare their effects? 
    
    '''
    New model structure idea:
    for each subcatchment, calculate a precipitation response curve
    from the incremental static descriptors + local soil moisture 
    and then multiply by the local precipitation and sum over
    all sub catchments.
    gains:
        -- we don't have to calculate precip-weighted tokens which 
            are a bit dodgy
        -- we can potentially save computation by calculating baseline
            precip responses only once per subcatchment?
            (though soil moisture and precip are dynamic)
    
    '''

