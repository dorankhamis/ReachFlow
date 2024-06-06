import numpy as np
import torch
import shutil
import rioxarray
from pathlib import Path
from matplotlib.path import Path as pltPath
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection  

#######################
## utility functions ##
#######################

def zeropad_strint(integer):
    if integer<10:
        return '0' + str(integer)
    else:
        return str(integer)

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def remove_prefix(input_string, prefix):
    if prefix and input_string.startswith(prefix):
        return input_string[len(prefix):]
    return input_string
    
def normalise_df_simplex_subset(df, col_pattern):
    norm_cols = [col for col in df if col.startswith(col_pattern)]
    clipped_vals = df[norm_cols].clip(lower=0)
    df.loc[:, norm_cols] = clipped_vals.values / clipped_vals.sum(axis=1).values[...,None]
    return df

########################
## plotting functions ##
########################

def plot_polygon(ax, poly, **kwargs):
    path = pltPath.make_compound_path(
        pltPath(np.asarray(poly.exterior.coords)[:, :2]),
        *[pltPath(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

####################################
## Soil moisture netcdf functions ##
####################################

def calculate_soil_wetness(soil_moisture, vwc_quantiles, incl_time=True):
    if not incl_time:        
        soil_moisture['soil_wetness'] = (('y', 'x'),
            (soil_moisture.theta.values - vwc_quantiles.vwc_l95.values[0,:,:])/(vwc_quantiles.vwc_u95.values[0,:,:] - vwc_quantiles.vwc_l95.values[0,:,:])
        )
    else:
        soil_moisture['soil_wetness'] = (('time', 'y', 'x'),
            (soil_moisture.theta.values - vwc_quantiles.vwc_l95.values[0,:,:])/(vwc_quantiles.vwc_u95.values[0,:,:] - vwc_quantiles.vwc_l95.values[0,:,:])
        )
    soil_moisture['soil_wetness'] = soil_moisture.soil_wetness.clip(min=0, max=1)
    return soil_moisture

def trim_netcdf(nc_dat, XY, var_name, the_dims, t_pos=None, y_pos=1, x_pos=2):
    if len(the_dims)>2:
        if t_pos==0 and y_pos==1 and x_pos==2:
            nc_dat['trimmed_'+var_name].values[:,XY[1],XY[0]] = nc_dat[var_name].values[:,XY[1],XY[0]]
        elif t_pos==0 and y_pos==2 and x_pos==1:
            nc_dat['trimmed_'+var_name].values[:,XY[0],XY[1]] = nc_dat[var_name].values[:,XY[0],XY[1]]
        if t_pos==2 and y_pos==0 and x_pos==1:
            nc_dat['trimmed_'+var_name].values[XY[1],XY[0],:] = nc_dat[var_name].values[XY[1],XY[0],:]
        elif t_pos==2 and y_pos==1 and x_pos==0:
            nc_dat['trimmed_'+var_name].values[XY[0],XY[1],:] = nc_dat[var_name].values[XY[0],XY[1],:]
    else:
        nc_dat['trimmed_'+var_name].values[XY[1],XY[0]] = nc_dat[var_name].values[XY[1],XY[0]]           
    return nc_dat

def merge_two_soil_moisture_days(this_sm_obj, this_day, prev_day,
                                 sm_data_dir, x_inds, y_inds):
    # calculate the day weighting from hours since midnight        
    this_day_weight = (this_day.hour + this_day.minute/60)/24

    if not ((this_day.year==prev_day.year) and (this_day.month==prev_day.month)):
        prev_sm = rioxarray.open_rasterio(
            sm_data_dir + f'/{prev_day.year}/SM_{prev_day.year}{zeropad_strint(prev_day.month)}.nc'
        )
        prev_sm = prev_sm.isel(time=prev_day.day-1) # zero-indexed so -1 from day
        this_sm = this_sm_obj.isel(time=this_day.day-1) # zero-indexed so -1 from day
    else:
        prev_sm = this_sm_obj.isel(time=prev_day.day-1) # zero-indexed so -1 from day
        this_sm = this_sm_obj.isel(time=this_day.day-1) # zero-indexed so -1 from day

    this_sm_trim = this_sm.isel(x=x_inds, y=y_inds)
    prev_sm_trim = prev_sm.isel(x=x_inds, y=y_inds)
    this_sm_trim['theta'] = (this_sm_trim.theta * this_day_weight + 
        prev_sm_trim.theta * (1 - this_day_weight))
    
    return this_sm_trim

#############################
## Checkpointing functions ##
#############################

def setup_checkpoint(model, device, load_prev_chkpnt,
                     model_outdir, log_dir, specify_chkpnt=None,
                     reset_chkpnt=False):
    if load_prev_chkpnt:
        if specify_chkpnt is None:
            loadmodel = model_outdir + 'best_model.pth'
            print('Loading best checkpoint...')
        else:
            ## to load different weights to begin        
            # specify_chkpnt of form "modelname/checkpoint.pth" or 
            # "OTHERMODEL/best_model.pth" or "OTHERMODEL/checkpoint.pth"
            loadmodel = f'{log_dir}/{specify_chkpnt}'
            print(f'Loading {log_dir}/{specify_chkpnt}...')
        try:
            model, checkpoint = load_checkpoint(loadmodel, model, device)
            print('Loaded checkpoint successfully')
            print(f'Best loss: {checkpoint["best_loss"]}')
            print(f'current epoch: {checkpoint["epoch"]}')
        except:
            print('Failed loading checkpoint')
            checkpoint = None
    else: 
      checkpoint = None
      loadmodel = None

    if reset_chkpnt is True:        
        checkpoint = None # adding this to reset best loss and loss trajectory
    
    return model, checkpoint
    
def save_checkpoint(state, is_best, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    f_path = outdir + '/checkpoint.pth'
    torch.save(state, f_path)
    if is_best:
        print ("=> Saving a new best")
        best_fpath = outdir + '/best_model.pth'
        shutil.copyfile(f_path, best_fpath)
    else:
        print ("=> Validation loss did not improve")

def load_checkpoint(checkpoint_fpath, model, device):
    if device.type=='cpu':
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint.pop('state_dict'))
    if not model.optimizer is None:
        model.optimizer.load_state_dict(checkpoint.pop('optimizer'))  
    return model, checkpoint

def update_checkpoint(epoch, model, best_loss, losses, val_losses):
    return {'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'best_loss': best_loss,
            'torch_random_state': torch.random.get_rng_state(),
            'numpy_random_state': np.random.get_state(),
            'losses': losses,
            'val_losses': val_losses}

def prepare_run(checkpoint):
    if checkpoint is None:
        curr_epoch = 0
        best_loss = np.inf
        losses = []
        val_losses = []
    else:
        curr_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        torch.random.set_rng_state(checkpoint['torch_random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        try:
            losses = checkpoint['losses']
            val_losses = checkpoint['val_losses']
        except:
            losses = []
            val_losses = []
    return losses, val_losses, curr_epoch, best_loss

def send_batch_to_device(batch, device):
    for k in batch.keys():
        if type(batch[k])==torch.Tensor:
            batch[k] = batch[k].to(device)
    return batch
