import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions import kl

import numpy as np
import pandas as pd

import pkbar

from torch.special import erf as torch_erf
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from utils import save_checkpoint, prepare_run, update_checkpoint

def get_event(file_df, sample_type='train'):
    got_bad_sample = True
    while got_bad_sample:
        this_file = file_df.sample(n=1, weights='p', axis=0)
        dat = np.load(this_file.file.values[0])
        if sample_type=='train':
            keys = list(dat)[:this_file.train_val_ind_split.values[0]]
        elif sample_type=='val':
            keys = list(dat)[this_file.train_val_ind_split.values[0]:]
        this_key = np.random.choice(keys)
        sample = dat[this_key]
        got_bad_sample = np.any(np.isnan(sample))
    return sample   

def normalise_sample(sample, names, norm_df):
    sample = pd.DataFrame(sample, columns=names)

    #flow = sample.FLOW.values
    #precip = sample.PRECIP.values
    # here precip is in mm * km^2, multiply by 1e6 and divide by 
    # number of seconds in 15 mins to get m^3/s to match flow units?? too large surely

    overlap_names = [n for n in names if n in norm_df.columns]
    normalised_cols = sample[overlap_names].values / norm_df[overlap_names].values
    sample = pd.concat([
        pd.DataFrame(normalised_cols, columns=overlap_names),
        sample[[n for n in names if n not in norm_df.columns]]
        ], axis=1
    )[names]
    return sample

def get_batch(file_df, names, norm_df, device, sample_type='train'):
    sample = get_event(file_df, sample_type=sample_type)
    sample = normalise_sample(sample, names, norm_df)

    flow_true = (torch.from_numpy(sample.FLOW.values)
        .to(torch.float32)
        .unsqueeze(0).unsqueeze(2)
        .to(device)
    )
    inputs = (torch.from_numpy(sample[[n for n in names if n!='FLOW']].values)
        .to(torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    precip = (torch.from_numpy(sample.PRECIP.values)
        .to(torch.float32)
        .unsqueeze(0).unsqueeze(2)
        .to(device)
    )        
    return {'inputs':inputs, 'flow_true':flow_true, 'precip':precip}

def skew_pdf(x, xi, omega, alpha, rho, dist='normal'):
    """ x: (Reals), xi:location (Reals), omega:spread (Reals+),
        alpha:skew (Reals+), rho:multiplicative scale (Reals+) """
    if dist=='normal':
        sqrt2 = np.sqrt(2)
        sqrtpi = np.sqrt(np.pi)
        norm_term = 1. / (omega * sqrt2 * sqrtpi)    
        exp_term = torch.exp(-0.5 * torch.square((x - xi)/omega))
        skew_term = 1 + torch_erf(alpha * ((x - xi) / (sqrt2 * omega)))
    elif dist=='laplace':
        norm_term = 1. / (omega * (alpha + 1./(alpha + 1e-12)))
        s = torch.sign(x - xi)
        exp_term = torch.exp(-(x - xi)/omega * s * torch.pow(alpha + 1e-12, s))    
        skew_term = 1        
    return rho * norm_term * exp_term * skew_term

def generate_single_flow_contribution(precip, xi, omega, alpha, rho, i,
                                      x=None, zeros=None, dist='normal'):
    if x is None:
        x = (torch.from_numpy(np.arange(xi.shape[1]))
            .unsqueeze(0)
            .to(torch.float32)
            .to(xi.device)
        )
    if zeros is None:
         zeros = torch.zeros_like(x).to(torch.float32).to(xi.device)
        
    this_flow_contrib = torch.cat([zeros[:, :i], skew_pdf(
        x[:, i:] - i, # "recentre" at source token for skew distribution calculation
        xi[:, i:(i+1)],
        omega[:, i:(i+1)],
        alpha[:, i:(i+1)],
        rho[:, i:(i+1)],
        dist=dist)], dim = 1
    )
    return precip[:, i:(i+1)] * this_flow_contrib
    
def generate_flow_prediction(precip, xi, omega, alpha, rho,
                             Bf, dist='laplace'):
    x = (torch.from_numpy(np.arange(xi.shape[1]))
        .unsqueeze(0)
        .to(torch.float32)
        .to(xi.device)
    )
    zeros = torch.zeros_like(x).to(torch.float32).to(xi.device)    
    base_flow = Bf
    
    # must be a way of doing this without a loop?
    for i in range(x.shape[1]):
        this_flow_contrib = generate_single_flow_contribution(
            precip, xi, omega, alpha, rho,
            i, x=x, zeros=zeros, dist=dist
        )        
        if i==0:
            flow_pred = this_flow_contrib
        else:
            flow_pred = flow_pred + this_flow_contrib
    
    return flow_pred + base_flow

class NegMSELoss:
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def lprob(self, y_pred, y_true):
        where_finite = torch.isfinite(y_true)
        return -self.mse_loss(y_pred[where_finite], y_true[where_finite])

@torch.no_grad()
def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        y = m.in_features        
        m.weight.data.normal_(0.0, 1/np.sqrt(y))        
        m.bias.data.fill_(0)

class FeedForward(nn.Module):
    """ Implements FFN equation """
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation=='relu': self.a = F.relu
        elif activation=='gelu': self.a = F.gelu
        elif activation=='lrelu': self.a = F.leaky_relu()
        self.apply(weights_init_normal)

    def forward(self, x):
        return self.w_2(self.dropout(self.a(self.w_1(x))))

class Encoder(nn.Module):    
    def __init__(self, cfg, d1_out):
        super(Encoder, self).__init__()
        d_in = cfg.d_in
        d_model = cfg.d_model
        d_ff = cfg.d_ff
        dropout = cfg.dropout        
        
        self.embed_inputs = nn.Linear(d_in, d_model)
        self.ff_network = FeedForward(d_model, d_ff, dropout=dropout, activation='relu')
        self.predict_skewdist_param = nn.Linear(d_model, d1_out)
        self.predict_baseflow_params1 = nn.Linear(d_model, 1)
    
        self.apply(weights_init_normal)

    def forward(self, x):
        x_embed = self.embed_inputs(x)
        x_embed = self.ff_network(x_embed)
        skew_dist_params = self.predict_skewdist_param(x_embed)

        x_pool_b = F.avg_pool1d(x_embed.transpose(-1,-2), x.shape[1]).transpose(-1,-2)
        base_flow_const_params = self.predict_baseflow_params1(x_pool_b)
        return skew_dist_params, base_flow_const_params

class FlowAE(nn.Module):
    def __init__(self, cfg):
        super(FlowAE, self).__init__()
        self.encoder = Encoder(cfg, 4)
        # 4 dims are value of skew dist params:
        #    (xi (delay), omega (spread), alpha (skew), rho (magnitude))

    def forward(self, batch):
        pmap_vae_params, bf_const_vae_params = self.encoder(batch['inputs'])
            
        xi = pmap_vae_params[:,:,0].exp()
        omega = pmap_vae_params[:,:,1].exp()
        alpha = pmap_vae_params[:,:,2].exp()
        rho = pmap_vae_params[:,:,3].exp()
        Bf = bf_const_vae_params[:,:,0].exp()    
        flow_pred = generate_flow_prediction(
            batch['precip'][:,:,0],
            xi, omega, alpha, rho, Bf,
            dist='laplace'
        )
        log_lik = NegMSELoss().lprob(flow_pred.unsqueeze(2), batch['flow_true']).sum()
        loss = -log_lik
        return (loss, log_lik, flow_pred, 
                {'xi':xi, 'omega':omega, 'alpha':alpha, 'rho':rho, 'Bf':Bf})

def make_train_step(model, optimizer):
    def train_step(batch):
        model.train()        
        loss, log_lik, flow_pred, flow_params = model(batch)     
        # propagate derivatives
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()        
        return {'loss':loss.item(), 'log_lik':log_lik.item()}
    return train_step

def make_val_step(model):
    def val_step(batch):
        model.eval()
        with torch.no_grad():            
            loss, log_lik, flow_pred, flow_params = model(batch)
        return {'loss':loss.item(), 'log_lik':log_lik.item()}
    return val_step

def update_running_loss(running_ls, loss_dict):
    if running_ls is None:
        running_ls = {k:[loss_dict[k]] for k in loss_dict}
    else:
        for loss_key in running_ls:
            running_ls[loss_key].append(loss_dict[loss_key])
    return running_ls

def fit(model, optimizer, cfg,
        train_df, val_df, names, norm_df,
        LR_scheduler=None, outdir='/logs/',
        checkpoint=None, device=None):
    
    ## setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_best = False
    train_step = make_train_step(model, optimizer)
    val_step = make_val_step(model)
    losses, val_losses, curr_epoch, best_loss = prepare_run(checkpoint)
           
    ## train
    for epoch in range(curr_epoch, cfg.max_epochs):
        model.train()
        kbar = pkbar.Kbar(target=cfg.train_len, epoch=epoch,
                          num_epochs=cfg.max_epochs,
                          width=15, always_stateful=False)        
        running_ls = None
        for bidx in range(1, cfg.train_len+1):
            batch = get_batch(train_df, names, norm_df, device, sample_type='train')
            loss_dict = train_step(batch)
            if np.isnan(loss_dict['loss']):
                return model, batch
            running_ls = update_running_loss(running_ls, loss_dict)            
            print_values = [(key, running_ls[key][-1]) for key in running_ls]
            kbar.update(bidx, values=print_values)
        losses.append(np.mean(running_ls['loss'])) # append epoch average loss
        
        # save newest checkpoint before doing validation
        checkpoint_interim = update_checkpoint(epoch, model, optimizer,
                                               best_loss, losses, [])
        save_checkpoint(checkpoint_interim, False, outdir)
        
        if (not LR_scheduler is None) and (epoch>5):
            LR_scheduler.step()
        
        # validation
        with torch.no_grad():
            kbarv = pkbar.Kbar(target=cfg.val_len, epoch=epoch,
                               num_epochs=cfg.max_epochs,
                               width=15, always_stateful=False)
            model.eval()
            running_ls = None
            for bidx in range(1, cfg.val_len+1):
                batch = get_batch(val_df, names, norm_df, device, sample_type='val')
                loss_dict = val_step(batch)
                if np.isnan(loss_dict['loss']):
                    return model, batch
                running_ls = update_running_loss(running_ls, loss_dict)            
                print_values = [(key, running_ls[key][-1]) for key in running_ls]
                kbarv.update(bidx, values=print_values)                
            val_losses.append(np.mean(running_ls['loss']))
            kbarv.add(1, values=[("val_nll", val_losses[-1])])
            
            is_best = bool(val_losses[-1] < best_loss) if not np.isnan(val_losses[-1]) else False
            best_loss = min(val_losses[-1], best_loss) if not np.isnan(val_losses[-1]) else best_loss
            checkpoint = update_checkpoint(epoch, model, optimizer,
                                           best_loss, losses, val_losses)
            save_checkpoint(checkpoint, is_best, outdir)
        print("Done epoch %d" % (epoch+1))
    return model, losses, val_losses
