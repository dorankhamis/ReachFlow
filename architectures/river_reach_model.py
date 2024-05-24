import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from contextlib import nullcontext

from architectures.components.mlp import PositionwiseFeedForward
from architectures.components.attention import MultiHeadedAttention
from architectures.components.nn_utils import subsequent_mask, clones, weights_init_normal

def gather_params(model):    
    params = []
    with torch.no_grad():
        for param in model.parameters():
            params.append(param.view(-1))
        params = torch.cat(params)
    return params

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.apply(weights_init_normal)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class AttentionBlock(nn.Module):    
    def __init__(self, layer, N):
        super(AttentionBlock, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        self.apply(weights_init_normal)
        
    def forward(self, x, y=None, mask=None):        
        for layer in self.layers:
            if y is None:
                x = layer(x, mask) # self-attention
            else:
                x = layer(x, y, mask) # cross-attention
        return self.norm(x)


class EncoderLayer(nn.Module):    
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, cross_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()        
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)    
        self.size = size    
        
    def forward(self, x, enc_output, src_mask):        
        x = self.sublayer[0](x, lambda x: self.cross_attn(x, enc_output, enc_output, src_mask))
        return self.sublayer[1](x, self.feed_forward)


class Transformer(nn.Module):    
    def __init__(self, d_in, d_trg, d_model, d_ff, dropout, N_h, N_l):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        
        # embedding
        self.embed_src = nn.Linear(d_in, d_model)
        self.embed_trg = nn.Linear(d_trg, d_model)
        self.act = nn.GELU()        
        
        # encoder
        attn = MultiHeadedAttention(N_h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.self_att_blck = AttentionBlock(EncoderLayer(d_model, c(attn), c(ff), dropout), N_l)
        
        # decoder
        xattn = MultiHeadedAttention(N_h, d_model)
        xff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.cross_att_blck = AttentionBlock(DecoderLayer(d_model, c(xattn), c(xff), dropout), N_l)
        
        self.apply(weights_init_normal)
        
    def forward(self, x_src, x_trg, mask=None):
        xtrg_decoded = self.act(self.embed_trg(x_trg))
        
        if x_src.shape[1]>0:
            # i.e. not the initial time point
            xsrc_encoded = self.act(self.embed_src(x_src))
            
            if x_src.shape[1]>1:
                # i.e. not the first or second time point
                xsrc_encoded = self.self_att_blck(xsrc_encoded, mask=mask)
            
            xtrg_decoded = self.cross_att_blck(xtrg_decoded, y=xsrc_encoded, mask=mask)
        
        return xtrg_decoded


class LinearDecoder(nn.Module):
    def __init__(self, d_model, d_out, final_relu=True):
        super(LinearDecoder, self).__init__()        
        self.decode = nn.Linear(d_model, d_out)        
        self.final_relu = final_relu
        self.relu = nn.ReLU()
        self.apply(weights_init_normal)

    def forward(self, x):
        if self.final_relu:
            return self.relu(self.decode(x))
        else:
            return self.decode(x)
        
        
class MLPPredictor(nn.Module):
    def __init__(self, d_input, d_hidden, d_out, dropout=0.1, final_relu=True):
        super(MLPPredictor, self).__init__()
        self.act = nn.GELU()        
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Linear(d_input, d_hidden)
        self.decoder = LinearDecoder(d_hidden, d_out, final_relu=final_relu)        
        self.apply(weights_init_normal)

    def forward(self, inputs):
        x = self.act(self.dropout(self.encoder(inputs)))
        return self.decoder(x)


class ReachFlow(nn.Module):
    def __init__(self, cfg):
        super(ReachFlow, self).__init__()
        self.s_model = MLPPredictor(cfg.d_s_in, cfg.d_model, 1, dropout=cfg.dropout, final_relu=True)
        self.ls_model = Transformer(cfg.d_ls_src, cfg.d_ls_trg, cfg.d_model, cfg.d_ff, cfg.dropout, cfg.N_h, cfg.N_l)
        self.l_from_ls_model = LinearDecoder(cfg.d_model, 1, final_relu=True)
        self.s_from_ls_model = LinearDecoder(cfg.d_model, 1, final_relu=True)
        self.fstream_model = Transformer(cfg.d_instream_src, cfg.d_instream_trg, cfg.d_model, cfg.d_ff, cfg.dropout, cfg.N_h, cfg.N_l)
        #self.fh_model = Transformer(cfg.d_fh_src, cfg.d_fh_trg, cfg.d_model, cfg.d_ff, cfg.dropout, cfg.N_h, cfg.N_l)
        self.latent2flow_model = LinearDecoder(cfg.d_model, 1, final_relu=True)
        
        self.max_batch_size = cfg.max_batch_size
        self.optimizer = None
        self.loss_opt = None
        self.opt_counter = 0
    
    def setup_optimizer(self, cfg):
        self.optimizer = Adam(self.parameters(), lr=cfg.lr)

    def calculate_loss(self, river_obj, tsteps):
        for i, nid in enumerate(river_obj.flow_data.keys()):
            gauge = river_obj.points_in_catchments.query('nrfa_id == {0}'.format(nid))
            downstream_id = int(gauge.id)
            dnorm = float(gauge.norm_dist_upstream)
            
            upstream_cells = (river_obj.network[(river_obj.network.dwn_id==downstream_id)]
                .query('tidal==False')
            )
            Fup = sum([river_obj.F_t[k][:,tsteps,:] for k in upstream_cells.id])
            Fdown = river_obj.F_t[downstream_id][:,tsteps,:]
            F_obs = torch.from_numpy(
                river_obj.flow_data[nid].iloc[tsteps].values
            ).to(torch.float32).unsqueeze(0).to(Fdown.device)
            ''' check for NAs in obs and mask out for loss '''
            if i==0:
                loss = F.mse_loss(dnorm*Fup + (1-dnorm)*Fdown, F_obs)
            else:
                loss = loss + F.mse_loss(dnorm*Fup + (1-dnorm)*Fdown, F_obs)
        return loss
    
    def opt_step(self):               
        # propagate derivatives
        self.loss_opt = self.loss_opt / self.opt_counter
        self.loss_opt.backward()
        self.optimizer.step()
        
        # reset
        self.optimizer.zero_grad()        
        self.opt_counter = 0        

    def forward(self, river_obj, date_range, tstep, device,
                nt_opt=1, teacher_forcing=False, train=False):
        if train:
            self.train()
        else:
            self.eval()
        
        # perform a single forward-pass time tick across all reaches
        if tstep==0:
            ## do initial one-off things
            river_obj = self.prepare_inputs(river_obj, device, teacher_forcing=teacher_forcing)
            
            ## run forward initial timepoint
            river_obj = self.batched_initial_lateral_inflow_and_storage(river_obj, train=train)                
            for rid in river_obj.sorted_ids:
                # prepare initial flows
                river_obj.F_t[rid] = river_obj.flow_teacherforcing[rid][:,:1,:]
                river_obj.Fu_t[rid] = self.zeros_vec.clone().to(device)
                river_obj.Fl_t[rid] = self.zeros_vec.clone().to(device)
                
        else:
            ## do normal forward algorithm parts            
            # in-flow from land and storage state of catchment
            river_obj = self.batched_lateral_inflow_and_storage(river_obj, date_range, tstep, train=train)
            
            # contribution to flow from land at drainage cell
            river_obj = self.batched_terrestrial_flow_contribution(river_obj, tstep, train=train)
            
            for rid in river_obj.network.sort_values('reach_level', ascending=False).id:                
                # contribution to flow from upstream at drainage cell                
                F_u = self.upstream_flow_contribution(
                    river_obj,
                    rid,
                    tstep,
                    teacher_forcing=teacher_forcing,
                    train=train
                )
                river_obj.Fu_t[rid] = torch.cat([river_obj.Fu_t[rid], F_u], dim=1)
                
                # attenuation/hysteresis of flow at drainage cell                
                #F_h = self.history_flow_contribution(river_obj, rid, tstep, train=train)
                
                # sum contributions and output total flow value at drainage cell
                Flow = F_u + river_obj.Fl_t[rid][:,-1:,:] # + F_h            
                river_obj.F_t[rid] = torch.cat([river_obj.F_t[rid], Flow], dim=1) # B, L, C
        
        loss_tstep = self.calculate_loss(river_obj, [tstep])
        
        if self.opt_counter==0:
            self.loss_opt = loss_tstep
        else:
            self.loss_opt = self.loss_opt + loss_tstep
        self.opt_counter += 1
                
        if (tstep % nt_opt == 0 and tstep > 0) or tstep == (len(date_range)-1):
            # tsteps_opt = list(range(tstep+1-nt_opt, tstep+1))
            # loss_opt = self.calculate_loss(river_obj, tsteps_opt)
            if train:
                self.opt_step()
            else:
                pass
        
        return river_obj, loss_tstep.item(), self.loss_opt.item(), self.opt_counter
    
    def prepare_inputs(self, river_obj, device, teacher_forcing=False):
        river_obj.S_t = {}
        river_obj.L_t = {}
        river_obj.F_t = {}
        river_obj.P_t = {}
        river_obj.Fu_t = {}
        river_obj.Fl_t = {}
        river_obj.c_feats = {}
        river_obj.r_feats = {}
        #river_obj.h_feats = {}
        
        river_obj.sorted_ids = river_obj.network.sort_values('reach_level', ascending=False).id.values
        
        if teacher_forcing:
            river_obj.flow_teacherforcing = {}

        # encode time
        self.time_ins = torch.from_numpy(np.stack([
            np.sin(river_obj.precip_data[river_obj.river_id].index.day_of_year / 365 * 2 * np.pi),
            np.cos(river_obj.precip_data[river_obj.river_id].index.day_of_year / 365 * 2 * np.pi),
            np.sin((river_obj.precip_data[river_obj.river_id].index.hour + 
                river_obj.precip_data[river_obj.river_id].index.minute/60) / 24. * 2 * np.pi),
            np.cos((river_obj.precip_data[river_obj.river_id].index.hour + 
                river_obj.precip_data[river_obj.river_id].index.minute/60) / 24. * 2 * np.pi)
        ], axis=1)).to(torch.float32).unsqueeze(0).to(device) # B, L, C
        
        # useful structures
        self.ones_vec = torch.ones((1,1,1), dtype=torch.float32).to(device)
        self.zeros_vec = torch.zeros((1,1,1), dtype=torch.float32).to(device)
        self.indicator_U = self.ones_vec.clone().to(device)
        self.indicator_L = self.ones_vec.clone().to(device) * 0.5

        for rid in river_obj.sorted_ids:
            # precipitation
            river_obj.P_t[rid] = (torch.from_numpy(river_obj.precip_data[rid].values)
                .to(torch.float32)
                .unsqueeze(0)
                .to(device) # B, L, C
            )
            
            # reach features
            river_obj.r_feats[rid] = torch.from_numpy(
                river_obj.cds_increm.loc[rid, ['REACH_LENGTH', 'REACH_SLOPE']].values
            ).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device) # B, L, C
            
            # catchment features
            these_cds = river_obj.cds_increm.loc[rid].drop(
                ['REACH_LENGTH', 'REACH_SLOPE', 'QUEX', 'QUE2'] + 
                river_obj.lc_1990_names + river_obj.lc_2015_names +                 
            )
            river_obj.c_feats[rid] = torch.from_numpy(
                np.hstack([these_cds, river_obj.mean_urbext.loc[rid], river_obj.mean_lc.loc[rid]])
            ).to(torch.float32)[None,...].unsqueeze(0).to(device) # B, L, C
            
            # # history features
            # river_obj.h_feats[rid] = torch.from_numpy(np.hstack([
                # river_obj.cds_increm.loc[rid, ['REACH_SLOPE']].values,
                # river_obj.cds_cumul.loc[rid, ['CCAR']].values
            # ])).to(torch.float32).unsqueeze(0).unsqueeze(0) # B, L, C
            
            if teacher_forcing:
                river_obj.flow_teacherforcing[rid] = torch.from_numpy(
                    river_obj.flow_est[rid].flow.values
                ).to(torch.float32).unsqueeze(0).unsqueeze(-1).to(device) # B, L, C
        
        return river_obj
    
    def initial_lateral_inflow_and_storage(self, river_obj, rid, train=False):
        # this is different for a variety of reasons!        
        thisdevice = river_obj.P_t[river_obj.river_id].device
        
        ## calculate initial storage
        ''' S_0 = Linear(soil moisture, CDs) '''
        ante_soilwetness = torch.from_numpy(
            np.array([river_obj.antecedent_soil_wetness.at[rid, 'soil_wetness']])
        ).to(torch.float32)[None,...].unsqueeze(0).to(thisdevice) # B, L, C

        S_inputs = torch.cat([ante_soilwetness, river_obj.c_feats[rid]], dim=-1)  
        
        if train:
            S0 = self.s_model(S_inputs) # B, L, C
        else:
            with torch.no_grad():
                S0 = self.s_model(S_inputs) # B, L, C
        
        ## first timepoint of lateral in-flow
        LS_src = torch.zeros((1,0,1), dtype=torch.float32).to(thisdevice)
        LS_trg = torch.cat([
            S0,
            river_obj.P_t[rid][:, :1, :],            
            river_obj.c_feats[rid],
            self.time_ins[:, :1, :]
        ], dim=-1)
        
        with torch.no_grad() if not train else nullcontext():
            LS_latent = self.ls_model(LS_src, LS_trg, mask=None) # B, L, C    
            L_out = self.l_from_ls_model(LS_latent) # B, L, C
            S_out = self.s_from_ls_model(LS_latent) # B, L, C        
        
        return L_out, torch.cat([S0, S_out], dim=1)
    
    def batched_initial_lateral_inflow_and_storage(self, river_obj, train=False):        
        new_batch = True
        thisdevice = river_obj.P_t[river_obj.river_id].device
        n = 0
        while n < len(river_obj.sorted_ids):
            rid = river_obj.sorted_ids[n]
            ante_soilwetness = torch.from_numpy(
                np.array([river_obj.antecedent_soil_wetness.at[rid, 'soil_wetness']])
            ).to(torch.float32)[None,...].unsqueeze(0).to(thisdevice) # B, L, C
            catchment_inputs = torch.cat([
                river_obj.P_t[rid][:, :1, :],        
                river_obj.c_feats[rid]
            ], dim=-1)
            
            if new_batch:
                S_inputs = torch.cat([ante_soilwetness, river_obj.c_feats[rid]], dim=-1)        
                LS_src = torch.zeros((1,0,1), dtype=torch.float32).to(thisdevice)
                LS_trg = torch.cat([catchment_inputs, self.time_ins[:, :1, :]], dim=-1)
                batch_rids = [rid]
            else:
                # append to current batch
                S_inputs = torch.cat([
                    S_inputs,
                    torch.cat([ante_soilwetness, river_obj.c_feats[rid]], dim=-1)
                ], dim=0)        
                LS_src = torch.cat([LS_src, torch.zeros((1,0,1), dtype=torch.float32).to(thisdevice)], dim=0)
                LS_trg = torch.cat([
                    LS_trg,
                    torch.cat([catchment_inputs, self.time_ins[:, :1, :]], dim=-1)
                ], dim=0)
                batch_rids.append(rid)
            
            if (n+1)%self.max_batch_size==0 or n==(len(river_obj.sorted_ids)-1):
                # run a batch        
                with torch.no_grad() if not train else nullcontext():
                    S0 = self.s_model(S_inputs) # B, L, C

                LS_trg = torch.cat([S0, LS_trg], dim=-1)

                with torch.no_grad() if not train else nullcontext():
                    LS_latent = self.ls_model(LS_src, LS_trg, mask=None) # B, L, C    
                    L_out = self.l_from_ls_model(LS_latent) # B, L, C
                    S_out = self.s_from_ls_model(LS_latent) # B, L, C                
                
                # store
                for j, rid in enumerate(batch_rids):            
                    river_obj.S_t[rid] = torch.cat([S0[j:(j+1),:,:], S_out[j:(j+1),:,:]], dim=1)
                    river_obj.L_t[rid] = L_out[j:(j+1),:,:]
                    
                new_batch = True
            else:
                new_batch = False
            n += 1
        return river_obj

    def build_LS_inputs(self, river_obj, rid, tstep):
        LS_src = torch.cat([
            river_obj.S_t[rid][:, :tstep, :],
            river_obj.L_t[rid][:, :tstep, :],
            river_obj.P_t[rid][:, :tstep, :],            
            river_obj.c_feats[rid].expand(-1, tstep, -1),
            self.time_ins[:, :tstep, :]
        ], dim=-1)
        LS_trg = torch.cat([
            river_obj.S_t[rid][:, tstep:(tstep+1), :],
            river_obj.P_t[rid][:, tstep:(tstep+1), :],            
            river_obj.c_feats[rid],
            self.time_ins[:, tstep:(tstep+1), :]
        ], dim=-1)
        return LS_src, LS_trg
    
    def lateral_inflow_and_storage(self, river_obj, rid, date_range, tstep, train=False):
        ## calculate lateral in-flow and new storage 
        thisdevice = river_obj.P_t[river_obj.river_id].device        
        '''
        LS_latent = Transformer(
            src: [{S_0, S(t_{i-n} <= t < t_{i-1})}, L(t_{i-n} <= t < t_i), Precip(t_{i-n} <= t < t_i), time_sines, CDs]
            trg: [Precip(t_i), S(t_{i-1}), time_sines, CDs]
        )
        L = Linear(LS_latent)
        S = Linear(LS_latent)
        '''
        
        LS_src, LS_trg = self.build_LS_inputs(river_obj, rid, tstep)
        
        with torch.no_grad() if not train else nullcontext():
            LS_latent = self.ls_model(LS_src, LS_trg, mask=None) # B, L, C
            L_out = self.l_from_ls_model(LS_latent) # B, L, C
            S_out = self.s_from_ls_model(LS_latent) # B, L, C        
        
        # check if new day has ticked to calculate S_assim from soil wetness
        # do something like average of S_assim and S_out at new day ticks?    
        if date_range[tstep].hour==0 and date_range[tstep].minute==0:
            current_soilwetness = torch.from_numpy(
                np.array([river_obj.soil_wetness_data[rid].at[date_range[tstep, 'soil_wetness']]])
            ).to(torch.float32)[None,...].unsqueeze(0).to(thisdevice) # B, L, C
            
            S_inputs = torch.cat([current_soilwetness, river_obj.c_feats[rid]], dim=-1)        
            
            with torch.no_grad() if not train else nullcontext():
                S_assim = self.s_model(S_inputs) # B, L, C
            
            S_out = 0.5*(S_out + S_assim)
        else:
            pass
            
        return L_out, S_out
    
    def batched_lateral_inflow_and_storage(self, river_obj, date_range, tstep, train=False):        
        new_batch = True
        thisdevice = river_obj.P_t[river_obj.river_id].device
        n = 0
        while n < len(river_obj.sorted_ids):
            rid = river_obj.sorted_ids[n]
            
            L_s, L_t = self.build_LS_inputs(river_obj, rid, tstep)
            
            if new_batch:
                LS_src = L_s.clone()
                LS_trg = L_t.clone()
                batch_rids = [rid]
            else:
                LS_src = torch.cat([LS_src, L_s.clone()], dim=0)
                LS_trg = torch.cat([LS_trg, L_t.clone()], dim=0)
                batch_rids.append(rid)

            # check if new day has ticked to calculate S_assim from soil wetness
            # do something like average of S_assim and S_out at new day ticks?    
            if date_range[tstep].hour==0 and date_range[tstep].minute==0:
                current_soilwetness = torch.from_numpy(
                    np.array([river_obj.soil_wetness_data[rid].at[date_range[tstep, 'soil_wetness']]])
                ).to(torch.float32)[None,...].unsqueeze(0).to(thisdevice) # B, L, C
                
                if new_batch:
                    S_inputs = torch.cat([current_soilwetness, river_obj.c_feats[rid]], dim=-1)
                else:
                    S_inputs = torch.cat([
                        S_inputs,
                        torch.cat([current_soilwetness, river_obj.c_feats[rid]], dim=-1),
                    ], dim=0)
                    
                if (n+1)%self.max_batch_size==0 or n==(len(river_obj.sorted_ids)-1):
                    with torch.no_grad() if not train else nullcontext():
                        S_assim = self.s_model(S_inputs) # B, L, C                    
                    
            if (n+1)%self.max_batch_size==0 or n==(len(river_obj.sorted_ids)-1):
                with torch.no_grad() if not train else nullcontext():
                    LS_latent = self.ls_model(LS_src, LS_trg, mask=None) # B, L, C
                    L_out = self.l_from_ls_model(LS_latent) # B, L, C
                    S_out = self.s_from_ls_model(LS_latent) # B, L, C                
                
                if date_range[tstep].hour==0 and date_range[tstep].minute==0:       
                    S_out = 0.5*(S_out + S_assim)

                # store
                for j, rid in enumerate(batch_rids):            
                    river_obj.S_t[rid] = torch.cat([river_obj.S_t[rid], S_out[j:(j+1),:,:]], dim=1)#.requires_grad_(True)
                    river_obj.L_t[rid] = torch.cat([river_obj.L_t[rid], L_out[j:(j+1),:,:]], dim=1)#.requires_grad_(True)
                    
                new_batch = True        
            else:
                new_batch = False
            n += 1
        return river_obj
    
    def build_Fu_inputs(self, river_obj, rid, tstep, Fup_sum, thisdevice):
        if tstep>1:
            Fu_src = torch.cat([
                river_obj.Fu_t[rid][:, :tstep, :],
                torch.cat([self.zeros_vec, self.ones_vec.expand(-1, tstep-1, -1)], dim=1), # presence indicator                    
                Fup_sum[:, :tstep, :],
                self.indicator_U.expand(-1, tstep, -1),
                river_obj.r_feats[rid].expand(-1, tstep, -1),
                self.time_ins[:, :tstep, :],
            ], dim=-1)
            Fu_trg = torch.cat([        
                Fup_sum[:, tstep:(tstep+1), :],
                self.indicator_U,
                river_obj.r_feats[rid],
                self.time_ins[:, tstep:(tstep+1), :]
            ], dim=-1)
        else:
            Fu_src = torch.zeros((1,0,1), dtype=torch.float32).to(thisdevice)
            Fu_trg = torch.cat([        
                Fup_sum[:, :1, :],
                self.indicator_U,
                river_obj.r_feats[rid],
                self.time_ins[:, :1, :]
            ], dim=-1)
        return Fu_src, Fu_trg
                
    def upstream_flow_contribution(self, river_obj, rid, tstep,
                                   teacher_forcing=False, train=False):
        # check for upstream flows and calculate propagation downstream    
        upstream_cells = river_obj.network[(river_obj.network.dwn_id==rid)].query('tidal==False')
        thisdevice = river_obj.P_t[river_obj.river_id].device
        if upstream_cells.shape[0]>0:
            # do the upstream model
            '''
            F_u = Transformer(
                src: [F_u(t_{i-n} <= t < t_i), presence_indicator, 
                        sum(F_upstream)(t_{i-n} <= t < t_i), upstream_indicator,
                        reach_features, time_sines]
                trg: [sum(F_upstream)(t_i), upstream_indicator,
                        reach_features, time_sines]
            )
            '''
            if teacher_forcing:
                Fup_sum = sum([river_obj.flow_teacherforcing[k] for k in upstream_cells.id])
            else:
                Fup_sum = sum([river_obj.F_t[k] for k in upstream_cells.id])
            
            Fu_src, Fu_trg = self.build_Fu_inputs(river_obj, rid, tstep, Fup_sum, thisdevice)
            
            with torch.no_grad() if not train else nullcontext():
                Fu_latent = self.fstream_model(Fu_src, Fu_trg, mask=None)
                F_u = self.latent2flow_model(Fu_latent)
                
        else:
            # set upstream contribution to zero
            F_u = torch.zeros((1,1,1), dtype=torch.float32).to(thisdevice)
        
        return F_u
    
    def build_Fl_inputs(self, river_obj, rid, tstep):
        Fl_src = torch.cat([
            river_obj.Fl_t[rid][:, :tstep, :],
            torch.cat([self.zeros_vec, self.ones_vec.expand(-1, tstep-1, -1)], dim=1), # presence indicator   
            river_obj.L_t[rid][:, :tstep, :],
            self.indicator_L.expand(-1, tstep, -1),
            river_obj.r_feats[rid].expand(-1, tstep, -1),
            self.time_ins[:, :tstep, :]
        ], dim=-1)
        Fl_trg = torch.cat([
            river_obj.L_t[rid][:, tstep:(tstep+1), :],
            self.indicator_L,
            river_obj.r_feats[rid],
            self.time_ins[:, tstep:(tstep+1), :]
        ], dim=-1)
        return Fl_src, Fl_trg
    
    def terrestrial_flow_contribution(self, river_obj, rid, tstep, train=False):
        # calculate lateral in-flow contribution at the drainage cell
        '''
        F_l = Transformer(
            src: [F_l(t_{i-n} <= t < t_i), presence_indicator, L(t_{i-n} <= t < t_i),
                    terrestrial_indicator, reach_features, time_sines]
            trg: [L(t_i), terrestrial_indicator, reach_features, time_sines]
        )
        '''        
        Fl_src, Fl_trg = self.build_Fl_inputs(river_obj, rid, tstep)
        
        with torch.no_grad() if not train else nullcontext():
            Fl_latent = self.fstream_model(Fl_src, Fl_trg, mask=None)
            F_l = self.latent2flow_model(Fl_latent)        
        
        return F_l
        
    def batched_terrestrial_flow_contribution(self, river_obj, tstep, train=False):        
        new_batch = True
        n = 0
        F_l_out = {}
        while n < len(river_obj.sorted_ids):
            rid = river_obj.sorted_ids[n]

            Fl_s, Fl_t = self.build_Fl_inputs(river_obj, rid, tstep)
            
            if new_batch:
                Fl_src = Fl_s.clone()
                Fl_trg = Fl_t.clone()
                batch_rids = [rid]
            else:
                Fl_src = torch.cat([Fl_src, Fl_s.clone()], dim=0)
                Fl_trg = torch.cat([Fl_trg, Fl_t.clone()], dim=0)                
                batch_rids.append(rid)
                
            if (n+1)%self.max_batch_size==0 or n==(len(river_obj.sorted_ids)-1):
                with torch.no_grad() if not train else nullcontext():
                    Fl_latent = self.fstream_model(Fl_src, Fl_trg, mask=None)
                    F_l = self.latent2flow_model(Fl_latent)                
                
                # store
                for j, rid in enumerate(batch_rids):                    
                    river_obj.Fl_t[rid] = torch.cat([river_obj.Fl_t[rid], F_l[j:(j+1),:,:]], dim=1)#.requires_grad_(True)
                    
                new_batch = True        
            else:
                new_batch = False
            n += 1
        #return F_l_out
        return river_obj
    
    def history_flow_contribution(self, river_obj, rid, tstep, train=False):
        # calculate flow history attenuation (hysteresis of reach flow?)
        '''
        F_h = Transformer(
            src: [F(t_{i-n} <= t <= t_{i-1}), history_features, time_sines]
            trg: [history_features, time_sines]
        )
        '''        
        Fh_src = torch.cat([
            river_obj.F_t[rid][:, :tstep, :],
            river_obj.h_feats[rid].expand(-1, tstep, -1),
            self.time_ins[:, :tstep, :]
        ], dim=-1)
        Fh_trg = torch.cat([            
            river_obj.h_feats[rid],
            self.time_ins[:, tstep:(tstep+1), :]
        ], dim=-1)
        
        with torch.no_grad() if not train else nullcontext():
            Fh_latent = self.fh_model(Fh_src, Fh_trg, mask=None)
            F_h = self.latent2flow_model(Fh_latent)        
        
        return F_h
    

if False:       
    
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
    
    river_obj, rid, date_range, event = sample_event(train_events, vwc_quantiles=vwc_quantiles)
    
    # or to load same event
    rid = 78809
    date_range = pd.date_range('2000-01-10 13:00:00', '2000-01-14 03:00:00', freq="15T")
    river_obj, rid, date_range, event = sample_event(
        train_events,
        vwc_quantiles=vwc_quantiles,
        rid=rid, date_range=date_range
    )
        
    tstep = 0
    train = True
    RUN = False
        
    if train:
        model.train()
    else:
        model.eval()
    
    # perform a single forward-pass time tick across all reaches
    if tstep==0:
        ## do initial one-off things
        river_obj = model.prepare_inputs(river_obj, device, teacher_forcing=teacher_forcing)
        
        ## run forward initial timepoint
        river_obj = model.batched_initial_lateral_inflow_and_storage(river_obj, train=train)                
        for rid in river_obj.network.sort_values('reach_level', ascending=False).id:
            # prepare initial flows
            river_obj.F_t[rid] = river_obj.flow_teacherforcing[rid][:,:1,:]
            river_obj.Fu_t[rid] = model.zeros_vec.clone().to(device)
            river_obj.Fl_t[rid] = model.zeros_vec.clone().to(device)
    else:        
        ## do normal forward algorithm parts            
        # in-flow from land and storage state of catchment
        river_obj = model.batched_lateral_inflow_and_storage(river_obj, date_range, tstep, train=train)
        
        print("S_t")
        print(river_obj.S_t[river_obj.river_id])
        print("L_t")
        print(river_obj.L_t[river_obj.river_id])
        
        # contribution to flow from land at drainage cell
        river_obj = model.batched_terrestrial_flow_contribution(river_obj, tstep, train=train)
        
        print("Fl_t")
        print(river_obj.Fl_t[river_obj.river_id])
        
        for rid in river_obj.sorted_ids:
            # contribution to flow from upstream at drainage cell                
            F_u = model.upstream_flow_contribution(
                river_obj,
                rid,
                tstep,
                teacher_forcing=teacher_forcing,
                train=train
            )
            river_obj.Fu_t[rid] = torch.cat([river_obj.Fu_t[rid], F_u], dim=1)
            
            # sum contributions and output total flow value at drainage cell
            Flow = F_u + river_obj.Fl_t[rid][:,-1:,:] # + F_h            
            river_obj.F_t[rid] = torch.cat([river_obj.F_t[rid], Flow], dim=1) # B, L, C
        
        print("Fu_t")
        print(river_obj.Fu_t[river_obj.river_id])
        print("F_t")
        print(river_obj.F_t[river_obj.river_id])
        
    loss_tstep = model.calculate_loss(river_obj, [tstep])
    
    if model.opt_counter==0:
        model.loss_opt = loss_tstep
    else:
        model.loss_opt = model.loss_opt + loss_tstep
        
    if tstep>0:
        model.opt_counter += 1
            
    if (tstep % nt_opt == 0 and tstep > 0) or tstep == (len(date_range)-1):
        old_params = gather_params(model)
        if train:
            model.opt_step()
            # then detach variables from graph so we aren't 
            # propagating back through same graph twice?
            print("S_t")
            print(river_obj.S_t[rid])
            print("L_t")
            print(river_obj.L_t[rid])
            print("Fl_t")
            print(river_obj.Fl_t[rid])
            print("Fu_t")
            print(river_obj.Fu_t[rid])
            print("F_t")
            print(river_obj.F_t[rid])
            for rid in river_obj.sorted_ids:
                river_obj.S_t[rid].detach_() # missing gradients here? only when using the batched methods??
                river_obj.L_t[rid].detach_() # missing gradients here? only when using the batched methods??
                river_obj.Fl_t[rid].detach_() # missing gradients here? only when using the batched methods??
                river_obj.Fu_t[rid].detach_()
                river_obj.F_t[rid].detach_()
           
            new_params = gather_params(model)
            print((new_params - old_params).sum())
            print((new_params - old_params).max())
            print((new_params - old_params).min())
        else:
            pass
    tstep += 1
