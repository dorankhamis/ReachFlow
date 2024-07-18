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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

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

# get feature order from grab_names()
event_dat_path = hj_base + '/flood_events/event_encoding/'
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

catchment_file_list = glob.glob(event_dat_path + '*.npz')

# for a catchment, load up events and examine
dat = np.load(catchment_file_list[0])
keys = list(dat)
flows = pd.DataFrame()
tokens = pd.DataFrame()
for k in keys:
    sample = dat[k]
    this_flow = (pd.DataFrame(sample[:,0], columns=['FLOW'])
        .assign(Event = k)
        .assign(t_ind = np.arange(sample.shape[0]))
    )
    flows = pd.concat([flows, this_flow], axis=0)
    these_tokens = (pd.DataFrame(sample[:,1:], columns=names[1:])
        .assign(Event = k)
    )
    tokens = pd.concat([tokens, these_tokens], axis=0)

# number of tokens with rainfall versus event length?

# variability in rainfall>0 tokens?
p_tokens = tokens.query('PRECIP > 0')
p_stats = p_tokens.groupby("Event").agg({'PRECIP':'sum'}).reset_index()
p_stats.columns = ['Event', 'total_precip']
f_stats = flows.groupby("Event").agg({'FLOW':['max', 'sum', 'min'], 't_ind':'max'}).reset_index()
f_stats.columns = ['Event', 'peak_flow', 'total_volume', 'base_flow', 'event_length']
f_stats['excess_volume'] = f_stats.total_volume - (f_stats.base_flow * f_stats.event_length) # still in m3/s rather than m3
f_stats = f_stats.merge(p_stats, on='Event', how='left')

for cc in np.setdiff1d(p_tokens.columns, ['Event']):
    if cc in norm_dict.keys():
        p_tokens.loc[:,cc] = p_tokens[cc] / norm_dict[cc]
    p_tokens.loc[:,cc] = (p_tokens[cc] - p_tokens[cc].mean()) / p_tokens[cc].std()
p_tokens = p_tokens.fillna(0)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_out = pca.fit_transform(p_tokens.drop(['Event', 'CCAR', 'PRECIP'], axis=1))
print(pca.explained_variance_ratio_)
pca_out = pd.DataFrame(pca_out, columns=["pca_1", "pca_2", "pca_3"])
pca_out = pd.concat([pca_out.reset_index(drop=True), p_tokens.reset_index(drop=True)], axis=1)
pca_out = pca_out.merge(f_stats, on='Event', how='left')

plt.plot(pca_out.total_precip, pca_out.excess_volume, 'o', markersize=1.4); plt.show()
plt.plot(pca_out.total_precip, pca_out.peak_flow, 'o', markersize=1.4); plt.show()
plt.plot(pca_out.pca_1, pca_out.excess_volume, 'o', markersize=1.4); plt.show()
plt.plot(pca_out.pca_1, pca_out.peak_flow, 'o', markersize=1.4); plt.show()
plt.plot(pca_out.pca_2, pca_out.excess_volume, 'o', markersize=1.4); plt.show()
plt.plot(pca_out.pca_2, pca_out.peak_flow, 'o', markersize=1.4); plt.show()
plt.plot(pca_out.pca_2, pca_out.excess_volume, 'o', markersize=1.4); plt.show()
plt.plot(pca_out.pca_2, pca_out.peak_flow, 'o', markersize=1.4); plt.show()

import seaborn as sns
sns.scatterplot(x="pca_1", y="pca_2", data=pca_out, hue="Event")
sns.scatterplot(x="pca_1", y="pca_2", data=pca_out, hue="peak_flow", s=3.9, palette='viridis')
sns.scatterplot(x="pca_1", y="pca_2", data=pca_out, hue="excess_volume", s=3.9, palette='viridis')
for nn in pca.feature_names_in_:
    sns.scatterplot(x="pca_1", y="pca_2", data=pca_out, hue=nn)
    
for ev in f_stats.Event.iloc[:10]:
    pltdat = pca_out[pca_out.Event==ev]
    plt.scatter(pltdat.pca_1, pltdat.pca_2, s=1.5)
    plt.xlim(pca_out.pca_1.min(), pca_out.pca_1.max())
    plt.ylim(pca_out.pca_2.min(), pca_out.pca_2.max())
plt.show()


# bin up the events into clusters that have similar total rainfall
# or total rainfall/event length?
rain_over_time = tokens.groupby('Event').agg('mean')[['PRECIP']].sort_values('PRECIP')
rain_max_instant = tokens.groupby('Event').agg('max')[['PRECIP']].sort_values('PRECIP')
rain_total = tokens.groupby('Event').agg('sum')[['PRECIP']].sort_values('PRECIP')

pltdat = pd.merge(
    rain_total.reset_index(),
    flows.groupby('Event').agg('max').reset_index(),
    on='Event', how='left'
)
plt.plot(pltdat.PRECIP, pltdat.FLOW, 'o')
plt.show()

# train random forest to predict max flow from statistics
# of the precip-normed tokens?

# load all events in this way and look at clustering between catchments?

agg_tokens = pd.merge(tokens[['Event', 'PRECIP']].groupby('Event').agg('sum'),
                      tokens.groupby('Event').agg('mean').drop('PRECIP', axis=1), on='Event')
pltdat = pd.merge(
    agg_tokens,
    flows.groupby('Event').agg('max'),
    on='Event', how='left'
)
for nn in names[1:]:
    plt.plot(pltdat[nn], pltdat.FLOW, 'o')
    plt.title(nn)
    plt.show()
'''
is interesting patterns in the land cover/flow plots
but what else can we say? want to use time dependence really... 
'''

import seaborn as sns
sns.jointplot(x=pltdat.PRECIP, y=pltdat.FLOW, kind="hex", color="#4CB391")
plt.show()
