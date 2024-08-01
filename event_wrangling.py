import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from river_reach_modelling.utils import remove_prefix, remove_suffix
from river_reach_modelling.clip_catchment_boundaries import grab_catchment_rainfall
from river_reach_modelling.catchment_wrangling import wrangle_descriptors

# define the path to the data
basedir = '/gws/nopw/j04/hydro_jules/data/uk/flood_events/flood_event_archive/'
#basedir = '/gws/nopw/j04/hydro_jules/data/uk/flood_events/'
#datadir = basedir + '/old_event_archive/NFEA_total_flow-based_v6_csv/'
#statdir = basedir + '/old_event_archive/NFEA_total_flow-based_v6_statistics/'


def load_statistics():    
    stats_df = pd.read_csv(basedir + "/event_statistics.csv")
    return stats_df

def load_old_statistics():
    # load event statistics
    stats_fs = glob.glob(statdir + '*.csv')
    stats_df = pd.DataFrame()
    for i,f in enumerate(stats_fs):
        dd = pd.read_csv(f)
        sname = remove_suffix(remove_prefix(f, statdir), '.csv')
        dd = (dd.melt(id_vars='Station')
            .rename({'value':sname, 'variable':'Event'}, axis=1)
            .dropna())
        if i==0:
            stats_df = dd
        else:
            stats_df = stats_df.merge(dd, how='left', on=['Station', 'Event'])

    stats_df['EventDuration'] = pd.to_datetime(stats_df.FlowEndDate) - pd.to_datetime(stats_df.FlowStartDate)
    stats_df['EventDuration'] = stats_df.EventDuration / datetime.timedelta(seconds=1)
    stats_df['EventDuration_15min'] = (stats_df.EventDuration / (60 * 15)).astype(int)

    # extracting station flag types
    events_f = glob.glob(datadir + "/*Station*event*.csv")
    flag_types = []
    station_ids = []
    for ef in events_f:
        flag_str = ef.split('/')[-1].split('_')[0]
        flag_types.append(flag_str)
        station_id = int(ef.split('Station')[-1].split('_')[1])
        station_ids.append(station_id)
    station_flags = pd.DataFrame({'Station':station_ids, 'flag':flag_types}).drop_duplicates().reset_index(drop=True)
    stats_df = stats_df.merge(station_flags, how='left', on='Station')

    return stats_df


def load_event(stats_df, l=None, event_num=None, station_num=None,
               antecedent_rain_days=0, catch_descs=None):
                 
    if (l is None) and ((event_num is None) or (station_num is None)):
        uniq_lengths, num_lengths = np.unique(stats_df.EventDuration.values, return_counts=True)
        l = np.random.choice(uniq_lengths, size=1, p=num_lengths/sum(num_lengths))[0]
            
    if (event_num is None) or (station_num is None):
        df = stats_df.loc[stats_df['EventDuration_15min']==l].sample(1, replace=False)
        event_num = df.Event.values[0].replace('Event_','')
        station_num = df.Station.values[0]        
    else:
        event_num = event_num.replace('Event_','')
        station_num = int(station_num)
        mask1 = stats_df['Event']==f'Event_{event_num}'
        mask2 = stats_df['Station']==station_num
        mask3 = mask1 * mask2
        df = stats_df.loc[mask3]
        
    date_str = df.FlowStartDate.values[0][:10]
    time_str = df.FlowStartDate.values[0][11:13] + df.FlowStartDate.values[0][14:16]
    flag_str = df.flag.values[0]

    # load and process event timeseries    
    dat = pd.read_csv(datadir + f'/{flag_str}_Station_{station_num}_event_{date_str}_{time_str}.csv')
    catch_id = str(df.Station.values[0])
    dat['DateTime'] = pd.to_datetime(dat['DateTime'])#, format="%Y %m %d %H:%M:%S") 
    start_date = dat.DateTime.iloc[0]
    end_date = dat.DateTime.iloc[-1]
    dat = dat.set_index('DateTime')

    # spread hourly rainfall backwards to 15 min res
    dat['minute'] = dat.index.minute    
    dat.loc[dat['minute']!=0, 'Rain'] = np.nan
    dat['Rain'] /= 4. # divide equally between 15min blocks
    dat['Rain'] = dat.Rain.fillna(method='bfill')
    dat = dat.drop('minute', axis=1)
        
    if antecedent_rain_days>0:
        # load antecedent rainfall from clipped GEAR hourly data
        start_date_ante = start_date - datetime.timedelta(days = antecedent_rain_days)
        gear_dict = grab_catchment_rainfall(catch_id, start_date_ante, end_date,
                                            spatial=False, land_attrs=False)
        precip = (gear_dict['ca_timeseries'].to_dataframe()
            .reset_index()
            .rename({'time':'DateTime', 'rainfall_amount': 'Antecedent_Rain'}, axis=1)
            .set_index('DateTime')
            .drop(['spatial_ref', 'crs'], axis=1)
        )
        precip = precip.loc[pd.date_range(start_date_ante, start_date, freq="H")[:-1]]
        precip.index.name = 'DateTime'

        # join to flow        
        flow_time = pd.date_range(start_date - datetime.timedelta(days = antecedent_rain_days), end_date, freq="15min")
        template = pd.DataFrame({'dummy':0, 'DateTime':flow_time}).set_index('DateTime')
        dat = template.merge(dat, on='DateTime', how='left').merge(precip, on='DateTime', how='left')
        dat['Antecedent_Rain'] /= 4. # divide equally between 15min blocks
        dat['Antecedent_Rain'] = dat.Antecedent_Rain.fillna(method='bfill')
        
        dat['Rain'] = np.nansum(dat[['Rain','Antecedent_Rain']], axis=1)
        dat = dat.drop(['dummy', 'Antecedent_Rain'], axis=1)
    
    # normalise by catchment area    
    if catch_descs is None:
        catch_descs, _ = wrangle_descriptors()        
    c_area = catch_descs.loc[int(catch_id)].CCAR
    to_normalise_by_area = ['Flow', 'FlowSplined']
    dat.loc[:,to_normalise_by_area] = dat.loc[:,to_normalise_by_area].div(c_area, axis=0)
    
    meta = {
        'event_num':event_num,
        'station_num':station_num,
        'start_date':start_date,
        'end_date':end_date,
        'flag_str':flag_str
    }
    return dat, meta
