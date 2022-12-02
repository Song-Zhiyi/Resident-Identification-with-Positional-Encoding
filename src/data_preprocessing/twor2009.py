import re
import datetime
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

__all__ = ['get_data']

classrep = re.compile(r"R\d+")

logger = logging.getLogger("e.dataset.twor2009.preprocess")

def annotate_row(df):
    id_list = []
    event_list = []
    current_class = None
    current_event = None
    for index, r in df.iterrows():
        event = r['event']
        if not pd.isna(event):
            current_class = event.split("_")[0]
            current_event = event
        event_list.append(current_event)
        if bool(classrep.match(current_class)):
            id_list.append(current_class)
        else:
            id_list.append(pd.NA)

    df['id'] = id_list
    df['event_annotated'] = event_list

    #df = df.drop(columns=['event', 'event_status'])
    return df

def fix_datetime(df):
    datetime_list = []

    for index, r in df.iterrows():
        try:
            ts = "{} {}".format(r['date'], r['time'])
            time = datetime.datetime.strptime(
                ts,
                "%Y-%m-%d %H:%M:%S.%f"
            )
        except Exception as e:
            time = pd.NA
            logger.warning(f"Not a valid time format: {e!s}")

        datetime_list.append(time)

    df['datetime'] = pd.to_datetime(datetime_list)
    df = df.drop(columns=['date', 'time'])
    return df

def sort_by_time(df):
    df.sort_values(by='datetime', inplace=True)
    return df

def sensor_reading_to_number(dat):
    dm_sensor_pat = re.compile(r"[DM]\d+")
    def fix_open_close(r):
        if r in ("CLOSE", "OFF", "ABSENT"):
            return 0
        elif r in ("OPEN", "ON", "PRESENT"):
            return 1
        return pd.NA

    dat['read_bin'] = dat['read'].map(fix_open_close)
    return dat

def encode_event_resident(dat):
    dat.id = pd.Categorical(dat.id)
    dat.event_annotated = pd.Categorical(dat.event_annotated)
    dat['resident_id'] = dat.id.cat.codes
    dat['event_id'] = dat.event_annotated.cat.codes
    return dat

def cyclical_time(dat):

    def apply_cyclical_time(df, col, max_t):
        ang = 2 * np.pi * df[col] / max_t
        df[col + "_sin"] = np.sin(ang)
        df[col + "_cos"] = np.cos(ang)

    def seconds_of_day(hour, minute, second):
        return hour * 3600 + minute * 60 + second

    dat_dt = dat['datetime'].dt
    dat['year'] = dat_dt.year
    dat['dayofyear'] = dat_dt.dayofyear - 1
    dat['secondofday'] = seconds_of_day(dat_dt.hour, dat_dt.minute, dat_dt.second) - 1
    dat['weekday'] = dat_dt.weekday

    apply_cyclical_time(dat, 'weekday', 7)
    apply_cyclical_time(dat, 'dayofyear', 365)
    apply_cyclical_time(dat, 'secondofday', 86400)

    return dat

def get_data(force: bool=False,
             data_path="./data/twor.2009/annotated",
             cache_path="preprocessed/data/twor-2009-annotated-preprocessed.pkl"):
    data_path = Path(data_path)
    saved_path = Path(cache_path)

    try:
        if not force:
            logger.info(f"loading cached data from {saved_path!s}")
            return pd.read_pickle(saved_path, compression="xz")
    except Exception as e:
        logger.warning(f"load cache failed: {e!s}. Try to re-preprocess file")

    orig_dat = pd.read_table(
        data_path,
        sep=" ",
        names=["date", "time", "sensor", "read", "event", "event_status"]
    )
    dat = orig_dat
    dat = fix_datetime(dat)
    dat = annotate_row(dat)
    dat = sort_by_time(dat)
    dat = encode_event_resident(dat)
    dat = cyclical_time(dat)
    dat = sensor_reading_to_number(dat)
    #dat = dat.dropna()

    try:
        logger.info(f"saving preprocessed data into {saved_path!s}")
        dat.to_pickle(saved_path, compression="xz")
    except Exception as e:
        logger.warning(f"saving preprocessed file to {saved_path!s} failed: {e!s}")

    return dat
