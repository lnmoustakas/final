import numpy as np
import torch
import darts
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import statsmodels.api as sm

from datetime import datetime
from scipy.optimize import curve_fit
from darts.utils.utils import ModelMode
from darts import TimeSeries
from darts.metrics import rmse
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
    ExponentialSmoothing,
)
from darts.metrics import mape, smape
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset


## Description: Fit the data to a hyperbolic function

#----------------------------------#

## Variables ##
# t time in consistent units from start of production
# qi initial production rate
# b hyperbolic decline variable
# di initial decline rate

production = "oil" # "oil" or "gas"
qi_cutoff = "30" # Number of initial days to pull qi indexed at 1 being the first month

#----------------------------------#

##Equations

def hyperbolic(t, qi, b, di):
    return qi/(((1+b*di*t)**(1/b))+.000001)

def exponential(t, qi, di):
    return qi*np.exp(-di*t)

#----------------------------------#

## Parse Data ##

def parse_data():
    df = pd.read_csv('21_01- 6.csv')
    df = df.dropna()
    #df = df.loc[~(df[production]<=20)]
    return(df)

#----------------------------------#

## ONLY USED FOR QI != FIRST PRODUCTION DATE ## not working

def qi_multi(df, qi_cutoff):
    df = df.groupby(['well_id']).apply(lambda x: x[production].iloc[0, qi_cutoff-1].max())
    df.columns.values[1] = 'qi'
    return(df)

#----------------------------------#

## Create Qi (assuming month 1 for qi) and Time columns (in days) added LOESS smoothing ##

def conversion(df):
    ## Use first production date for each well as qi date ##
    df["qi"] = df.groupby(['well_id']).transform('first')[production]
    ## Convert dates to time t from qi ##
    df['date'] = pd.to_datetime(df['date'])
    # t = df.groupby(['well_id'], as_index=False).apply(lambda x: x['date'] - x['date'].iloc[0]).reset_index()
    # !!!!MUST BE FIXED TO USE ON MULTIPLE WELL_IDs!!!!

    #code here is a temp fix For only one well_id
    df['time'] = (df['date'] - df['date'][0]).dt.days

    return(df)

## Fit Data after smoothing  ##

def fit_data(df):

    #Smoothing Data
    for well in df['well_id'].unique():
        df1 = df[df['well_id'] == well]
        df.loc[df['well_id'] == well, 'smooth_prod'] = sm.nonparametric.lowess(df1[production], df1['time'], frac=0.1)[:,1]
    
    #Exponential fit
    for well in df['well_id'].unique():
        df1 = df[df['well_id'] == well]
        popt, _ = curve_fit(exponential, df1['time'], df1['smooth_prod'], bounds=(0, [df1['smooth_prod'].iloc[0],20]))
        df.loc[df['well_id'] == well, 'qi_exp'] = popt[0]
        df.loc[df['well_id'] == well, 'di_exp'] = popt[1]

    #Hyperbolic fit
    for well in df['well_id'].unique():
        df1 = df[df['well_id'] == well]
        popt, _ = curve_fit(hyperbolic, df1['time'], df1['smooth_prod'], bounds=(0, [df1['smooth_prod'].iloc[0],2,20]))
        df.loc[df['well_id'] == well, 'qi_hyp'] = popt[0]
        df.loc[df['well_id'] == well, 'b_hyp'] = popt[1]
        df.loc[df['well_id'] == well, 'di_hyp'] = popt[2]
        
    #Calculate values using fitted parameters
    for well in df['well_id'].unique():
        df1 = df[df['well_id'] == well]
        df.loc[df['well_id'] == well, 'exp_value'] = exponential(df1['time'], df1['qi_exp'], df1['di_exp'])
        df.loc[df['well_id'] == well, 'hyp_value'] = hyperbolic(df1['time'], df1['qi_hyp'], df1['b_hyp'], df1['di_hyp'])

    return(df)

#----------------------------------#

## Plot Data ##

def plot_data(df):
    for well in df['well_id'].unique():
        df1 = df[df['well_id'] == well]
        plt.plot(df1['time'], df1[production], '.', label='data')
        plt.plot(df1['time'], df1['exp_value'], 'r-' ,label='exponential fit')
        plt.plot(df1['time'], df1['hyp_value'] , 'k--' ,label='hyperbolic fit')
        plt.plot(df1['time'], df1['smooth_prod'] , 'g--' ,label='smoothed data')
        plt.title('Well ID: ' + str(well))
        plt.xlabel('time (days)')
        plt.ylabel(production)
        plt.legend()
        plt.show()
    return(df)


df = parse_data()
df = conversion(df)
df = fit_data(df)
#plot_data(df)

#----------------------------------#

## Machine Learning ##

#Standardize data

def standardize(df):
    for well in df['well_id'].unique():
        df1 = df[df['well_id'] == well]

        #We plot scaled real production not smoothed production

        series_prod_plot = TimeSeries.from_dataframe(df1, 'date', production)
        scaler_prod_plot = Scaler()
        series_prod_plot_scaled = scaler_prod_plot.fit_transform(series_prod_plot)

        #----------------------------------#

        series_prod = TimeSeries.from_dataframe(df1, 'date', 'smooth_prod', fill_missing_dates=True , freq='D') 
        series_hyp = TimeSeries.from_dataframe(df1, 'date', 'hyp_value', fill_missing_dates=True , freq='D')
        scaler_prod, scaler_hyp = Scaler(), Scaler()

        scaler_prod1, scaler_hyp1 = Scaler(), Scaler()

        series_prod_scaled = scaler_prod1.fit_transform(series_prod)
        series_hyp_scaled = scaler_hyp1.fit_transform(series_hyp)

        train_prod, _ = series_prod_scaled[:-365*3], series_prod_scaled[-365*3:]
        train_hyp, _ = series_hyp_scaled[:-365*3], series_hyp_scaled[-365*3:]

        prod_month = datetime_attribute_timeseries(series_prod_scaled, attribute='month')
        prod_day = datetime_attribute_timeseries(series_prod_scaled, attribute='day')

        hyp_month = datetime_attribute_timeseries(series_hyp_scaled, attribute='month')
        hyp_day = datetime_attribute_timeseries(series_hyp_scaled, attribute='day')

        hyp_covariates = hyp_month.stack(hyp_day)
        prod_covariates = prod_month.stack(prod_day)

        prod_covariates = scaler_prod.fit_transform(prod_covariates)
        hyp_covariates = scaler_hyp.fit_transform(hyp_covariates)

        prod_train_covariates, prod_val_covariates = prod_covariates[:-365*3], prod_covariates[-365*3:]
        hyp_train_covariates, hyp_val_covariates = hyp_covariates[:-365*3], hyp_covariates[-365*3:]

        model_multi = BlockRNNModel(model="LSTM",input_chunk_length=365*3, output_chunk_length=365*3, n_epochs=10, random_state=0)
        model_multi.fit(series=[train_prod, train_hyp], past_covariates=[prod_train_covariates, hyp_train_covariates], verbose=True)

        pred_multi = model_multi.predict(n=365*3, series=train_prod, past_covariates=prod_train_covariates)
        series_prod_plot_scaled.plot(label='actual')
        pred_multi.plot(label='forecast')
        
        plt.legend()
        plt.show()
    return()

standardize(df)