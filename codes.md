# Liteshwar Rao
## Assignment 
## Statistical analysis of JP Morgan Stock
```python
# importing the packages
import time
import numpy as np
import datetime
from pandas_datareader import data as web
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
%matplotlib inline

import math
```
```python
import warnings
warnings.filterwarnings("ignore")
```
```python
# loading the JP Morgan closing stock price data
df1 = pd.read_excel('JPM.xlsx')
# converting the date column to a format readable in pandas
pd.to_datetime(df1['Date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
# checking the number of datapoints
print('There are {} number of days in the dataset.'.format(df1.shape[0]))
```
```python
# creating a division for training and testing data
df1['Date'] = df1['Date'].apply(pd.Timestamp)
# plot the stock prices for the last nine years. The dashed vertical line represents the separation between training and test data.
plt.figure(figsize=(14, 5), dpi=100)
plt.plot(df1['Date'], df1['JPM'], label='JP Morgan stock')
#plt.axvline(datetime.date(2016-4-20),0,270, linestyles='--', colors='gray', label='Train/Test data cut-off')
plt.axvline(pd.Timestamp('2016-4-20'),color='gray',linestyle='--',label='Train/Test data cut-off' )
plt.xlabel('Date')
plt.ylabel('USD')
plt.title('Figure 1: JP Morgan stock price')
plt.legend()
plt.show()
```
```python
# checking the number of testing and training data points
num_training_days = int(df1.shape[0]*.7)
print('Number of training days: {}. Number of test days: {}.'.format(num_training_days, df1.shape[0]-num_training_days))
```
```python
# defining the technical indicators of JPM
def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['JPM'].rolling(window=7).mean()
    dataset['ma21'] = dataset['JPM'].rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['JPM'].ewm(span=26).mean()
    dataset['12ema'] = dataset['JPM'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    
    #Create Bollinger Bands
    dataset['20sd'] = (dataset['JPM'].rolling(window=20).std())
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset['JPM'].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = dataset['JPM']-1
    
    return dataset

```
```python
# executing the above function
dataset_TI_df = get_technical_indicators(df1[['JPM']])
```
```python
# plotting the technical indicators
def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days
    
    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)
    
    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(dataset['JPM'],label='Closing Price', color='b')
    plt.plot(dataset['ma21'],label='MA 21', color='r',linestyle='--')
    plt.plot(dataset['upper_band'],label='Upper Band', color='c')
    plt.plot(dataset['lower_band'],label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for JP Morgan - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()
    
    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'],label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['momentum'],label='Momentum', color='b',linestyle='-')

    plt.legend()   
    plt.show()
```
```python
# executing the plotting function
plot_technical_indicators(dataset_TI_df, 400)
```
## applying the Fourier transformation
$$G(f) = \int\limits_{- \infty}^{\infty}g(t) e^{-i2 \pi ft}dt $$

```python
# defining the data
data_FT = df1[['Date', 'JPM']]
# applying the Fourier transform
close_fft = np.fft.fft(np.asarray(data_FT['JPM'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
```
```python
# plotting the Fourier transform
plt.figure(figsize=(14, 7), dpi=100)
fft_list = np.asarray(fft_df['fft'].tolist())
for num_ in [3, 6, 9, 100]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
    plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
plt.plot(data_FT['JPM'],  label='Real')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 3: JP Morgan (close) stock prices & Fourier transforms')
plt.legend()
plt.show()
```
## Running time series analysis

```python
# importing the package
from statsmodels.tsa.arima.model import ARIMA
# running the ARIMA model with lags = 5, differencing = 1, moving average = 0
series = data_FT['JPM']
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())
```
```python
# plotting the autocorreation
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)
plt.figure(figsize=(10, 7), dpi=80)
plt.show() 
```
```python
# calculating mean square error
from sklearn.metrics import mean_squared_error
# calculating mean square error between predicted and actual value
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
# printing the mean square error
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
```
```python
# Plot the predicted (from ARIMA) and actual prices

plt.figure(figsize=(12, 6), dpi=100)
plt.plot(test, label='Real')
plt.plot(predictions, color='red', label='Predicted')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 5: ARIMA model on JPM stock')
plt.legend()
plt.show()
```
## Running statistical tests
```python
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
```
## Checking for Heteroscedasticity
```python
# Running Breusch-Pagan test for Heteroscedasticity
# Extract the 'JPM' column as a pandas Series
series = series

# Create a time index if necessary
# series.index = pd.to_datetime(series.index)

# Fit an OLS regression model
X = pd.Series([i for i in range(len(series))])
X = sm.add_constant(X)
model = OLS(series, X)
results = model.fit()

# Perform the Breusch-Pagan test
name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
bp_test = het_breuschpagan(results.resid, X)
results_summary = pd.DataFrame(lzip(name, bp_test), columns=['Test', 'Value'])
print(results_summary)
```
### since $p$ value is less than any significance levels of $5 \%, 1 \%, 10 \%$  the residuals are not distributed with equal variance.

## checking for autocorrelation
```python
from statsmodels.graphics.tsaplots import plot_acf
# Calculate the autocorrelation function (ACF)
jpm_column = df1['JPM']
acf = plot_acf(jpm_column, lags=20)

# Customize the plot
plt.title('Autocorrelation Plot - JPM')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()
```
### the series of JPM is autocorrelated
