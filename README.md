
# Removing Trends - Lab

## Introduction

In this lab, you'll practice your detrending skills!

## Objectives

In this lab you will: 

- Use a log transformation to minimize non-stationarity 
- Use rolling means to reduce non-stationarity 
- Use differencing to reduce non-stationarity 
- Use rolling statistics as a check for stationarity 
- Create visualizations of transformed time series as a visual aid to determine if stationarity has been achieved 
- Use the Dickey-Fuller test and conclude whether or not a dataset is exhibiting stationarity 


## Detrending the Air passenger data 

In this lab you will work with the air passenger dataset available in `'passengers.csv'`. First, run the following cell to import the necessary libraries. 


```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
%matplotlib inline
```

- Import the `'passengers.csv'` dataset 
- Change the data type of the `'Month'` column to a proper date format 
- Set the `'Month'` column as the index of the DataFrame 
- Print the first five rows of the dataset 


```python
# Import 'passengers.csv' dataset
data = pd.read_csv('passengers.csv')

# Change the data type of the 'Month' column
data['Month'] = pd.to_datetime(data['Month'])

# Set the 'Month' column as the index
ts = data.set_index('Month')

# Print the first five rows
ts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#Passengers</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1949-01-01</td>
      <td>112</td>
    </tr>
    <tr>
      <td>1949-02-01</td>
      <td>118</td>
    </tr>
    <tr>
      <td>1949-03-01</td>
      <td>132</td>
    </tr>
    <tr>
      <td>1949-04-01</td>
      <td>129</td>
    </tr>
    <tr>
      <td>1949-05-01</td>
      <td>121</td>
    </tr>
  </tbody>
</table>
</div>



Plot this time series. 


```python
# Plot the time series
ts.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbc221e44e0>




![png](index_files/index_6_1.png)


## Create a stationarity check

Your next task is to use the code from previous labs to create a function `stationarity_check()` that takes in a time series and performs stationarity checks including rolling statistics and the Dickey-Fuller test. 

We want the output of the function to: 

- Plot the original time series along with the rolling mean and rolling standard deviation (use a window of 8) in one plot 
- Output the results of the Dickey-Fuller test 


```python
# Create a function to check for the stationarity of a given time series using rolling stats and DF test
# Collect and package the code from previous labs
def stationarity_check(data):
    roll_mean = data.rolling(window=8, center=False).mean()
    roll_std = data.rolling(window=8, center=False).std()
    
    fig = plt.figure(figsize=(16,6))
    plt.plot(data, color='blue', label="original")
    plt.plot(roll_mean, color='red', label="rolling mean")
    plt.plot(roll_std, color='black', label='rolling sd')
    plt.legend(loc='best')
    plt.title("Data, rolling mean & std")
    
    adf_test = adfuller(data['#Passengers'])
    adf_output = pd.Series(adf_test[0:4], index=["Test Statistic", 'p-value', '#Lags used', 'Number of obs used'])
    print(adf_output)
    return None
```

Use your newly created function on the `ts` timeseries. 


```python
# Code here
stationarity_check(ts)
```

    Test Statistic          0.815369
    p-value                 0.991880
    #Lags used             13.000000
    Number of obs used    130.000000
    dtype: float64



![png](index_files/index_10_1.png)


## Perform a log and square root transform

Plot a log transform of the original time series (`ts`). 


```python
# Plot a log transform
ts_log = np.log(ts)
plt.plot(ts_log);
```


![png](index_files/index_12_0.png)


Plot a square root  transform of the original time series (`ts`). 


```python
# Plot a square root transform
plt.plot(np.sqrt(ts));
```


![png](index_files/index_14_0.png)


Going forward, let's keep working with the log transformed data before subtracting rolling mean, differencing, etc.

## Subtracting the rolling mean

Create a rolling mean using your log transformed time series, with a time window of 7. Plot the log-transformed time series and the rolling mean together.


```python
# your code here
roll_mean = ts_log.rolling(window=7, center=False).mean()
fig = plt.figure(figsize=(11,7)) 
plt.plot(ts_log, color='blue', label='Data log transform')
plt.plot(roll_mean, color='red', label='Data rolling mean');
```


![png](index_files/index_18_0.png)


Now, subtract this rolling mean from the log transformed time series, and look at the 10 first elements of the result.  


```python
# Subtract the moving average from the log transformed data
data_minus_roll_mean = ts_log - roll_mean

# Print the first 10 rows
data_minus_roll_mean.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#Passengers</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1949-01-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1949-02-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1949-03-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1949-04-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1949-05-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1949-06-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1949-07-01</td>
      <td>0.150059</td>
    </tr>
    <tr>
      <td>1949-08-01</td>
      <td>0.110242</td>
    </tr>
    <tr>
      <td>1949-09-01</td>
      <td>0.005404</td>
    </tr>
    <tr>
      <td>1949-10-01</td>
      <td>-0.113317</td>
    </tr>
  </tbody>
</table>
</div>



Drop the missing values from this time series. 


```python
# Drop the missing values
data_minus_roll_mean.dropna(inplace=True)
```

Plot this time series now. 


```python
# Plot the result
data_minus_roll_mean.plot();
```


![png](index_files/index_24_0.png)


Finally, use your function `check_stationarity()` to see if this series is stationary!


```python
# Your code here
stationarity_check(data_minus_roll_mean)
```

    Test Statistic         -2.348027
    p-value                 0.156946
    #Lags used             14.000000
    Number of obs used    123.000000
    dtype: float64



![png](index_files/index_26_1.png)


### Based on the visuals and on the Dickey-Fuller test, what do you conclude?


# Your conclusion here
P value is greater than 0.05, so we cannot reject the null hypothesis that there is no stationarity to this data. Meaning, that this data is not stationary despite the transformations visual above.

## Subtracting the weighted rolling mean

Repeat all the above steps to calculate the exponential *weighted* rolling mean with a halflife of 4. Start from the log-transformed data again. Compare the Dickey-Fuller test results. What do you conclude?


```python
# Calculate Weighted Moving Average of log transformed data
exp_roll_mean = ts_log.ewm(halflife=2).mean()

# Plot the original data with exp weighted average
fig = plt.figure(figsize=(11,5))
plt.plot(ts_log, color='blue', label='Log transformed data')
plt.plot(exp_roll_mean, color='red', label='Weighted moving average')
plt.title("Log transformed & Weighted moving average")
plt.legend(loc='best')
plt.show();
```


![png](index_files/index_31_0.png)


- Subtract this exponential weighted rolling mean from the log transformed data  
- Print the resulting time series 


```python
# Subtract the exponential weighted rolling mean from the original data 
data_minus_exp_roll_mean = ts_log - exp_roll_mean

# Plot the time series
data_minus_exp_roll_mean.plot();
```


![png](index_files/index_33_0.png)


Check for stationarity of `data_minus_exp_roll_mean` using your function. 


```python
# Do a stationarity check
stationarity_check(data_minus_exp_roll_mean)
```

    Test Statistic         -3.087696
    p-value                 0.027477
    #Lags used             13.000000
    Number of obs used    130.000000
    dtype: float64



![png](index_files/index_35_1.png)


### Based on the visuals and on the Dickey-Fuller test, what do you conclude?


# Your conclusion here
Yes, this data is staionary, p-value is 0.02, and the test statistic is negative, therefore we can strongly reject the null hypothesis (that it is not stationary).

## Differencing

Using exponentially weighted moving averages, we seem to have removed the upward trend, but not the seasonality issue. Now use differencing to remove seasonality. Make sure you use the right amount of `periods`. Start from the log-transformed, exponentially weighted rolling mean-subtracted series.

After you differenced the series, drop the missing values, plot the resulting time series, and then run the `stationarity check()` again.


```python
# Difference your data
data_diff = data_minus_exp_roll_mean.diff(periods=12)

# Drop the missing values
data_diff.dropna(inplace=True)

# Check out the first few rows
data_diff.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#Passengers</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1950-01-01</td>
      <td>-0.037016</td>
    </tr>
    <tr>
      <td>1950-02-01</td>
      <td>0.016679</td>
    </tr>
    <tr>
      <td>1950-03-01</td>
      <td>0.033226</td>
    </tr>
    <tr>
      <td>1950-04-01</td>
      <td>0.013826</td>
    </tr>
    <tr>
      <td>1950-05-01</td>
      <td>-0.001451</td>
    </tr>
    <tr>
      <td>1950-06-01</td>
      <td>0.049340</td>
    </tr>
    <tr>
      <td>1950-07-01</td>
      <td>0.067287</td>
    </tr>
    <tr>
      <td>1950-08-01</td>
      <td>0.049525</td>
    </tr>
    <tr>
      <td>1950-09-01</td>
      <td>0.042844</td>
    </tr>
    <tr>
      <td>1950-10-01</td>
      <td>0.001584</td>
    </tr>
    <tr>
      <td>1950-11-01</td>
      <td>-0.014139</td>
    </tr>
    <tr>
      <td>1950-12-01</td>
      <td>0.045790</td>
    </tr>
    <tr>
      <td>1951-01-01</td>
      <td>0.075227</td>
    </tr>
    <tr>
      <td>1951-02-01</td>
      <td>0.012694</td>
    </tr>
    <tr>
      <td>1951-03-01</td>
      <td>0.050702</td>
    </tr>
  </tbody>
</table>
</div>



Plot the resulting differenced time series. 


```python
# Plot your differenced time series
stationarity_check(data_diff)
```

    Test Statistic         -4.158133
    p-value                 0.000775
    #Lags used             12.000000
    Number of obs used    119.000000
    dtype: float64



![png](index_files/index_42_1.png)



```python
# Perform the stationarity check
```

### Your conclusion

# Your conclusion here
looks a lot less seasonal, and stats show it is stationary.

## Summary 

In this lab, you learned how to make time series stationary through using log transforms, rolling means, and differencing.
