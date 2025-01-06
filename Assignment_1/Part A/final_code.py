import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define start and end dates
start_date = datetime(2013, 12, 31)  # December 31, 2013
end_date = datetime(2023, 12, 31)    # December 31, 2023
#create a series with all dates 
dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

# Create a DataFrame with the dates
dates_df = pd.DataFrame({'observation_date': dates})
# Print the first few dates 
print("First few dates:")
for date in dates[:5]:
    print(date.strftime('%Y-%m-%d'))  # Format the date as YYYY-MM-DD

# Print the last few dates
print("\nLast few dates:")
for date in dates[-5:]:
    print(date.strftime('%Y-%m-%d'))  # Format the date as YYYY-MM-DD

#load excel files and create df
excel_file = "C:\\Users\\kwstasbenek\\Desktop\\Filis\\Part A\\Data-sets\\DCOILBRENTEU.xls" 
excel_df = pd.read_excel(excel_file, header=0)
#count n of na obs
num_na_oil= excel_df['DCOILBRENTEU'].isna().sum()
print(num_na_oil)
#replace na with 0 and the linearly interpolate
excel_df['DCOILBRENTEU'] =excel_df['DCOILBRENTEU'].replace(0, pd.NA)
excel_df['DCOILBRENTEU'] = excel_df['DCOILBRENTEU'].interpolate(method='linear')
#merge excel data df with date df on date
merged_df = dates_df.merge(excel_df, on='observation_date', how='left')
merged_df = merged_df.rename(columns={'DCOILBRENTEU': 'oil_price'})

#load second excel data file and basically do the same 
excel_2_file = "C:\\Users\\kwstasbenek\\Desktop\\Filis\\Part A\\Data-sets\\DHHNGSP.xls"  
excel_2_df = pd.read_excel(excel_2_file)
#count n of na obs
num_na_values_gas = excel_2_df['DHHNGSP'].isna().sum()
print(num_na_values_gas)
#replace na with 0 and the linearly interpolate
excel_2_df['DHHNGSP'] =excel_2_df['DHHNGSP'].replace(0, pd.NA)
excel_2_df['DHHNGSP'] = excel_2_df['DHHNGSP'].interpolate(method='linear')
#merge df's
final_df= merged_df.merge(excel_2_df, on='observation_date', how='left')
final_df = final_df.rename(columns={'DHHNGSP': 'natural_gas_price'})

# Drop all rows with at least one NA value
final_df.dropna(inplace=True)
# Compute summary statistics
summary_oil= final_df['oil_price'].describe()
print(summary_oil)
summary_gas= final_df['natural_gas_price'].describe()
print(summary_gas)
# Calculate the first log difference for oil price
first_log_diff_oil = np.diff(np.log(final_df['oil_price']))

# Set the first element as NA
first_log_diff_oil = np.insert(first_log_diff_oil, 0, np.nan, axis=0)

# Assign the first log difference for oil price to the DataFrame
final_df.loc[:, 'first_log_diff_oil'] = first_log_diff_oil
    
# Calculate the first log difference for natural gas price
first_log_diff_gas = np.diff(np.log(final_df['natural_gas_price']))

# Set the first element as NA
first_log_diff_gas= np.insert(first_log_diff_gas, 0, np.nan, axis=0)

# Assign the first log difference for natural gas price to the DataFrame
final_df.loc[:, 'first_log_diff_gas'] = first_log_diff_gas                         
#excel file of control variable 1
excel_file = "C:\\Users\\kwstasbenek\\Desktop\\Filis\\Part A\\Data-sets\\GECON_indicator.xlsx" 
excel_df_3 = pd.read_excel(excel_file, header=0)
# excel file of control variable 2
excel_file="C:\\Users\\kwstasbenek\\Desktop\\Filis\\Part A\\Data-sets\\WTI.xlsx"
excel_df_4 = pd.read_excel(excel_file, header=0)
# excel file for control variable 3
excel_file="C:\\Users\\kwstasbenek\\Desktop\\Filis\\Part A\\Data-sets\\USACPALTT01CTGYM.xls"
excel_df_5 = pd.read_excel(excel_file, header=0)

# drop observation date column
excel_df_3 = excel_df_3.drop('observation_date', axis=1)
excel_df_4 = excel_df_4.drop('observation_date', axis=1)
excel_df_5 = excel_df_5.drop('observation_date', axis=1)
# Calculate the daily returns for natural gas price and oil 
final_df['return_natural_gas_price'] = final_df['first_log_diff_gas'].pct_change()
final_df['return_natural_gas_price'] = final_df['first_log_diff_gas'].replace(0, np.nan)
final_df['return_oil_price'] = final_df['first_log_diff_oil'].pct_change()
final_df['return_oil_price'] = final_df['first_log_diff_oil'].replace(0, np.nan)

#set date as time index 
final_df.set_index('observation_date', inplace=True)
# Calculate the monthly returns for both assets
monthly_returns_oil = final_df['return_oil_price'].resample('M').mean()
monthly_returns_gas = final_df['return_natural_gas_price'].resample('M').mean()
# Calculate the monthly standard deviation for both assets
monthly_std_oil = final_df['return_oil_price'].resample('M').std()
monthly_std_gas = final_df['return_natural_gas_price'].resample('M').std()
# create new df with monthly returns for oil and gas 
new_df = pd.DataFrame({
    
    'oil_monthly_std': monthly_std_oil.values,
    'gas_monthly_std': monthly_std_gas.values
})
# Annualize the monthly standard deviation
new_df['oil_annualized_volatility'] = new_df['oil_monthly_std'] * np.sqrt(12)
new_df['gas_annualized_volatility'] = new_df['gas_monthly_std'] * np.sqrt(12)
#drop NA row
new_df = new_df.drop(new_df.index[0])
#create df with both the volatilites and the 2 contol variables 
merged_df_2 = new_df.merge(excel_df_3, left_index=True, right_index=True, how='left')
merged_df_2 = merged_df_2.rename(columns={'GECON_indicator': 'GECON'})
merged_df_3= merged_df_2.merge(excel_df_4, left_index=True,right_index=True, how='left')
merged_df_3= merged_df_3.rename(columns={'WTI': 'WIP'})
merged_df_4=merged_df_3.merge(excel_df_5, left_index=True,right_index=True, how='left')
#drop first 2 row containing stdv's
merged_df_4 = merged_df_4.iloc[:, 2:]

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
print(merged_df_4.describe())
#import necesary libraries
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculate the correlation matrix
correlation_matrix = merged_df_4[['gas_annualized_volatility', 'oil_annualized_volatility', 'GECON','CPI','WIP'  ]].corr()

# Display the correlation matrix
# Set display options to show all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Print the correlation matrix
print(correlation_matrix)
merged_df_4 = merged_df_4.dropna()
independent_vars=merged_df_4[['gas_annualized_volatility', 'oil_annualized_volatility', 'GECON','CPI','WIP'] ]
#Add a constant column to the independent variables (required for VIF calculation)
ndependent_vars = sm.add_constant(independent_vars)

# Calculate VIF for each independent variable
vif = pd.DataFrame()
vif["Variable"] = independent_vars.columns
vif["VIF"] = [variance_inflation_factor(independent_vars.values, i) for i in range(independent_vars.shape[1])]

# Print the VIF values
print(vif)
# import library for graph
import matplotlib.pyplot as plt

# Plot oil annualized volatility
plt.plot(new_df.index, new_df['oil_annualized_volatility'], label='Oil')

# Plot natural gas annualized volatility
plt.plot(new_df.index, new_df['gas_annualized_volatility'], label='Natural Gas')

# Add legend and labels
plt.legend()
plt.xlabel('t')
plt.ylabel('Annualized Volatility')

# Show the plot
plt.show()


# Set the number of observations as the index
new_df['n_obs'] = range(1, len(new_df) + 1)
new_df.set_index('n_obs', inplace=True)
#import library for ADF test
from statsmodels.tsa.stattools import adfuller

result = adfuller(new_df['oil_annualized_volatility'])

# Extracting and printing the test statistic
print('ADF Statistic oil:', result[0])

# Extracting and printing the p-value
print('p-value oil:', result[1])
# Assuming your time series data is stored in a pandas Series named 'ts_data'
# Replace 'ts_data' with the name of your time series data
result = adfuller(new_df['gas_annualized_volatility'])

# Extracting and printing the test statistic
print('ADF Statistic gas:', result[0])

# Extracting and printing the p-value
print('p-value gas:', result[1])
# Set the frequency
new_df.index.freq = 'M'
merged_df_4.index.freg='M'

import statsmodels.api as sm

# Define dependent and independent variables
X = merged_df_4[['gas_annualized_volatility', 'GECON', 'CPI']]
y = new_df['oil_annualized_volatility']

# Add constant term
X = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X).fit()

# Print summary of regression results
print(model.summary())
from statsmodels.tsa.arima.model import ARIMA

# Define the ARIMA model (1,0,1)
model = ARIMA(new_df['oil_annualized_volatility'], order=(1,0,1), exog=merged_df_4[['gas_annualized_volatility', 'GECON', 'CPI']])

# Fit the ARIMA model
model_fit = model.fit()

# Print the summary of the model
print(model_fit.summary())

