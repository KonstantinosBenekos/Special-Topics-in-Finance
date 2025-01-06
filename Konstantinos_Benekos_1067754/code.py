import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
#load relevant data
df1= pd.read_csv('C:\\Users\\kwstasbenek\\Desktop\\fillis_2\\Data_sets\\DEM.csv', engine='python')
df1=df1.rename(columns={'IRLTLT01DEM156N': 'DEm_1Oyear'})
df2= pd.read_csv('C:\\Users\\kwstasbenek\\Desktop\\fillis_2\\Data_sets\\GRM.csv', engine='python')
df2=df2.rename(columns={'IRLTLT01GRM156N' : 'GRm_10year'})

merged_df=df1.merge(df2, on='DATE', how='left')
# Select numeric columns
numeric_cols = ['DEm_1Oyear', 'GRm_10year']

# Convert columns to numeric types
for col in numeric_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Interpolate missing values
merged_df[numeric_cols] = merged_df[numeric_cols].interpolate()
#calculate risk premum proxy
merged_df['risk_premium']=merged_df['GRm_10year']-merged_df['DEm_1Oyear']
#loadYrther data sets
df3=pd.read_csv('C:\\Users\\kwstasbenek\\Desktop\\fillis_2\\Data_sets\\GRQ.csv', engine='python')
# Select columns that you want to interpolate
numeric_cols = ['GRm_3month']
df3['DATE'] = pd.to_datetime(df3['DATE'])
#merge relevant data
merged_df['DATE']=pd.to_datetime(merged_df['DATE'])
merged_df=merged_df.merge(df3, on='DATE', how='left')
merged_df = merged_df.dropna(axis=1, how='all')

#calculate yield spread between maturities
merged_df['Yield_Spread']=merged_df['GRm_10year']-merged_df['GRm_3month']

#independetn variables df
inde_df = pd.DataFrame(merged_df['DATE'])
df4=pd.read_csv('C:\\Users\\kwstasbenek\\Desktop\\fillis_2\\Data_sets\\GR_gdp.csv', engine='python')
numeric_cols = ['GR_gdp']
df4['DATE'] = pd.to_datetime(df4['DATE'])
inde_df=inde_df.merge(df4, on='DATE', how='left')
# Fill any remaining NaN values 
inde_df['GR_gdp'] = inde_df['GR_gdp'].interpolate(method='linear')

df5=pd.read_csv('C:\\Users\\kwstasbenek\\Desktop\\fillis_2\\Data_sets\\BBKMGDP.csv', engine='python')
numeric_cols=['US_GDP_Growth']
df5['DATE'] = pd.to_datetime(df5['DATE'])
inde_df=inde_df.merge(df5, on='DATE', how='left')
inde_df['US_GDP_Growth'] = inde_df['US_GDP_Growth'].interpolate(method='linear')

df6=pd.read_csv('C:\\Users\\kwstasbenek\\Desktop\\fillis_2\\Data_sets\\CPI_gr.csv', engine='python')
numeric_cols=['CPI_gr']
df6['DATE'] = pd.to_datetime(df6['DATE'])
inde_df=inde_df.merge(df6, on='DATE', how='left')

df7=pd.read_csv('C:\\Users\\kwstasbenek\\Desktop\\fillis_2\\Data_sets\\PPI_gr.csv', engine='python')
numeric_cols=['PPI_gr']
df7['DATE'] = pd.to_datetime(df7['DATE'])
inde_df=inde_df.merge(df7, on='DATE', how='left')
inde_df['PPI_gr'] = inde_df['PPI_gr'].interpolate(method='linear')

df8=pd.read_csv('C:\\Users\\kwstasbenek\\Desktop\\fillis_2\\Data_sets\\net_migration.csv', engine='python')
numeric_cols=['Net_migration']
df8['DATE'] = pd.to_datetime(df8['DATE'])
inde_df=inde_df.merge(df8, on='DATE', how='left')

#df9=pd.read_csv('C:\\Users\\kwstasbenek\\Desktop\\fillis_2\\Data_sets\\Unemp.csv', engine='python')
#numeric_cols=['Unemp']
#df9['DATE'] = pd.to_datetime(df9['DATE'])
#inde_df=inde_df.merge(df9, on='DATE', how='left')


# Fit the linear regression model
X = inde_df.iloc[:, 1:]  # all columns except the first one ('DATE')
y = merged_df['risk_premium']
X = sm.add_constant(X)  # add a constant (intercept) to the independent variables
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

# Calculate the correlation matrix
inde_df = inde_df.iloc[:, 1:]  # Removing the 'DATE' column if it exists
corr_matrix = np.corrcoef(inde_df.values.T)
print("Correlation Matrix:")
print(pd.DataFrame(corr_matrix, columns=inde_df.columns, index=inde_df.columns))

df10=pd.read_csv('C:\\Users\\kwstasbenek\\Desktop\\fillis_2\\Data_sets\\Industrial_production.csv', engine='python')
numeric_cols=['IP']
df10['DATE'] = pd.to_datetime(df10['DATE'])
inde_df_2 = df10.copy()

inde_df_2=inde_df_2.merge(df6, on='DATE', how='left')

df11=pd.read_csv('C:\\Users\\kwstasbenek\\Desktop\\fillis_2\\Data_sets\\PCE.csv', engine='python')
numeric_cols=['PCE']
df11['DATE'] = pd.to_datetime(df11['DATE'])
inde_df_2=inde_df_2.merge(df11, on='DATE', how='left')
inde_df_2['PCE'] = inde_df_2['PCE'].interpolate(method='linear')

inde_df_2 = inde_df_2.merge(merged_df[['DATE', 'GRm_3month']], on='DATE', how='left')

# Fit the linear regression model
X = inde_df_2.iloc[:, 1:-1]  # all columns except the first one ('DATE') and the last one

y = merged_df['Yield_Spread']
X = sm.add_constant(X)  # add a constant (intercept) to the independent variables
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

# Calculate the correlation matrix
inde_df_2 = inde_df_2.iloc[:, 1:-1]  # Removing the 'DATE' column if it exists
corr_matrix = np.corrcoef(inde_df_2.values.T)
print("Correlation Matrix:")
print(pd.DataFrame(corr_matrix, columns=inde_df_2.columns, index=inde_df_2.columns))
#create plot
plt.figure(figsize=(10,6))
sns.lineplot(x="DATE", y="GRm_10year", data=merged_df, label="GRm_10year", linewidth=2, linestyle="-")
sns.lineplot(x="DATE", y="DEm_1Oyear", data=merged_df, label="DEm_1Oyear", linewidth=2, linestyle="-")

# Add y-axis label
plt.ylabel("Yield")

# Add a red horizontal line at y=0
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)

# Customize the legend
plt.legend(loc="upper right", fontsize=12)

# Add a title (optional)
plt.title("Comparison of German 10-Year Sovereign T-Bill vs. Greek 10-Year Sovereign T-Bill ")

# Show the plot
plt.show()
plt.figure(figsize=(10,6))
sns.lineplot(x="DATE", y="risk_premium", data=merged_df, label="GRm_10year", linewidth=2, linestyle="-")
# Add y-axis label
plt.ylabel("risk premium ")

# Add a red horizontal line at y=0
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)

# Customize the legend
plt.legend(loc="upper right", fontsize=12)

# Add a title (optional)
plt.title("Risk premium for Greece's 10 year sovereign T-bill ")
plt.show()

# Create a table with descriptive statistics for all variables in inde_df and merged_df['risk_premium']
stats_table = pd.concat([inde_df.describe().loc[['count', 'mean', 'std', 'min', 'max']], 
                         merged_df['risk_premium'].describe().loc[['count', 'mean', 'std', 'min', 'max']]], axis=1)
# Transpose the table
pivoted_table = stats_table.transpose()

# Print the pivoted table in full
print(pivoted_table.to_string())

plt.figure(figsize=(10,6))
sns.lineplot(x="DATE", y="GRm_10year", data=merged_df, label="GRm_10year", linewidth=2, linestyle="-")
sns.lineplot(x="DATE", y="GRm_3month", data=merged_df, label="GRm_3month", linewidth=2, linestyle="-")

# Add y-axis label
plt.ylabel("Yield")

# Add a red horizontal line at y=0
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)

# Customize the legend
plt.legend(loc="upper right", fontsize=12)

# Add a title (optional)
plt.title("Comparison of Greek 10-Year Sovereign T-Bill vs. Greek 3-month Sovereign T-Bill ")

# Show the plot
plt.show()

plt.figure(figsize=(10,6))
sns.lineplot(x="DATE", y="Yield_Spread", data=merged_df, label="Yield_Spread", linewidth=2, linestyle="-")
# Add y-axis label
plt.ylabel("Yield_Spread")

# Add a red horizontal line at y=0
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)

# Customize the legend
plt.legend(loc="upper right", fontsize=12)

# Add a title (optional)
plt.title("Yield spread between maturities ")
plt.show()

# Create a table with descriptive statistics for all variables in inde_df and merged_df['risk_premium']
stats_table = pd.concat([inde_df_2.describe().loc[['count', 'mean', 'std', 'min', 'max']], 
                         merged_df['Yield_Spread'].describe().loc[['count', 'mean', 'std', 'min', 'max']]], axis=1)
# Transpose the table
pivoted_table = stats_table.transpose()

# Print the pivoted table in full
print(pivoted_table.to_string())

merged_df_head = merged_df.head(40)
inde_df_head=inde_df.head(40)
X = inde_df_head  # all columns except the first one ('DATE')
y = merged_df_head['risk_premium']
X = sm.add_constant(X)  # add a constant (intercept) to the independent variables
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())


# Print the model summary
print(model.summary())

# Set the style for the plot
sns.set(style="whitegrid")

# Plotting the yield curve with two maturities
maturities = ['3-Month', '10-Year']
average_yields = [merged_df['GRm_3month'].mean(), merged_df['GRm_10year'].mean()]

plt.figure(figsize=(10, 6))

# Scatter plot for the two points
plt.scatter(maturities, average_yields, color='blue', s=100, label='Average Yield')

# Connecting the two points with a line
plt.plot(maturities, average_yields, linestyle='--', color='red')

# Adding titles and labels
plt.title('Yield Curve Comparison: 3-Month vs 10-Year', fontsize=16)
plt.xlabel('Maturity', fontsize=14)
plt.ylabel('Yield (%)', fontsize=14)

# Customize the ticks on x and y axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adding legend
plt.legend(fontsize=12)

# Adding annotations for the points
for i, txt in enumerate(average_yields):
    plt.annotate(f'{txt:.2f}%', (maturities[i], average_yields[i]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

# Display the plot
plt.show()
