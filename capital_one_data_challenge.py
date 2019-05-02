#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 23:19:11 2019

@author: clo
"""

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
# Matplotlib for additional customization
from matplotlib import pyplot as plt
%matplotlib inline

# Seaborn for plotting and styling
import seaborn as sns
sns.set()
#%%
# all the customerized functions that are being used in the analysis can be found in this section
# build a function to convert latitude and longtitude to zipcode
def convert_zipcode(lat_long):
    geolocator = Nominatim()
    location = geolocator.reverse(lat_long)
    return location.raw['address']['postcode']

# Since the values of price are charater with a '$' in the front, 
# we need to convert it to be numeric
def clean_text(text):
    text = ''.join(e for e in text if e.isalnum())
    num = int(text[:-2])
    return num 

def remove_outliers(column, df):
    Q1 = df['column'].quantile(0.25)
    Q3 = df['column'].quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df['column'] < (Q1 - 1.5 * IQR)) |(df['column'] > (Q3 + 1.5 * IQR)))]
    return df

#%%
# load the data
df_z = pd.read_csv('Zip_Zhvi_2bedroom.csv')
df_abn = pd.read_csv('listings.csv')
#%%
# check the numbers of rows and columns in original files
print (df_z.shape)
print (df_abn.shape)
#%% 
# Zellow Data
# just keep the properties within New York City
df_z = df_z[df_z.City == 'New York']
# rename the columns in zellow data 
df_z.rename(columns={'RegionName':'zipcode'}, inplace=True)
df_z.set_index(keys='zipcode', inplace=True)
df_z.head(5)
#%%
# Zellow Time Series Plot
df_z_transposed = df_z.transpose()
## we just want to keep the past 10 years data
df_z_transposed_10 = df_z_transposed.iloc[-120:]
df_z_transposed_10.reset_index(inplace=True)
df_z_transposed_10.rename(columns={'index':'month'}, inplace=True)
df_z_transposed_10.head(10)
# convert it to the desired data framn for time series plot
df_z_transposed_10.month = pd.to_datetime(df_z_transposed_10.month)
df_z_transposed_10.set_index(keys='month', inplace=True)
df_z_transposed_10.plot(figsize=(20,10), fontsize=10)
plt.xlabel('Year', fontsize=10)
#%%
# AirBnB Data
# Since there are over 95 features in airbnb set, we first need to specify which features we are most interested in 
# based on our domain knowledge and assumptions
# only extract 2-bedroom listing
df_abn = df_abn[df_abn.bedrooms == 2]
airbnb_important_features = ['neighbourhood', 'neighbourhood_group_cleansed','zipcode', 'latitude', 'longitude', 'bedrooms', 'square_feet', 'price', \
                             'security_deposit', 'cleaning_fee', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', \
                             'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'calculated_host_listings_count',\
                             'reviews_per_month']
df_abn = df_abn.loc[:,airbnb_important_features]
print(df_abn.shape)
#%%
# AirBnb Data Preprocessing - Zipcode
# unify zipcode to five digit 
df_abn['zipcode'] = df_abn['zipcode'].astype('str').apply(lambda x:x[:5])
# there are 62 listings with null zipcode
print(len(df_abn[df_abn.zipcode=='nan']))
print(df_abn.shape)
#%%
# split the airbnb data into missing and clean set and further fill in zipcode based on the lat and lon
df_abn_missing = df_abn[df_abn['zipcode'].isin(['nan'])]
df_abn_clean = df_abn[~df_abn['zipcode'].isin(['nan'])]

# check how many obs with no zipcodes
print(len(df_abn_missing.zipcode))
#%%
df_abn_missing['lat_long'] = df_abn_missing.apply(lambda row: '{}, {}'.format(row['latitude'], row['longitude']), axis=1)
df_abn_missing.zipcode = df_abn_missing.lat_long.apply(lambda x: convert_zipcode(x))
# after we convert latitude and longitude to zipcode for missing zip code 
# we combine the clean and missing set that we split earlier
df_abn_missing.drop(columns='lat_long', inplace=True)
airbnb_clean = df_abn_clean.append(df_abn_missing)
# we can notice that there are no missing zipcode now
len(airbnb_clean[airbnb_clean.zipcode=='nan'])
#%%
# AirBnb Data Preprocessing - Price
# clean the text in price field
airbnb_clean['price'] = airbnb_clean['price'].apply(lambda x: clean_text(x))
print(airbnb_clean.price.describe())
#%% 
# Outliers Analysis
# A scatter plot is used for detecting outliers
# https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
fig, ax = plt.subplots(figsize=(20,12))
ax.scatter(airbnb_clean['zipcode'], airbnb_clean['price'])
ax.set_xlabel('zipcode')
ax.set_ylabel('price')
plt.show()

# use IQR score to remove outliers
Q1 = airbnb_clean.price.quantile(0.25)
Q3 = airbnb_clean.price.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

airbnb_clean_iqr = airbnb_clean[~((airbnb_clean.price < (Q1 - 1.5 * IQR)) |(airbnb_clean.price > (Q3 + 1.5 * IQR)))]
airbnb_clean_iqr.shape

fig, ax = plt.subplots(figsize=(20,12))
ax.scatter(airbnb_clean_iqr['zipcode'], airbnb_clean_iqr['price'])
ax.set_xlabel('zipcode')
ax.set_ylabel('price')
plt.show()
#%%
# EDA
# Pairplot
sns.pairplot(airbnb_clean_iqr[['square_feet', 'price', \
                             'security_deposit', 'cleaning_fee', 'number_of_reviews', 'review_scores_rating',\
                             'review_scores_location', 'review_scores_value', 'calculated_host_listings_count',\
                             'reviews_per_month']])

airbnb_clean_iqr_rm = airbnb_clean_iqr[~(airbnb_clean_iqr.square_feet >= airbnb_clean_iqr.square_feet.max())]                                                     
sns.pairplot(airbnb_clean_iqr_rm[['square_feet', 'price', \
                             'security_deposit', 'cleaning_fee', 'number_of_reviews', 'review_scores_rating',\
                             'review_scores_location', 'review_scores_value', 'calculated_host_listings_count',\
                             'reviews_per_month']])
#%%
# correlation heat map
f, ax = plt.subplots(figsize=(20, 12))
airbnb_clean_iqr_plot = airbnb_clean_iqr_rm[['square_feet', 'price', \
                             'security_deposit', 'cleaning_fee', 'number_of_reviews', 'review_scores_rating',\
                             'review_scores_location', 'review_scores_value', 'calculated_host_listings_count',\
                             'reviews_per_month']]
r = airbnb_clean_iqr_plot.corr()
sns.heatmap(r, annot=True)
#%% 
# Box Plot 
price_df = airbnb_clean_iqr_rm[['neighbourhood_group_cleansed', 'price', 'zipcode']]
price_df_g = price_df.groupby(['neighbourhood_group_cleansed', 'zipcode'], as_index=False).size().reset_index(name='counts')

f, ax = plt.subplots(figsize=(15, 12))
sns.boxplot(x="neighbourhood_group_cleansed", y="price",
            palette='vlag',
            data=airbnb_clean_iqr, whis='range')
sns.despine(offset=10, trim=True)
#%%
# Check how many obs in each class
price_df_obs = price_df.groupby(['neighbourhood_group_cleansed','zipcode'], as_index=False).size().reset_index(name='counts')
price_df_obs.zipcode = price_df_obs.zipcode.apply(lambda x: int(x))

price_df_obs.neighbourhood_group_cleansed.value_counts()

#%%
# Bar Plot
group_order = ['Manhattan', 'Queens', 'Brooklyn', 'Bronx', 'Staten Island']

sns.palplot(sns.color_palette("husl", 5))
f, ax = plt.subplots(figsize=(15, 8))
sns.countplot(airbnb_clean_iqr['neighbourhood_group_cleansed'], palette='husl',hue_order=group_order)

f, ax = plt.subplots(figsize=(20, 12))
sns.barplot(x='zipcode', y='counts',hue='neighbourhood_group_cleansed', data=test, palette='husl', hue_order=group_order)


#%%
# GIS plot 
import gmaps
import gmaps.datasets
gmaps.configure(api_key="AIzaSyCe-6ATAGj_NMTqWdQie0NhQMzFhZw8lzY") # Your Google API key
map_list = list(zip(airbnb_clean_iqr.latitude, airbnb_clean_iqr.longitude))
# airbnb_clean[['latitude','longitude']].apply(lambda x: ','.join(x), axis=1)
#df['period'] = df[['Year', 'quarter']].apply(lambda x: ''.join(x), axis=1)
fig = gmaps.figure()
fig.add_layer(gmaps.heatmap_layer(map_list))
fig
#%%
# Combine Cost and Revenue Data
df_z_reset = df_z.reset_index()
cost_revenue = pd.merge(df_z_reset,price_df_obs, how='left', on='zipcode')

price_df = airbnb_clean_iqr[['neighbourhood_group_cleansed', 'price', 'zipcode']]
price_df_g = price_df.groupby(['neighbourhood_group_cleansed', 'zipcode'], as_index=False).size().reset_index(name='counts')
#%%
aggregation = {
      'square_feet': {'square_feet_median':'median'}
    , 'price':{'price_mediam':'median'}
    , 'number_of_reviews':{'number_of_reviews_total':'sum'}
    , 'review_scores_location':{'review_scores_location_avg':'mean'}
    , 'reviews_per_month':{'reviews_per_month_avg':'mean'}
}

airbnb_grouped = airbnb_clean_iqr_rm.groupby(['zipcode', 'neighbourhood_group_cleansed'], as_index=False).agg(aggregation)
airbnb_grouped.columns = ["_".join(x) for x in airbnb_grouped.columns.ravel()]
airbnb_grouped.rename(columns={'zipcode_':'zipcode'}, inplace=True)
airbnb_grouped.zipcode = airbnb_grouped.zipcode.apply(lambda x:int(x))
cost_revenue = pd.merge(df_z_reset, airbnb_grouped, how='left', on='zipcode')

cost_revenue.rename(columns={'2017-06':'cost'}, inplace=True)
cost_revenue[['zipcode','cost','price_price_mediam','number_of_reviews_number_of_reviews_total']]
cost_price_features = ['zipcode','price']
#%%
# Occupancy Rate Table
occ_table = pd.DataFrame({'Location_Score':['9.5-10','8.5-9.5','7.5-8.5','6.5-7.5','<6.5'],
                          'Occupancy_Rate':[0.8, 0.75, 0.7, 0.65, 0.5]})
important_features = ['zipcode','cost','neighbourhood_group_cleansed_', 'square_feet_square_feet_median',
       'price_price_mediam', 'number_of_reviews_number_of_reviews_total',
       'review_scores_location_review_scores_location_avg',
       'reviews_per_month_reviews_per_month_avg']
cost_revenue_final = cost_revenue.loc[:,important_features]
cost_revenue_final.dropna(axis=0, subset=['neighbourhood_group_cleansed_','review_scores_location_review_scores_location_avg'], inplace=True)

data=cost_revenue_final['review_scores_location_review_scores_location_avg']
condlist = [(data<=10) & (data>9.5), (data<=9.5) & (data>8.5), (data<=8.5) & (data>7.5), \
                (data<=7.5) & (data>6.5), (data<= 6.5)]
choicelist = [0.8, 0.75, 0.7, 0.65, 0.5]
cost_revenue_final['occupancy_rate'] = np.select(condlist,choicelist, default = 0)
# total revenue would be daily price * time * occupancy rate * 0.97 (we assume there is a 3% service fee for airbnb)
cost_revenue_final['expected_return'] = (cost_revenue_final.price_price_mediam * 365 * cost_revenue_final.occupancy_rate)*(1-0.03)
cost_revenue_final['total_cost'] = cost_revenue_final.cost + (cost_revenue_final.price_price_mediam * 365 * cost_revenue_final.occupancy_rate)*0.03
cost_revenue_final['return_ratio'] = cost_revenue_final.expected_return / cost_revenue_final.total_cost

cost_revenue_final['total_cost'] = cost_revenue_final['total_cost'].apply(lambda x:int(x))
#%%
# Revenue and Cost Analysis
f, ax = plt.subplots(figsize=(20, 12))
sns.barplot(x='zipcode', y='total_cost',hue='neighbourhood_group_cleansed_', data=cost_revenue_final, palette='husl')

f, ax = plt.subplots(figsize=(20, 12))
sns.barplot(x='zipcode', y='expected_return',hue='neighbourhood_group_cleansed_', data=cost_revenue_final, palette='husl')
#%%
cost_revenue_final['breakeven_period'] = 1/cost_revenue_final.return_ratio
f, ax = plt.subplots(figsize=(20, 12))
sns.barplot(x='zipcode', y='breakeven_period',hue='neighbourhood_group_cleansed_', data=cost_revenue_final, palette='husl')

