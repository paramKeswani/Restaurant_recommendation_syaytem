import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import openpyxl

import app
model = pickle.load(open('model_pred.pkl', 'rb'))
df = pd.read_excel(
    r'C:\Users\Lenovo\OneDrive\Desktop\mini_final\Location-based-Restaurant-Recommendation-System-GUI-main\data.xlsx')

df.drop(['dish_category', 'id', 'res_link', 'price_for_two'], axis=1, inplace=True)

df.rename(columns={'veg_status': 'dish_category'}, inplace=True)

numerical_col = []
categorical_col = []

for column in df.columns:
    try:
        float(df[column].mode()[0])
        numerical_col.append(column)
    except:
        categorical_col.append(column)

drop_cusines = ['Juices', 'Fast Food', 'Beverages', 'Snacks', 'Kebabs', 'Bakery', 'Rolls & Wraps',
                'Desserts', 'Waffle', 'Biryani', 'Combo', 'Grill', 'Barbecue', 'Home Food', 'Ice Cream Cakes'
                                                                                            'Pastas', 'Ice Cream',
                'Sweets', 'Chaat', 'Burgers', 'Healthy Food', 'Paan', 'Tex-Mex', 'Sushi'
                                                                                 'Pizzas', 'Cafe', 'Tandoor', 'Thalis',
                'Salads', 'Seafood', 'Street Food', 'Cakes and Pastries']


def temp(val):
    z = val.split(',')
    if len(z) == 2:
        return z[0]
    else:
        return val


df['cusines'] = df['cusines'].apply(temp)

for cuisine_to_drop in drop_cusines:
    df['cusines'] = df['cusines'].str.replace(cuisine_to_drop, '', regex=True)

df['cusines'] = df['cusines'].replace('', 'Unkonwn')

numerical_col = []
categorical_col = []

for column in df.columns:
    try:
        float(df[column].mode()[0])
        numerical_col.append(column)
    except:
        categorical_col.append(column)

# from geopy.geocoders import Nominatim

# locations=pd.DataFrame({"Name":df['location'].unique()})
# locations['Name']=locations['Name'].apply(lambda x: "Bangalore " + str(x))
# lat_lon=[]
# geolocator=Nominatim(user_agent="app")
# for location in locations['Name']:
#     location = geolocator.geocode(location)
#     if location is None:
#         lat_lon.append(np.nan)
#     else:
#         geo=(location.latitude,location.longitude)
#         lat_lon.append(geo)


# # locations['geo_loc']=lat_lon
# locations.to_csv('locations.csv',index=False)


# locations["Name"] = locations['Name'].apply(lambda x: " ".join(filter(lambda y: y!="Bangalore",x.split())))

# Rest_locations=pd.DataFrame(df['location'].value_counts().reset_index())
# Rest_locations.columns=['Name','count']
# Rest_locations=Rest_locations.merge(locations,on='Name',how="left").dropna()
# Rest_locations['count'].max()

# # def generateBaseMap(default_location=[12.97, 77.59], default_zoom_start=12):
# #     base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
# #     return base_map

# # import folium
# # from folium.plugins import HeatMap

# lat,lon=zip(*np.array(Rest_locations['geo_loc']))
# Rest_locations['lat']=lat
# Rest_locations['lon']=lon


df1 = df.copy()

for column in categorical_col:
    if df[column].nunique() >= 40:
        df1 = df1.drop(column, axis=1)

x = df1.drop('price_for_one', axis=1)
y = df1['price_for_one']

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

for i in df1:
    if (df1[i].dtypes != 'O') and i != 'price_for_one':
        df1[i] = sc.fit_transform(df1[[i]])

x = pd.get_dummies(df1, columns=['dish_category', 'cusines'], drop_first=True)

from sklearn.decomposition import PCA

pca = PCA(n_components=40)
pca
pca1 = pca.fit_transform(x)

pca2 = PCA(n_components=1)
pca2.fit(x)
x_ = pca2.transform(x)
x_ = pd.DataFrame(x_)

# LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics

x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.2, random_state=10)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

r2_score(y_test, y_pred)

# LOGISTIC REGRESSION

df2 = df.copy()  # Create a copy of the original DataFrame

for column in categorical_col:
    if df[column].nunique() >= 40:
        df2 = df2.drop(column, axis=1)

numerical_co = []
categorical_co = []

for column in df2.columns:
    try:
        float(df2[column].mode()[0])
        numerical_co.append(column)
    except:
        categorical_co.append(column)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

for i in df2:
    if (df2[i].dtypes != 'O'):
        df2[i] = sc.fit_transform(df2[[i]])

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df2['cusines'] = label_encoder.fit_transform(df2['cusines'])

from sklearn.linear_model import LogisticRegression

df2 = pd.get_dummies(df2, columns=['dish_category'], drop_first=True)
x = df2.drop('cusines', axis=1)
y = df2['cusines']
lg = LogisticRegression(max_iter=1000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

lg.fit(x_train, y_train)

y_pred = lg.predict(x_test)

metrics.confusion_matrix(y_test, y_pred)

metrics.accuracy_score(y_test, y_pred)

# RANDOM FOREST CLASSIFIER

df['location'].nunique()

df3 = df.copy()  # Create a copy of the original DataFrame

for column in categorical_col:
    if df[column].nunique() >= 135:
        df3 = df3.drop(column, axis=1)

df3.drop('cusines', axis=1, inplace=True)

numerical_column = []
categorical_column = []

for column in df3.columns:
    try:
        float(df3[column].mode()[0])
        numerical_column.append(column)
    except:
        categorical_column.append(column)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

for i in df3:
    if (df3[i].dtypes != 'O'):
        df3[i] = sc.fit_transform(df3[[i]])

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df3['location'] = label_encoder.fit_transform(df3['location'])

df3 = pd.get_dummies(df3, columns=['dish_category'], drop_first=True)

x = df3.drop('location', axis=1)

y = df3['location']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=55)

from sklearn.ensemble import RandomForestClassifier

rc = RandomForestClassifier(max_depth=10, min_samples_split=5)

rc.fit(x_train, y_train)

y_pred = rc.predict(x_test)

metrics.accuracy_score(y_test, y_pred)

df5 = pd.read_excel(
    r'C:\Users\Lenovo\OneDrive\Desktop\mini_final\Location-based-Restaurant-Recommendation-System-GUI-main\data.xlsx')

df5.duplicated().sum()
df5.isnull().sum()
df5.drop('dish_category', axis=1, inplace=True)
df5.rename(columns={'veg_status': 'dish_category'}, inplace=True)
df5.drop(['dish_name', 'id', 'res_link', 'price_for_two', 'delivery_review_number', 'dish_category'], axis=1,
         inplace=True)


def temp(val):
    z = val.split(',')
    if len(z) == 2:
        return z[0]
    else:
        return val


df5['cusines'] = df5['cusines'].apply(temp)
for cuisine_to_drop in drop_cusines:
    df5['cusines'] = df5['cusines'].str.replace(cuisine_to_drop, '', regex=True)
df5['cusines'] = df5['cusines'].replace('', 'Unkonwn')

numerical_col = []
categorical_col = []

for column in df5.columns:
    try:
        float(df5[column].mode()[0])
        numerical_col.append(column)
    except:
        categorical_col.append(column)

df5 = df5.drop('location', axis=1)

numerical_co = []
categorical_co = []

for column in df5.columns:
    try:
        float(df5[column].mode()[0])
        numerical_co.append(column)
    except:
        categorical_co.append(column)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

for i in df5:
    if (df5[i].dtypes != 'O'):
        df5[i] = sc.fit_transform(df5[[i]])

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df5['cusines'] = label_encoder.fit_transform(df5['cusines'])

df5 = pd.get_dummies(df5, columns=['res_name'], drop_first=True)

x = df5.drop('cusines', axis=1)
y = df5['cusines']

from sklearn.ensemble import RandomForestClassifier

rc = RandomForestClassifier()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

rc.fit(x_train, y_train)

y_pred = rc.predict(x_test)

print(metrics.classification_report(y_test, y_pred))

metrics.confusion_matrix(y_test, y_pred)

metrics.accuracy_score(y_test, y_pred)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle

df.drop_duplicates(inplace=True)
avg_df = pd.DataFrame(df.groupby(['cusines', 'location']).agg({'price_for_one': 'mean'})).reset_index()

location_list = list(avg_df['location'].unique())
loc_list = sorted(location_list)

cusine_list = list(avg_df['cusines'].unique())
cus_list = sorted(cusine_list)

loc_dit = {}
for i in range(len(loc_list)):
    loc_dit[loc_list[i]] = i
cus_dit = {}
for i in range(len(cus_list)):
    cus_dit[cus_list[i]] = i

avg_df['location'] = avg_df['location'].apply(lambda x: loc_dit[x])
avg_df['cusines'] = avg_df['cusines'].apply(lambda x: cus_dit[x])
x = avg_df.drop(['price_for_one'], axis=1)
y = avg_df['price_for_one']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
sc = StandardScaler()
x_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)
new_df = avg_df.copy()

model = LinearRegression()
model.fit(x_sc, y_train)
y_pred = model.predict(x_test_sc)

pickle.dump(model, open('model_pred.pkl', 'wb'))

from math import radians, sin, cos, sqrt, atan2


def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Debugging prints
    print(f"lat1: {lat1}, lon1: {lon1}")
    print(f"lat2: {lat2}, lon2: {lon2}")
    print(f"dlat: {dlat}, dlon: {dlon}")

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate the distance
    distance = R * c

    return distance


import pandas as pd

df = df[df['cusines'] != 'Unkonwn']

API_KEY = '30d53b8a3e784f36a82719922962fa96'
from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(API_KEY)


def geocode_address(address):
    result = geocoder.geocode(address)
    if result and len(result):

        location = result[0]['geometry']
        latitude, longitude = location['lat'], location['lng']
        return latitude, longitude
    else:
        return None


def haversinee(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371  # Earth radius in kilometers

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return distance


import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import pickle
import pandas as pd
from ttkthemes import ThemedStyle
from PIL import Image, ImageTk
from IPython.display import display, HTML

def abc(Cusine1, Location, current_location, Dish_Category):
    import numpy as np
    print("hello")

    print(Cusine1)

    try:
        # print(cus_dit)
        # print(loc_dit)
        Cusine1_encode = cus_dit.get(Cusine1, -1)
        Location_encode = loc_dit.get(Location, -1)

        print("encoding")

        arr = np.array([[Cusine1_encode, Location_encode]])
        print("contains encoding for cusines_encode , location")
        print(arr)
        prediction = model.predict(arr)[0]
        print(prediction)
        # if len(prediction) == 0:
        #     print("Error: Empty prediction sequence")
        #     return
        # prediction = max(prediction, 0)

        pop_cuisine = df[df['location'] == current_location]['cusines'].value_counts().idxmax()
        pop_res = df[df['cusines'] == Cusine1]['res_name'].value_counts().idxmax()
        most_pop_res = df[df['location'] == current_location]['res_name'].value_counts().idxmax()
        just_res = df[df['location'] == current_location]['res_name']
        pop_serves = df[df['location'] == current_location]['dish_name'].value_counts().idxmax()
        rec_predict = df[df['location'] == Location]['price_for_one'].value_counts().idxmax()

        user_location = geocode_address(Location)
        current_location_coordinates = geocode_address(current_location)
        distance_to_user_loc = haversinee(user_location, current_location_coordinates)

        top_restaurants = df[(df['location'] == Location) & (df['cusines'] == Cusine1) & (
                    df['dish_category'] == Dish_Category)].sort_values('rating', ascending=False).groupby(
            'res_name').head(1).head(5)[['res_name', 'rating', 'cusines', 'dish_name']]

        print("Top Restaurants:")
        for index, row in top_restaurants.iterrows():
            distance_formatted = '{:.2f} km'.format(distance_to_user_loc)
            price_for_dish = \
            df[(df['location'] == Location) & (df['cusines'] == Cusine1) & (df['dish_name'] == row['dish_name'])][
                'price_for_one'].values[0]
            print(
                f"Restaurant Name: {row['res_name']}, Rating: {row['rating']}, Cuisine: {row['cusines']}, Dish Name: {row['dish_name']}, Distance: {distance_formatted}, Category: {Dish_Category}, Price: {price_for_dish}")

        top_restaurants_current_location = df[(df['location'] == current_location) & (df['cusines'] == Cusine1) & (
                    df['dish_category'] == Dish_Category)].sort_values('rating', ascending=False).groupby(
            'res_name').head(1).head(5)[['res_name', 'rating', 'cusines', 'dish_name']]

        print("\nTop Restaurants in Current Location:")
        for index, row in top_restaurants_current_location.iterrows():
            price_for_dish = df[(df['location'] == current_location) & (df['cusines'] == Cusine1) & (
                        df['dish_name'] == row['dish_name'])]['price_for_one'].values[0]
            print(
                f"Restaurant Name: {row['res_name']}, Rating: {row['rating']}, Cuisine: {row['cusines']}, Dish Name: {row['dish_name']}, Category: {Dish_Category}, Price: {price_for_dish}")

        print("pop_res")
        print(pop_res)
        print("pop_cuisine")
        print(pop_cuisine)
        print("most_pop_res")
        print(most_pop_res)
        print("just_res")
        print(just_res.unique())
        print("pop_erves")
        print(pop_serves)
        print("top_restaurants")
        print(top_restaurants)
        print("t_r_c_l")
        print(top_restaurants_current_location)

        list =[]

        list_data = just_res.unique().tolist()
        print(list_data)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return list_data

# Function to handle the prediction logic




