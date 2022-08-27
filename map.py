import pandas as pd
import numpy as np
from urllib.request import urlopen
import json
import geopandas as gpd
import plotly.graph_objects as go
import folium
import os

with urlopen('https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/russia.geojson') as response:
    geoJson = json.load(response)


gdf = gpd.GeoDataFrame(geoJson['features'])
df = pd.DataFrame(columns=list(gdf['properties'][0].keys()), index=[*range(0, 83)])
for i in range(len(gdf['properties'])):
    for j in df.columns:
        df[j][i] = gdf['properties'][i][j]
df['coordinates'] = 0
df['type'] = 0
df = df.drop(['created_at', 'updated_at'], axis=1)
for i in range(len(gdf['geometry'])):
    for j in gdf['geometry'][0].keys():
        df[j][i] = gdf['geometry'][i][j]

gdf=gpd.GeoDataFrame(df)

def region(to_map):
    fig = go.Figure(go.Choroplethmapbox(geojson=geoJson,
                                            locations=gdf['name'],
                                            z=to_map['amount_connect'],
                                            featureidkey="properties.name",
                                            colorscale='greens',
                                            #customdata=np.stack(train['inf_rate']),
                                            zmin=0,
                                            zmax=max(to_map['amount_connect'])+100,
                                            marker_opacity=0.5,
                                            marker_line_width=1))
    fig.update_layout(mapbox_style="open-street-map",
                      mapbox_zoom=2,mapbox_center = {"lat":60.18678 , "lon": 95.857324} )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig.show()
    return fig
def label(train):
    train=train.reset_index(drop=True)
    m = folium.Map(
        location=[-59.1759, -11.6016],
        #tiles="open-street-map",
        zoom_start=2,
    )
    folium.Choropleth(geo_data=geoJson, line_color='blue',
                      line_weight=3).add_to(m)
    for i in range(len(train)):
        if i % 10 == 0:
            folium.Marker(
                location=[train['hex_lat'][i], train['hex_lon'][i]],
                # tooltip=folium.Tooltip(text='dufha'),
                icon=folium.Icon(color='red')
            ).add_to(m)
    return m
