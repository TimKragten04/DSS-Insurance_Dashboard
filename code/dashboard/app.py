import geopandas as gpd
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go   
import json
import os
from flask import Flask
from datetime import datetime
from typing import Any
import sqlalchemy as sa
from sqlalchemy import create_engine, text, inspect, Table
from sqlalchemy import Table, MetaData, Column, Integer, String, TIMESTAMP
from DataLoader import DataLoader


data_handler = DataLoader()
updated_dataset = data_handler.run_data_pipeline()

current_path = os.path.abspath(__file__)
data_path = os.path.join(os.path.dirname(current_path),"../data")
data_file_paths = {
    "crime_rates": f"{data_path}/crime_rates.csv",
    "crime_counts":f"{data_path}/crime_counts.csv",
    "population" : f"{data_path}/population.csv",
    "region_code" : f"{data_path}/region_code.csv",
    "crime_codes": f"{data_path}/selected_crimes_codes.json",
    "geo" : f"{data_path}/geo.json"
}

alldata = pd.read_csv(data_file_paths.get("crime_rates"))
geo = gpd.read_file(data_file_paths.get("geo"))

server = Flask(__name__)
app = dash.Dash(__name__, server=server,)
server = app.server

app.layout = html.Div(className='row', children=[
    html.Div([
        dcc.Dropdown(id="crime-type-selector", options=[
        {'label': 'Theft', 'value': 'theft'},
        {'label': 'Damage', 'value': 'damage'},
        {'label': 'Total', 'value': 'total'}
        ], value='total'),
        dcc.Graph(id="crime-heatmap"),
        dcc.Slider(id="year-selector",min=2010,  max=2022, step=1,  marks={str(year): str(year) for year in range(2012, 2023)},  
            value=2022 ),
    ], style={'display': 'inline-block', 'width': '59%','padding': '0 20'}),  

    html.Div([
        dcc.Dropdown(id="crime-subtype-selector", options=[], 
                     searchable=True, multi=True),
        dcc.Graph(id="crime-linechart"),
    ], style={'display': 'inline-block', 'width': '37%'}), 
])

@app.callback(
    Output("crime-heatmap", "figure"),
    Output("crime-subtype-selector", "options"),
    [Input("year-selector", "value"),
     Input("crime-type-selector", "value")]
)


def update_crime_heatmap(selected_year=2022, crime_type = "total"):
    with open(data_file_paths.get("crime_codes"), 'r') as json_file:
        crime_codes = json.load(json_file)

    # Select year
    data_year = alldata[alldata["period"].str.contains(str(selected_year))]
    # Select crime types 
    subtype_options = []

    if crime_type != "total":
        subtype_codes = crime_codes[crime_type]
        print(f"{crime_type}:{subtype_codes}")
        data_year = data_year[data_year['crime_code'].isin(list(subtype_codes.keys()))]
    # Construct the subtypes options of the selected crime type that will pass to the linechart
        for code, label in subtype_codes.items():
            option = {'label': label, 'value': code}
            subtype_options.append(option)
    # Process the data
    data_year['period'] = pd.to_datetime(data_year['period'])
    print(data_year.iloc[87,])
    data = data_year.groupby(['region_code', data_year['period'].dt.year])['crime_rate'].sum().reset_index()

    print(data.iloc[87,])

    # merge with geo data
    merged_data = pd.merge(geo, data, left_on = "statcode", right_on = "region_code")
    merged_data = merged_data.dropna()
    merged_data.to_crs(epsg=4326,inplace=True)
    gdf = merged_data.copy()
    gdf['geoid'] = gdf.index.astype(str)
    gdf = gdf[['geoid', 'geometry', 'statnaam','crime_rate']]
    # Visualization of the selected data
    fig = px.choropleth_mapbox(gdf, 
                           geojson=gdf.__geo_interface__, 
                           locations=gdf.geoid, 
                           color="crime_rate", 
                            mapbox_style="open-street-map", 
                            hover_name = "statnaam",
                            zoom=6,
                            featureidkey='properties.geoid',
                            center = {"lat":52.12,"lon":5.16},
                            color_continuous_scale = "sunset",
                            title=f"Crime Heatmap {selected_year}")
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0}  )
    return fig, subtype_options

@app.callback(
    Output("crime-linechart", "figure"),
    [Input("crime-heatmap", "clickData"),
    Input('crime-subtype-selector','value'),
    Input("year-selector", "value")]
)

def update_crime_linechart(clickData, crime_subtype, selected_year = 2022):
    #print(json.dumps(clickData, indent=2))
    if not clickData:
        selected_city = ["Amsterdam"]
    else:
        selected_city = [point['hovertext'] for point in clickData['points']]

    # Select city
    data_city = alldata[alldata["region"].isin(selected_city)]

    if not crime_subtype:
        data_city['period'] = pd.to_datetime(data_city['period'])
        data = data_city.groupby(['region_code', data_city['period'].dt.year])['crime_rate'].sum().reset_index()
        fig = px.line(data, x="period", y="crime_rate", 
                  title=f"Total Crime Rates of {selected_city[0]} over Years")
        fig.update_layout(margin={"r": 0, "t": 50, "l": 10, "b": 0},
                      plot_bgcolor="white",
                      yaxis=dict(title=None) )
        fig.update_traces(line=dict(color="black"),
                      textposition='top center',  
                      mode='lines+markers', 
                      texttemplate='%{text}')
        print("No selected subtypes.")
        return fig
    
    # Select subtypes
    selected_types = crime_subtype 
    data = data_city[data_city["crime_code"].isin(selected_types)]
    selected_data = data[data["period"].str.contains(str(selected_year))]
    data_sum = selected_data.groupby(['region_code','period'])['crime_rate'].sum().reset_index()

    fig = px.line(data_sum, x="period", y="crime_rate", 
                  title=f"Selected Crime Rates of {selected_city[0]} in Year {selected_year}")
    fig.update_layout(margin={"r": 0, "t": 50, "l": 10, "b": 0},
                      plot_bgcolor="white",
                      yaxis=dict(title=None) )
    fig.update_traces(line=dict(color="black"),
                      textposition='top center',  
                      mode='lines+markers', 
                      texttemplate='%{text}')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=True, host='0.0.0.0', port=8050) 