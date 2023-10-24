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
import DB_loader
from sqlalchemy import create_engine, text, inspect, Table
import cbsodata

# db = DB_loader()

current_path = os.path.abspath(__file__)
geo_path = os.path.join(os.path.dirname(current_path),"../data/geo.json")
geo = gpd.read_file(geo_path)

data_path = os.path.join(os.path.dirname(current_path),"../data/total_crimes_data.csv")
crimes = pd.read_csv(data_path,usecols=[0,1,2,3,7])

region_code = pd.read_csv(os.path.join(os.path.dirname(current_path),"../data/region_code.csv"))

alldata = pd.merge(region_code, crimes, left_on = "Key", right_on = "region_code").drop(columns='Key')

# Load the crime types dict for the interface
with open(os.path.join(os.path.dirname(current_path),"../data/selected_crimes_codes.json"), 'r') as json_file:
    crimes_codes = json.load(json_file)

app = dash.Dash(__name__)

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
    # Select year
    data_year = alldata[alldata["period"].str.contains(str(selected_year))]
    # Select crime types 
    subtype_options = []
    if crime_type == "total":
        #subtype_options = [{'label': 'Total', 'value': 'total'}]
        print(data_year.columns)
    else:
        subtype_codes = crimes_codes[crime_type]
        data_year = data_year[data_year['crime_code'].isin(list(subtype_codes.keys()))]
    # Construct the subtypes options of the selected crime type that will pass to the linechart
        for code, label in subtype_codes.items():
            option = {'label': label, 'value': code}
            subtype_options.append(option)
    # Process the data
    data_year['period'] = pd.to_datetime(data_year['period'].str.replace('-','',regex=True),format='%Y%m').dt.to_period('M')
    data_sum = data_year.groupby(['region_code', data_year['period'].dt.year])['crimes'].sum().reset_index()
    data =  pd.merge(region_code, data_sum, left_on = "Key", right_on = "region_code").drop(columns='Key')
    merged_data = pd.merge(geo, data, left_on = "statcode", right_on = "region_code")
    merged_data.to_crs(epsg=4326,inplace=True)
    gdf = merged_data.copy()
    gdf['geoid'] = gdf.index.astype(str)
    gdf = gdf[['geoid', 'geometry', 'region','period','crimes']]
    # Visualization of the selected data
    fig = px.choropleth_mapbox(gdf, 
                           geojson=gdf.__geo_interface__, 
                           locations=gdf.geoid, 
                           color="crimes", 
                            mapbox_style="stamen-toner", 
                            hover_name = "region",
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
        print("No selected cities.")
        selected_city = ["Amsterdam"]
    else:
        selected_city = [point['hovertext'] for point in clickData['points']]
    if not crime_subtype:
        print("No selected subtypes.")
        raise PreventUpdate
    # Select city
    data_city = alldata[alldata["region"].isin(selected_city)]
    # Select subtypes
    selected_types = crime_subtype 
    data = data_city[data_city["crime_code"].isin(selected_types)]
    selected_data = data[data["period"].str.contains(str(selected_year))]
    data_sum = selected_data.groupby(['region_code','period'])['crimes'].sum().reset_index()
    data =  pd.merge(region_code, data_sum, left_on = "Key", right_on = "region_code").drop(columns='Key')
    fig = px.line(data, x="period", y="crimes", 
                  title=f"Selected Crimes Count of {selected_city[0]} in Year {selected_year}")
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