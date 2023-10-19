import geopandas as gpd
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import json
import os


current_path = os.path.abspath(__file__)
geo_path = os.path.join(os.path.dirname(current_path),"../data/geo.json")
data_path = os.path.join(os.path.dirname(current_path),"../data/total_crimes_per_region&year.csv")
geo = gpd.read_file(geo_path)
data = pd.read_csv(data_path)

merged_data = pd.merge(geo, data, left_on = "statcode", right_on = "region")
merged_data.to_crs(epsg=4326,inplace=True)
gdf = merged_data.copy()
gdf['geoid'] = gdf.index.astype(str)
gdf = gdf[['geoid', 'geometry', 'statnaam', 'crimes_count','year']]

app = dash.Dash(__name__)

app.layout = html.Div(className='row', children=[
    html.Div([
        dcc.Graph(id="crime-heatmap"),
        dcc.Slider(id="year-selector",min=2010,  max=2022, step=1,  marks={str(year): str(year) for year in range(2010, 2023)},  
            value=2022 ),
    ], style={'display': 'inline-block', 'width': '59%','padding': '0 20'}),  

    html.Div([
        dcc.Dropdown(id="crime-type-selector", options=[
        {'label': 'Theft', 'value': 'theft'},
        {'label': 'Property Destruction', 'value': 'property_destruction'},
        {'label': 'Total', 'value': 'total'}
    ], value='total'),
        dcc.Graph(id="crime-linechart"),
    ], style={'display': 'inline-block', 'width': '37%'}), 
])

@app.callback(
    Output("crime-heatmap", "figure"),
    [Input("year-selector", "value")]
)


def update_crime_heatmap(selected_year=2022):
    data = gdf[gdf["year"].str.contains(str(selected_year))]
    fig = px.choropleth_mapbox(data, 
                           geojson=data.__geo_interface__, 
                           locations=data.geoid, 
                           color="crimes_count", 
                            mapbox_style="stamen-toner", 
                            hover_name = "statnaam",
                            zoom=6,
                            featureidkey='properties.geoid',
                            center = {"lat":52.12,"lon":5.16},
                            color_continuous_scale = "sunset",
                            title=f"Crime Heatmap {selected_year}")
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0}  )
    #fig.update_layout(mapbox={"style": "stamen-toner"}, showlegend=False)
    return fig

@app.callback(
    Output("crime-linechart", "figure"),
    [Input("crime-heatmap", "clickData"),
     Input("crime-type-selector", "value")]
)

def update_crime_linechart(clickData, crime_type='total'):
    #print(json.dumps(clickData, indent=2))
    if not clickData:
        print("No selected data.")
        selected_city = ["Amsterdam"]
    else:
        selected_city = [point['hovertext'] for point in clickData['points']]
    data = gdf[gdf["statnaam"].isin(selected_city)]  # & (gdf["type"] == crime_type)
    fig = px.line(data, x="year", y="crimes_count", title=f"{crime_type.capitalize()} Crimes Count of {selected_city[0]} Over Years")
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