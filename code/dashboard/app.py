import os
import dash
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
from dash import Dash, dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import sqlalchemy as sa
from pathlib import Path


def _fetch_data_from_db(dataset: str) -> pd.DataFrame:
    """
    Fetches data from the database, based on the dataset name.

    Args:
        dataset (str): The dataset name.

    Returns:
        pd.Dataframe: The DataFrame containing the data.
    """
    engine = sa.create_engine("postgresql://student:infomdss@dashboard:5432/dashboard")
    df = pd.read_sql_table(dataset, engine, index_col="index")

    return df


def create_geodataframe(merge_dataframe: pd.DataFrame, geojson_path: Path, merge_key: str) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame from a .geojson file, and merges it on some value with a regular DataFrame object. 
    The keys it will merge on must have the same name.

    Args:
        merge_dataframe (pd.DataFrame): DataFrame to be merged with the GeoDataFrame.
        geojson_path (Path): Path to the location of the .geojson file.
        merge_key (str): Key both DataFrames will be merged on.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame that has been merged with a regular DataFrame.
    """
    geo_df: gpd.GeoDataFrame = gpd.read_file(geojson_path)
    geo_df = geo_df.merge(merge_dataframe, on=merge_key)
    geo_df = geo_df.set_index(merge_key)

    return geo_df


def filter_dataframe(df: pd.DataFrame, column: str, values: [str], remove: bool = True) -> pd.DataFrame:
    """
    Filters a dataframe at some column, removing or keeping the rows that have specific values.

    Args:
        df (pd.DataFrame): Dataframe to be filtered.
        column (str): The column for which the values will be checked.
        values (str]): The values that are compared to the column values.
        remove (bool, optional): Indicates whether the rows should be removed or kept. Defaults to True.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if remove:
        mask = ~df[column].isin(values)
    else:
        mask = df[column].isin(values)

    filtered_df: pd.DataFrame = df[mask]

    return filtered_df


def create_income_inequality_map(geo_df: gpd.GeoDataFrame):
    """
    Create a map that displays income inequality data, which can be displayed on the dashboard.

    Args:
        geo_df (gpd.GeoDataFrame): The GeoDataFrame the map will be based on.

    Returns:
        Figure: The map that can be displayed on the dashboard.
    """
    fig = px.choropleth(
        geo_df,
        geojson=geo_df.geometry,
        locations=geo_df.index,
        color="inequality_income",
        hover_name="region_name",
    )
    fig.update_layout(
        title="Map that shows income inequality",
        coloraxis_colorbar_title="Average minus<br>median income",
        geo=dict(
            bgcolor="#e5ecf6"
        ),
        paper_bgcolor="#f0f0f0",
        height=450,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )
    fig.update_geos(
        visible=False,
        fitbounds="locations"
    )

    return fig


def create_income_inequality_graph(df: pd.DataFrame):
    """
    Create a graph that displays income inequality data, which can be displayed on the dashboard.

    Args:
        df (pd.DataFrame): The DataFrame the graph will be based on.

    Returns:
        Figure: The graph that can be displayed on the dashboard.
    """
    labels: dict[str, str] = {
        "time_period": "Year",
        "inequality_income": "Income Inequality",
        "region_name": "Region"
    }
    fig = px.line(df, x="time_period", y="inequality_income", color="region_name", labels=labels)
    fig.update_layout(
        title="Line chart that shows income inequality over the years",
        height=450,
        paper_bgcolor="#f0f0f0",
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    return fig


def create_people_in_debt_map(geo_df: gpd.GeoDataFrame):
    """
    Create a map that displays income inequality data, which can be displayed on the dashboard.

    Args:
        geo_df (gpd.GeoDataFrame): The GeoDataFrame the map will be based on.

    Returns:
        Figure: The map that can be displayed on the dashboard.
    """
    fig = px.choropleth(
        geo_df,
        geojson=geo_df.geometry,
        locations=geo_df.index,
        color="people_in_debt",
        hover_name="region_name",
    )
    fig.update_layout(
        title="Map that shows the # people that have debt",
        coloraxis_colorbar_title="# of people",
        geo=dict(
            bgcolor="#e5ecf6"
        ),
        paper_bgcolor="#f0f0f0",
        height=450,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )
    fig.update_geos(
        visible=False,
        fitbounds="locations"
    )

    return fig


def create_people_in_debt_graph(df: pd.DataFrame):
    """
    Create a graph that displays income inequality data, which can be displayed on the dashboard.

    Args:
        df (pd.DataFrame): The DataFrame the graph will be based on.

    Returns:
        Figure: The graph that can be displayed on the dashboard.
    """
    labels: dict[str, str] = {
        "time_period": "Year",
        "inequality_income": "Income Inequality",
        "region_name": "Region"
    }
    fig = px.line(df, x="time_period", y="people_in_debt", color="region_name", labels=labels)
    fig.update_layout(
        title="Line chart that shows income inequality over the years",
        paper_bgcolor="#f0f0f0",
        height=450,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    return fig


def create_property_value_map(geo_df: gpd.GeoDataFrame):
    """
    Create a map that displays income inequality data, which can be displayed on the dashboard.

    Args:
        geo_df (gpd.GeoDataFrame): The GeoDataFrame the map will be based on.

    Returns:
        Figure: The map that can be displayed on the dashboard.
    """
    fig = px.choropleth(
        geo_df,
        geojson=geo_df.geometry,
        locations=geo_df.index,
        color="property_value",
        hover_name="region_name",
    )
    fig.update_layout(
        title="Map that shows income inequality",
        coloraxis_colorbar_title="Property Value",
        geo=dict(
            bgcolor="#e5ecf6"
        ),
        paper_bgcolor="#f0f0f0",
        height=450,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )
    fig.update_geos(
        visible=False,
        fitbounds="locations"
    )

    return fig


def create_property_value_graph(df: pd.DataFrame):
    """
    Create a graph that displays income inequality data, which can be displayed on the dashboard.

    Args:
        df (pd.DataFrame): The DataFrame the graph will be based on.

    Returns:
        Figure: The graph that can be displayed on the dashboard.
    """
    labels: dict[str, str] = {
        "time_period": "Year",
        "property_value": "Property Value",
        "region_name": "Region"
    }
    fig = px.line(df, x="time_period", y="property_value", color="region_name", labels=labels)
    fig.update_layout(
        title="Line chart that shows property value over the years",
        paper_bgcolor="#f0f0f0",
        height=450,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    return fig


external_stylesheets: list[str] = ["static/style.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)
children_poverty_years = pd.read_csv("../data/cleaned/child_poverty/poverty_all_data.csv")["Perioden"].unique()
children_poverty_municipalities = pd.read_csv("../data/cleaned/child_poverty/poverty_all_data.csv")["Title"].unique()
debt_restructuring_years = pd.read_csv("../data/cleaned/debt_restructuring/debt_all_data.csv")["Perioden"].unique()
debt_restructuring_municipalities = pd.read_csv("../data/cleaned/debt_restructuring/debt_all_data.csv")["Title"].unique()

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label="Crime Statistics", children=[
            html.Div(className="row", children=[
                html.Div([
                    dcc.Dropdown(id="crime_selected_type", options=[
                    {"label": "Theft", "value": "theft"},
                    {"label": "Damage", "value": "damage"},
                    {"label": "Total", "value": "total"}
                    ], value="total"),
                    dcc.Graph(id="crime_heatmap"),
                    dcc.Slider(id="crime_selected_year",min=2010,  max=2022, step=1,  marks={str(year): str(year) for year in range(2012, 2023)},  
                        value=2022 ),
                ], style={"display": "inline-block", "width": "59%","padding": "0 20"}),  

                html.Div([
                    dcc.Dropdown(id="crime_selected_subtype", options=[], 
                                searchable=True, multi=True),
                    dcc.Graph(id="crime_linechart"),
                ], style={"display": "inline-block", "width": "37%"}), 
            ])
        ]),
        dcc.Tab(label="Income Inequality", children=[
            html.Div([
            html.H4("Income Inequality"),
            html.P("Select region:"),
            dcc.RadioItems(
                id="income_inequality_region",
                options=["Province", "Municipality"],
                value="Municipality",
                inline=False
            ),
            ], style={"width": "15%", "display": "inline-block", "vertical-align": "top", "margin": "0", "padding": "0px 5px 0px 5px"}),
            html.Div([
                dcc.Graph(id="income_inequality_map"),
            ], style={"width": "40%", "display": "inline-block"}),
            html.Div([
                dcc.Graph(id="income_inequality_line"),
            ], style={"width": "44%", "display": "inline-block"}),
        ]),
        dcc.Tab(label="Children Poverty", children=[
            html.Div([
                html.H4("Other Statistics"),
                html.P("Select Municipality for Line Chart:"),
                dcc.Dropdown(
                    id="child_poverty_selected_municipality",
                    options=[{"label": municipailty, "value": municipailty} for municipailty in children_poverty_municipalities],
                    value=children_poverty_municipalities[0],
                    clearable=False
                ),
                html.P("Select Year for Histogram:"),
                dcc.Dropdown(
                    id="child_poverty_selected_year",
                    options=[{"label": year, "value": year} for year in children_poverty_years],
                    value=children_poverty_years[0],
                    clearable=False
                ),
            ], style={"width": "15%", "display": "inline-block", "vertical-align": "top", "margin": "0", "padding": "0px 5px 0px 5px"}),
            html.Div([
                dcc.Graph(id="child_poverty_graph"),
            ], style={"width": "40%", "display": "inline-block"}),
            html.Div([
                dcc.Graph(id="child_poverty_histogram"),
            ], style={"width": "44%", "display": "inline-block"}),
        ]),
        dcc.Tab(label="Debt Restructuring", children=[
            html.Div([
                html.H4("Other Statistics"),
                html.Label("Select Municipality for Line Chart:"),
                dcc.Dropdown(
                    id="debt_restructuring_selected_municipality",
                    options=[{"label": municipality, "value": municipality} for municipality in debt_restructuring_municipalities],
                    value=debt_restructuring_municipalities[0],
                    clearable=False
                ),
                html.P("Select Year for Histogram:"),
                dcc.Dropdown(
                    id="debt_restructuring_selected_year",
                    options=[{"label": year, "value": year} for year in debt_restructuring_years],
                    value=debt_restructuring_years[0],
                    clearable=False
                ),
            ], style={"width": "15%", "display": "inline-block", "vertical-align": "top", "margin": "0", "padding": "0px 5px 0px 5px"}),
            html.Div([
                dcc.Graph(id="debt_restructuring_graph"),
            ], style={"width": "40%", "display": "inline-block"}),
            html.Div([
                dcc.Graph(id="debt_restructuring_histogram"),
            ], style={"width": "44%", "display": "inline-block"}),
        ]),
        dcc.Tab(label="People in Debt", children=[
            html.Div([
            html.H4("People in Debt"),
            html.P("Select region:"),
            dcc.RadioItems(
                id="people_in_debt_region",
                options=["Province", "Municipality"],
                value="Municipality",
                inline=False
            ),
            ], style={"width": "15%", "display": "inline-block", "vertical-align": "top", "margin": "0", "padding": "0px 5px 0px 5px"}),
            html.Div([
                dcc.Graph(id="people_in_debt_map"),
            ], style={"width": "40%", "display": "inline-block"}),
            html.Div([
                dcc.Graph(id="people_in_debt_line"),
            ], style={"width": "44%", "display": "inline-block"}),
        ]),
        dcc.Tab(label="Property Value", children=[
            html.Div([
            html.H4("Property Value"),
            html.P("Select region:"),
            dcc.RadioItems(
                id="property_value_region",
                options=["Province", "Municipality"],
                value="Municipality",
                inline=False
            ),
            ], style={"width": "15%", "display": "inline-block", "vertical-align": "top", "margin": "0", "padding": "0px 5px 0px 5px"}),
            html.Div([
                dcc.Graph(id="property_value_map"),
            ], style={"width": "40%", "display": "inline-block"}),
            html.Div([
                dcc.Graph(id="property_value_line"),
            ], style={"width": "44%", "display": "inline-block"}),
        ]),
    ])
])


def display_crime_heatmap(alldata: pd.DataFrame, crimes_codes: pd.DataFrame, region_code: pd.DataFrame, geo: gpd.GeoDataFrame, selected_year=2022, crime_type = "total"):
    # Select year
    data_year = alldata[alldata["period"].str.contains(str(selected_year))]
    # Select crime types 
    subtype_options = []

    if crime_type != "total":
        subtype_codes = crimes_codes[crime_type]
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
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
    return fig, subtype_options


def display_crime_linechart(alldata: pd.DataFrame, clickData, crime_subtype, region_code: pd.DataFrame, selected_year = 2022):
    print("test")
    #print(json.dumps(clickData, indent=2))
    if not clickData:
        print("No selected cities.")
        selected_city = ["Amsterdam"]
    else:
        selected_city = [point["hovertext"] for point in clickData["points"]]
    if not crime_subtype:
        print("No selected subtypes.")
        crime_subtype = [{"label": "Total", "value": "total"}]
    
    # Select city
    data_city = alldata[alldata["region_code"].isin(selected_city)]
    
    
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


@app.callback(
    Output("crime_heatmap", "figure"),
    Output("crime_selected_subtype", "options"),
    Output("crime_linechart", "figure"),
    [Input("crime_heatmap", "clickData"),
    Input("crime_selected_subtype","value"),
    Input("crime_selected_type", "value"),
    Input("crime_selected_year", "value")
    ]
)
def display_crime_graphs(clickData, crime_subtybe, crime_type = "total", selected_year=2022):
    crime_rates_df: pd.DataFrame = _fetch_data_from_db("crime_rates")
    geo_path = "geodata/geo.json"
    geo = gpd.read_file(geo_path)
    region_code_df = pd.read_csv("../data/cleaned/crime/region.csv", delimiter=";")
    alldata = pd.merge(region_code_df, crime_rates_df, left_on = "Key", right_on = "region_code").drop(columns="Key")

    with open("../data/cleaned/crime/selected_crimes_codes.json", "r") as json_file:
        crimes_codes = json.load(json_file)

    crime_heatmap, subtype_options = display_crime_heatmap(alldata, crimes_codes, region_code_df, geo, selected_year, crime_type)
    crime_linechart = display_crime_linechart(alldata, clickData, crime_subtybe, region_code_df, selected_year)

    return crime_heatmap, subtype_options, crime_linechart


@app.callback(
    Output("income_inequality_map", "figure"),
    Output("income_inequality_line", "figure"),
    Input("income_inequality_region", "value"),
    Input("income_inequality_map", "clickData")
)
def display_income_inequality_graphs(region: str, click_data):
    """
    Fetches the income inequality data, generates the figures and displays it on the dashboard.
    Gets updated every time the user clicks on the map, or if the income_region is changed.

    Args:
        income_region (str): The region type that is selected. Can be either Municipality or Provice.
        click_data: Object that stores information where the user clicked

    Returns:
        Figure, Figure: Two figures are returned, one is the income map, the other the income graph.
    """
    income_inequality_df: pd.DataFrame = _fetch_data_from_db("income_houshold")

    if(region == "Province"):
        # Create GeoDataframe for Province data that is used in the Map
        income_inequality_df_filtered = filter_dataframe(income_inequality_df, "region_type", ["Country", "Municipality"], True)
        mask = (income_inequality_df_filtered["region_name"] == "Fryslân")
        income_inequality_df_filtered["name"] = income_inequality_df_filtered["region_name"]
        income_inequality_df_filtered.loc[mask, "name"] = "Friesland (Fryslân)"
        income_inequality_gdf: gpd.GeoDataFrame = create_geodataframe(income_inequality_df_filtered, "geodata/provinces-netherlands.geojson", "name")

        # Create DataFrame for for Province data that is used in the Graph
        income_inequality_df_filtered = filter_dataframe(income_inequality_df, "region_type", ["Municipality"], True)

        if click_data is not None:
            selected_province = click_data["points"][0]["hovertext"]
            income_inequality_df_filtered = filter_dataframe(income_inequality_df, "region_name", ["Nederland", selected_province], False)

        mask = (income_inequality_df_filtered["region_name"] == "Fryslân")
        income_inequality_df_filtered["name"] = income_inequality_df_filtered["region_name"]
        income_inequality_df_filtered.loc[mask, "name"] = "Friesland (Fryslân)"
    else:
        # Create GeoDataframe for Municipality data that is used in the Map
        income_inequality_df_filtered = filter_dataframe(income_inequality_df, "region_type", ["Country", "Province"], True)
        income_inequality_df_filtered["code"] = income_inequality_df_filtered["region_code"].apply((lambda x: x.replace("GM", "")))
        income_inequality_gdf: gpd.GeoDataFrame = create_geodataframe(income_inequality_df_filtered, "geodata/municipalities-netherlands.geojson", "code")

        # Create DataFrame for for Municipality data that is used in the Graph
        income_inequality_df_filtered = filter_dataframe(income_inequality_df, "region_type", ["Province"], True)

        if click_data is not None:
            selected_municipality = click_data["points"][0]["hovertext"]
            income_inequality_df_filtered = filter_dataframe(income_inequality_df, "region_name", ["Nederland", selected_municipality], False)

        income_inequality_df_filtered["code"] = income_inequality_df_filtered["region_code"].apply((lambda x: x.replace("GM", "")))

    income_map = create_income_inequality_map(income_inequality_gdf)
    income_graph = create_income_inequality_graph(income_inequality_df_filtered)

    return income_map, income_graph

@app.callback(
    Output("child_poverty_graph", "figure"),
    Output("child_poverty_histogram", "figure"),
    Input("child_poverty_selected_municipality", "value"),
    Input("child_poverty_selected_year", "value"),
)
def display_child_poverty_graphs(selected_municipality: str, selected_year: str):
    df = pd.read_csv("../data/cleaned/child_poverty/poverty_all_data.csv")
    filtered_df = df[df["Title"] == selected_municipality]

    child_poverty_graph = px.line(
        filtered_df,
        x="Perioden",
        y="KinderenMetKansOpArmoedeRelatief_3",
        title=f"Children with Potential Poverty Relative in {selected_municipality} Over Years",
        labels={"Perioden": "Year", "KinderenMetKansOpArmoedeRelatief_3": "Children with Potential Poverty Relative"}
    )
    child_poverty_graph.update_layout(
        paper_bgcolor="#f0f0f0",
        height=450,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    filtered_df = df[df["Perioden"] == selected_year]
    child_poverty_histogram = px.histogram(
        filtered_df,
        x="Title",
        y="KinderenMetKansOpArmoedeRelatief_3",
        color="Title",
        title=f"Children with Potential Poverty Relative in All Municipalities in {selected_year}",
        labels={"Title": "City", "KinderenMetKansOpArmoedeRelatief_3": "Children with Potential Poverty Relative"}
    )
    child_poverty_histogram.update_layout(
        paper_bgcolor="#f0f0f0",
        height=450,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    return child_poverty_graph, child_poverty_histogram

@app.callback(
    Output("debt_restructuring_graph", "figure"),
    Output("debt_restructuring_histogram", "figure"),
    Input("debt_restructuring_selected_municipality", "value"),
    Input("debt_restructuring_selected_year", "value"),
)
def display_debt_restructuring_graphs(selected_municipality: str, selected_year: str):
    df = pd.read_csv("../data/cleaned/debt_restructuring/debt_all_data.csv")

    filtered_df = df[df["Title"] == selected_municipality]
    debt_restructuring_graph = px.line(
        filtered_df,
        x="Perioden",
        y="PersonenMetUitgesprokenSchuldsanering_1",
        title=f"People with Debt Reconstruction Analysis in {selected_municipality} Over Years",
        labels={"Perioden": "Year", "PersonenMetUitgesprokenSchuldsanering_1": "People with Debt Reconstruction Analysis"}
    )
    debt_restructuring_graph.update_layout(
        paper_bgcolor="#f0f0f0",
        height=450,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    filtered_df = df[df["Perioden"] == selected_year]
    debt_restructuring_histogram = px.histogram(
        filtered_df,
        x="Title",
        y="PersonenMetUitgesprokenSchuldsanering_1",
        color="Title",
        title=f"People with Debt Reconstruction Analysis in All Municipalities in {selected_year}",
        labels={"Title": "City", "PersonenMetUitgesprokenSchuldsanering_1": "People with Debt Reconstruction Analysis"}
    )
    debt_restructuring_histogram.update_layout(
        paper_bgcolor="#f0f0f0",
        height=450,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    return debt_restructuring_graph, debt_restructuring_histogram


@app.callback(
    Output("people_in_debt_map", "figure"),
    Output("people_in_debt_line", "figure"),
    Input("people_in_debt_region", "value"),
    Input("people_in_debt_map", "clickData")
)
def display_people_in_debt_graphs(region: str, click_data):
    """
    Fetches the people in debt data, generates the figures and displays it on the dashboard.
    Gets updated every time the user clicks on the map, or if the region is changed.

    Args:
        region (str): The region type that is selected. Can be either Municipality or Provice.
        click_data: Object that stores information where the user clicked

    Returns:
        Figure, Figure: Two figures are returned, one is the income map, the other the income graph.
    """
    people_in_debt_df: pd.DataFrame = _fetch_data_from_db("people_in_debt")

    if(region == "Province"):
        # Create GeoDataframe for Province data that is used in the Map
        people_in_debt_df_filtered = filter_dataframe(people_in_debt_df, "region_type", ["Country", "Municipality"], True)
        mask = (people_in_debt_df_filtered["region_name"] == "Fryslân")
        people_in_debt_df_filtered["name"] = people_in_debt_df_filtered["region_name"]
        people_in_debt_df_filtered.loc[mask, "name"] = "Friesland (Fryslân)"
        people_in_debt_gdf: gpd.GeoDataFrame = create_geodataframe(people_in_debt_df_filtered, "geodata/provinces-netherlands.geojson", "name")

        # Create DataFrame for for Province data that is used in the Graph
        people_in_debt_df_filtered = filter_dataframe(people_in_debt_df, "region_type", ["Municipality"], True)

        if click_data is not None:
            selected_province = click_data["points"][0]["hovertext"]
            people_in_debt_df_filtered = filter_dataframe(people_in_debt_df, "region_name", ["Nederland", selected_province], False)

        mask = (people_in_debt_df_filtered["region_name"] == "Fryslân")
        people_in_debt_df_filtered["name"] = people_in_debt_df_filtered["region_name"]
        people_in_debt_df_filtered.loc[mask, "name"] = "Friesland (Fryslân)"
    else:
        # Create GeoDataframe for Municipality data that is used in the Map
        people_in_debt_df_filtered = filter_dataframe(people_in_debt_df, "region_type", ["Country", "Province"], True)
        people_in_debt_df_filtered["code"] = people_in_debt_df_filtered["region_code"].apply((lambda x: x.replace("GM", "")))
        people_in_debt_gdf: gpd.GeoDataFrame = create_geodataframe(people_in_debt_df_filtered, "geodata/municipalities-netherlands.geojson", "code")

        # Create DataFrame for for Municipality data that is used in the Graph
        people_in_debt_df_filtered = filter_dataframe(people_in_debt_df, "region_type", ["Province"], True)

        if click_data is not None:
            selected_municipality = click_data["points"][0]["hovertext"]
            people_in_debt_df_filtered = filter_dataframe(people_in_debt_df, "region_name", ["Nederland", selected_municipality], False)

        people_in_debt_df_filtered["code"] = people_in_debt_df_filtered["region_code"].apply((lambda x: x.replace("GM", "")))

    people_in_debt_map = create_people_in_debt_map(people_in_debt_gdf)
    people_in_debt_graph = create_people_in_debt_graph(people_in_debt_df_filtered)

    return people_in_debt_map, people_in_debt_graph


@app.callback(
    Output("property_value_map", "figure"),
    Output("property_value_line", "figure"),
    Input("property_value_region", "value"),
    Input("property_value_map", "clickData")
)
def display_property_value_graphs(region: str, click_data):
    """
    Fetches the income inequality data, generates the figures and displays it on the dashboard.
    Gets updated every time the user clicks on the map, or if the income_region is changed.

    Args:
        income_region (str): The region type that is selected. Can be either Municipality or Provice.
        click_data: Object that stores information where the user clicked

    Returns:
        Figure, Figure: Two figures are returned, one is the income map, the other the income graph.
    """
    property_value_df: pd.DataFrame = _fetch_data_from_db("property_value")

    if(region == "Province"):
        # Create GeoDataframe for Province data that is used in the Map
        property_value_df_filtered = filter_dataframe(property_value_df, "region_type", ["Country", "Municipality"], True)
        mask = (property_value_df_filtered["region_name"] == "Fryslân")
        property_value_df_filtered["name"] = property_value_df_filtered["region_name"]
        property_value_df_filtered.loc[mask, "name"] = "Friesland (Fryslân)"
        property_value_gdf: gpd.GeoDataFrame = create_geodataframe(property_value_df_filtered, "geodata/provinces-netherlands.geojson", "name")

        # Create DataFrame for for Province data that is used in the Graph
        property_value_df_filtered = filter_dataframe(property_value_df, "region_type", ["Municipality"], True)

        if click_data is not None:
            selected_province = click_data["points"][0]["hovertext"]
            property_value_df_filtered = filter_dataframe(property_value_df, "region_name", ["Nederland", selected_province], False)

        mask = (property_value_df_filtered["region_name"] == "Fryslân")
        property_value_df_filtered["name"] = property_value_df_filtered["region_name"]
        property_value_df_filtered.loc[mask, "name"] = "Friesland (Fryslân)"
    else:
        # Create GeoDataframe for Municipality data that is used in the Map
        property_value_df_filtered = filter_dataframe(property_value_df, "region_type", ["Country", "Province"], True)
        property_value_df_filtered["code"] = property_value_df_filtered["region_code"].apply((lambda x: x.replace("GM", "")))
        property_value_gdf: gpd.GeoDataFrame = create_geodataframe(property_value_df_filtered, "geodata/municipalities-netherlands.geojson", "code")

        # Create DataFrame for for Municipality data that is used in the Graph
        property_value_df_filtered = filter_dataframe(property_value_df, "region_type", ["Province"], True)

        if click_data is not None:
            selected_municipality = click_data["points"][0]["hovertext"]
            property_value_df_filtered = filter_dataframe(property_value_df, "region_name", ["Nederland", selected_municipality], False)

        property_value_df_filtered["code"] = property_value_df_filtered["region_code"].apply((lambda x: x.replace("GM", "")))

    property_value_map = create_property_value_map(property_value_gdf)
    property_value_graph = create_property_value_graph(property_value_df_filtered)

    return property_value_map, property_value_graph


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
