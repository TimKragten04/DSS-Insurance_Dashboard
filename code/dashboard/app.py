import dash
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import sqlalchemy as sa
from functools import reduce
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


def create_income_map(geo_df: gpd.GeoDataFrame):
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
        coloraxis_colorbar_title="Avgerage Minus<br>Median Income",
        geo=dict(
            bgcolor="#e5ecf6"
        ),
        height=300,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )
    fig.update_geos(
        visible=False,
        fitbounds="locations"
    )

    return fig


def create_income_graph(df: pd.DataFrame):
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
        height=300,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    return fig


app = Dash(__name__)
df = pd.read_csv('../data/cleaned/child_poverty/poverty_all_data.csv')

app.layout = html.Div([
    html.Div([
        html.H4("Crime and Wealth Statistics"),
        html.P("Select region:"),
        dcc.RadioItems(
            id="region",
            options=["Province", "Municipality"],
            value="Municipality",
            inline=False
        ),
    ], style={"width": "10%", "display": "inline-block", "vertical-align": "top", "magin": "0", "padding": "0"}),
    html.Div([
        dcc.Graph(id="graph_1"),
    ], style={"width": "45%", "display": "inline-block"}),
    html.Div([
        dcc.Graph(id="graph_2"),
    ], style={"width": "45%", "display": "inline-block"}),

    html.Div([
        html.H4("Other Statistics"),
        html.P("Select dataset:"),
        dcc.RadioItems(
            id="dataset_type",
            options=["Children poverty", "Debt reconstruction"],
            value="Children poverty",
            inline=False
        ),
        html.P("Select City for Line Chart:"),
        dcc.Dropdown(
            id='selected_city',
            options=[{'label': city, 'value': city} for city in df['Title'].unique()],
            value=df['Title'].unique()[0],
            clearable=False
        ),
        html.P("Select Year for Histogram:"),
        dcc.Dropdown(
            id='selected_year',
            options=[{'label': year, 'value': year} for year in df['Perioden'].unique()],
            value=df['Perioden'].unique()[0],
            clearable=False
        ),
    ], style={"width": "10%", "display": "inline-block", "vertical-align": "top", "magin": "0", "padding": "0"}),
    html.Div([
        dcc.Graph(id="graph_3"),
    ], style={"width": "45%", "display": "inline-block"}),
    html.Div([
        dcc.Graph(id="graph_4"),
    ], style={"width": "45%", "display": "inline-block"}),
    
])


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
    income_inequality_df: pd.DataFrame = _fetch_data_from_db("income_inequality")

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

    income_map = create_income_map(income_inequality_gdf)
    income_graph = create_income_graph(income_inequality_df_filtered)

    return income_map, income_graph


def display_child_poverty_graphs(selected_city: str, selected_year: str):
    df = pd.read_csv('../data/cleaned/child_poverty/poverty_all_data.csv')
    filtered_df = df[df['Title'] == selected_city]

    child_poverty_graph = px.line(
        filtered_df,
        x='Perioden',
        y='KinderenMetKansOpArmoedeRelatief_3',
        title=f'Children with Potential Poverty Relative in {selected_city} Over Years',
        labels={'Perioden': 'Year', 'KinderenMetKansOpArmoedeRelatief_3': 'Children with Potential Poverty Relative'}
    )

    filtered_df = df[df['Perioden'] == selected_year]
    child_poverty_histogram = px.histogram(
        filtered_df,
        x='Title',
        y='KinderenMetKansOpArmoedeRelatief_3',
        color='Title',
        title=f'Children with Potential Poverty Relative in All Cities in {selected_year}',
        labels={'Title': 'City', 'KinderenMetKansOpArmoedeRelatief_3': 'Children with Potential Poverty Relative'}
    )

    return child_poverty_graph, child_poverty_histogram


def display_debt_reconstruction_graphs(selected_city: str, selected_year: str):
    df = pd.read_csv('../data/cleaned/debt_reconstruction/debt_all_data.csv')

    filtered_df = df[df['Title'] == selected_city]
    debt_reconstruction_graph = px.line(
        filtered_df,
        x='Perioden',
        y='PersonenMetUitgesprokenSchuldsanering_1',
        title=f'People with Debt Reconstruction Analysis in {selected_city} Over Years',
        labels={'Perioden': 'Year', 'PersonenMetUitgesprokenSchuldsanering_1': 'People with Debt Reconstruction Analysis'}
    )

    filtered_df = df[df['Perioden'] == selected_year]
    debt_reconstruction_histogram = px.histogram(
        filtered_df,
        x='Title',
        y='PersonenMetUitgesprokenSchuldsanering_1',
        color='Title',
        title=f'People with Debt Reconstruction Analysis in All Cities in {selected_year}',
        labels={'Title': 'City', 'PersonenMetUitgesprokenSchuldsanering_1': 'People with Debt Reconstruction Analysis'}
    )

    return debt_reconstruction_graph, debt_reconstruction_histogram


@app.callback(
    Output("graph_1", "figure"),
    Output("graph_2", "figure"),
    Input("region", "value"),
    Input("graph_1", "clickData")
)
def display_top_graphs(region: str, click_data):
    """_summary_

    Args:
        income_region (str): _description_
        dataset_type (str): _description_
        click_data (_type_): _description_
    """
    return display_income_inequality_graphs(region, click_data)
        

@app.callback(
    Output("graph_3", "figure"),
    Output("graph_4", "figure"),
    Input("selected_city", "value"),
    Input("selected_year", "value"),
    Input("dataset_type", "value"),
)
def display_bottom_graphs(selected_city: str, selected_year: str, dataset_type: str):
    """_summary_

    Args:
        income_region (str): _description_
        dataset_type (str): _description_
        click_data (_type_): _description_
    """
    match dataset_type:
        case "Children poverty":
            return display_child_poverty_graphs(selected_city, selected_year)
        case "Debt reconstruction":
            return display_debt_reconstruction_graphs(selected_city, selected_year)
    


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
