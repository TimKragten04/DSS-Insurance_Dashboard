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
    engine = sa.create_engine("postgresql://student:infomdss@db_dashboard:5432/dashboard")
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
        coloraxis_colorbar_title="Avgerage Minus<br>Median Income",
        paper_bgcolor="#CCCCCC",
        geo=dict(
            bgcolor="#e5ecf6"
        )
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
        paper_bgcolor="#CCCCCC",
    )

    return fig


app = Dash(__name__)

app.layout = html.Div([
    html.H4("Income Inequality by Municipality"),
    html.P("Select region:"),
    dcc.RadioItems(
        id="income_region",
        options=["Province", "Municipality"],
        value="Municipality",
        inline=True
    ),
    html.Div([
        dcc.Graph(id="income_map"),
    ], style={"width": "49%", "display": "inline-block"}),
    html.Div([
        dcc.Graph(id="income_graph"),
    ], style={"width": "49%", "display": "inline-block"}),
    
])


@app.callback(
    Output("income_map", "figure"),
    Output("income_graph", "figure"),
    Input("income_region", "value"),
    Input("income_map", "clickData")
)
def display_income_graphs(income_region: str, click_data):
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

    if(income_region == "Province"):
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


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
