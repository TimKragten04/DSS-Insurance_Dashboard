#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd


# In[ ]:


#Graph of Children Poverty Analysis
df = pd.read_csv('code/data/poverty_all_data.csv')

# Create a Dash application
app = dash.Dash(__name__)

# Application layout
app.layout = html.Div([
    html.H1("Children with Potential Poverty Analysis"),

    # City selector for line chart
    html.Div([
        html.Label("Select City for Line Chart:"),
        dcc.Dropdown(
            id='city-selector-line',
            options=[{'label': city, 'value': city} for city in df['Title'].unique()],
            value=df['Title'].unique()[0],
            clearable=False
        ),
    ]),
    dcc.Graph(id='line-chart'),

    # Year selector for histogram
    html.Div([
        html.Label("Select Year for Histogram:"),
        dcc.Dropdown(
            id='year-selector-histogram',
            options=[{'label': year, 'value': year} for year in df['Perioden'].unique()],
            value=df['Perioden'].unique()[0],
            clearable=False
        ),
    ]),
    dcc.Graph(id='histogram')
])

# Callback function to update the line chart
@app.callback(
    Output('line-chart', 'figure'),
    [Input('city-selector-line', 'value')]
)
def update_line_chart(selected_city):
    filtered_df = df[df['Title'] == selected_city]
    fig = px.line(
        filtered_df,
        x='Perioden',
        y='KinderenMetKansOpArmoedeRelatief_3',
        title=f'Children with Potential Poverty Relative in {selected_city} Over Years',
        labels={'Perioden': 'Year', 'KinderenMetKansOpArmoedeRelatief_3': 'Children with Potential Poverty Relative'}
    )
    return fig

# Callback function to update the histogram
@app.callback(
    Output('histogram', 'figure'),
    [Input('year-selector-histogram', 'value')]
)
def update_histogram(selected_year):
    filtered_df = df[df['Perioden'] == selected_year]
    fig = px.histogram(
        filtered_df,
        x='Title',
        y='KinderenMetKansOpArmoedeRelatief_3',
        color='Title',
        title=f'Children with Potential Poverty Relative in All Cities in {selected_year}',
        labels={'Title': 'City', 'KinderenMetKansOpArmoedeRelatief_3': 'Children with Potential Poverty Relative'}
    )
    return fig

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:


#Graph of People with Debt Reconstruction Analysis
df = pd.read_csv('/Users/jess/Desktop/DSS/debt_all_data.csv')

# Create a Dash application
app = dash.Dash(__name__)

# Application layout
app.layout = html.Div([
    html.H1("People with Debt Reconstruction Analysis"),

    # City selector for the line chart
    html.Div([
        html.Label("Select City for Line Chart:"),
        dcc.Dropdown(
            id='city-selector-line',
            options=[{'label': city, 'value': city} for city in df['Title'].unique()],
            value=df['Title'].unique()[0],
            clearable=False
        ),
    ]),
    dcc.Graph(id='line-chart'),

    # Year selector for the histogram
    html.Div([
        html.Label("Select Year for Histogram:"),
        dcc.Dropdown(
            id='year-selector-histogram',
            options=[{'label': year, 'value': year} for year in df['Perioden'].unique()],
            value=df['Perioden'].unique()[0],
            clearable=False
        ),
    ]),
    dcc.Graph(id='histogram')
])

# Callback function to update the line chart
@app.callback(
    Output('line-chart', 'figure'),
    [Input('city-selector-line', 'value')]
)
def update_line_chart(selected_city):
    filtered_df = df[df['Title'] == selected_city]
    fig = px.line(
        filtered_df,
        x='Perioden',
        y='PersonenMetUitgesprokenSchuldsanering_1',
        title=f'People with Debt Reconstruction Analysis in {selected_city} Over Years',
        labels={'Perioden': 'Year', 'PersonenMetUitgesprokenSchuldsanering_1': 'People with Debt Reconstruction Analysis'}
    )
    return fig

# Callback function to update the histogram
@app.callback(
    Output('histogram', 'figure'),
    [Input('year-selector-histogram', 'value')]
)
def update_histogram(selected_year):
    filtered_df = df[df['Perioden'] == selected_year]
    fig = px.histogram(
        filtered_df,
        x='Title',
        y='PersonenMetUitgesprokenSchuldsanering_1',
        color='Title',
        title=f'People with Debt Reconstruction Analysis in All Cities in {selected_year}',
        labels={'Title': 'City', 'PersonenMetUitgesprokenSchuldsanering_1': 'People with Debt Reconstruction Analysis'}
    )
    return fig

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)

