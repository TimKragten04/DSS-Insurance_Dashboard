#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Children Poverty Histogram
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Read the data
df = pd.read_csv('/code/data/poverty_sum_data.csv')

# Create a Dash application
app = dash.Dash(__name__)

# Create a colored histogram with Plotly, with different colors for adjacent cities
fig = px.histogram(
    df,
    x='Title',
    y='KinderenMetKansOpArmoedeRelatief_3',
    color='Title',  # Assign a different color for each bar based on the city name from the 'Title' column
    title='Children with Potential Poverty Relative per City',
    labels={'Title': 'City', 'KinderenMetKansOpArmoedeRelatief_3': 'Children with Potential Poverty Relative'},
    # Choose a built-in Plotly color cycle to ensure a variety of colors
    color_discrete_sequence=px.colors.qualitative.Plotly
)

# Adjust the size of the chart
fig.update_layout(
    autosize=False,
    width=1200,   # Width in pixels
    height=600    # Height in pixels
)

# If the x-axis labels are too crowded, rotate the labels or increase the interval
fig.update_xaxes(tickangle=45)

# Define the layout of the application
app.layout = html.Div([
    html.H1('Poverty Analysis by City'),
    html.Div('Dash: A web application framework for Python.'),
    dcc.Graph(id='example-graph', figure=fig)
])

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)


# In[120]:


# Debt Histogram
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Read the data
df = pd.read_csv('/code/data/debt_sum_data.csv')

# Create a Dash application
app = dash.Dash(__name__)

# Create a colored histogram with Plotly, with different colors for adjacent cities
fig = px.histogram(
    df,
    x='Title',
    y='PersonenMetUitgesprokenSchuldsanering_1',
    color='Title',  # Assign a different color for each bar based on the city name from the 'Title' column
    title='People with Pronounced Debt Rescheduling per City',
    labels={'Title': 'City', 'PersonenMetUitgesprokenSchuldsanering_1': 'People with Pronounced Debt Rescheduling'},
    # Choose a built-in Plotly color cycle to ensure a variety of colors
    color_discrete_sequence=px.colors.qualitative.Plotly
)

# Adjust the size of the chart
fig.update_layout(
    autosize=False,
    width=1200,   # Width in pixels
    height=600    # Height in pixels
)

# If the x-axis labels are too crowded, rotate the labels or increase the interval
fig.update_xaxes(tickangle=45)

# Define the layout of the application
app.layout = html.Div([
    html.H1('Debt Analysis by City'),
    html.Div('Dash: A web application framework for Python.'),
    dcc.Graph(id='example-graph', figure=fig)
])

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)






