import pandas as pd
from dash import dash, dcc, html
from dash.dependencies import Output, Input
import plotly.express as px

app = dash.Dash(__name__)

# Sample dataset (replace with your actual dataset)
years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

df2 = pd.read_csv('./Prediction of debt number in province granularity.csv')

# Reset the index and set the 'years' column
df2['years'] = years
df2 = df2.reset_index(drop=True)

# Print the DataFrame to verify the changes
print(df2)

# Define the layout of the Dash app
app.layout = html.Div([
    html.H1("Multiple-Line Chart by Province"),
    dcc.Graph(id='line-chart'),
])

# Define a callback to update the line chart
@app.callback(
    Output('line-chart', 'figure'),
    Input('line-chart', 'relayoutData')
)
def update_chart(relayoutData):
    fig = px.line(
        df2,
        x='years',
        y=df2.columns[1:],
        labels={'index': 'Year'},
        title="Mean Property Value by Region"
    )
    fig.update_xaxes(
        tickmode='array',
        tickvals=years,  # Set the tick values to the specific years
        showline=True,  # Show x-axis line
        showgrid=False,  # Hide x-axis grid
        showticklabels=True,  # Show x-axis tick labels
        tickformat='d',  # Set tick format to display integers
    )
    fig.update_yaxes(range=[0, 8000])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)