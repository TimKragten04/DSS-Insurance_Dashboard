import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

app = dash.Dash(__name__)

# Sample dataset (replace with your actual dataset)

# years = [2019, 2020, 2021, 2022, 2023]

df2 = pd.read_csv('D:/DDS/venv/debt.csv')
first_column = df2.iloc[:, 0]
print(first_column)


print(df2)
# df2.set_index(years, drop=True, append=False, inplace=False, verify_integrity=False)
# print(df2)
# Define the layout of the Dash app
app.layout = html.Div([
    html.H1("Number of people in debt by Region"),
    dcc.Graph(id='line-chart'),
])

# Define a callback to update the line chart
@app.callback(
    dash.dependencies.Output('line-chart', 'figure'),
    dash.dependencies.Input('line-chart', 'relayoutData')
)
def update_chart(relayoutData):
    fig = px.line(
        df2,
        x='years',
        y=df2.columns[1:13],
        labels={'index': 'Year'},
        title="Number of people in debt by Region"
    )
    fig.update_xaxes(
        tickmode='array',
        tickvals=first_column,  # Set the tick values to the specific years
        showline=True,  # Show x-axis line
        showgrid=False,  # Hide x-axis grid
        showticklabels=True,  # Show x-axis tick labels
        tickformat='d',  # Set tick format to display integers
    )
    fig.update_yaxes(range=[0,41000])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
