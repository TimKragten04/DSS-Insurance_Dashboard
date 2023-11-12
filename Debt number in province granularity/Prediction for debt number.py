import pandas as pd
from sklearn.linear_model import LinearRegression


data = pd.read_csv('D:/DDS/venv/debt.csv')
print(data)

X_train = data.iloc[1:10, 0].values.reshape(-1, 1)

predictions = {}

for column in data.columns[1:]:
    y_train = data[column].iloc[1:10].values
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions[column] = model.predict([[2023]])

new_row = pd.DataFrame(predictions, index=[2023])
new_row.insert(0, 'years', 2023)
data = pd.concat([data, new_row], ignore_index=True)
print(data)

data.to_csv('Prediction of debt number in province and city granularity.csv', index=False)
print(data)
data_province = data.iloc[:, 0:13]
print(data_province)
data_province.to_csv('Prediction of debt number in province granularity.csv', index=False)