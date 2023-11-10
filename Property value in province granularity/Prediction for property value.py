import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('D:/DDS/CleanData.csv')
print(data)

X_train = data.iloc[1:5, 0].values.reshape(-1, 1)

predictions = {}

for column in data.columns[1:]:
    y_train = data[column].iloc[1:5].values
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions[column] = model.predict([[2024]])


new_row = pd.DataFrame(predictions, index=[2024])
new_row.insert(0, 'Perioden', 2024)
data = pd.concat([data, new_row], ignore_index=True)
print(data)

data.to_csv('Prediction of property value in province and city granularity.csv', index=False)
print(data)
data_province = data.iloc[:, 0:14]
print(data_province)
data_province.to_csv('Prediction of property value in province granularity.csv', index=False)