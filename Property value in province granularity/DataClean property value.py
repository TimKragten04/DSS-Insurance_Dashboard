import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import re
import numpy as np
df_csv1 = pd.read_csv('C:/Users/zhouj/Downloads/table.csv')
data = df_csv1

data = data.iloc[3:379]

data.columns = data.iloc[0]

data = data.reset_index(drop=True)

data = data.T

data = data.reset_index(drop=True)

data.columns = data.iloc[0]

data = data.drop(0)

data.drop('Eigendom', axis = 1, inplace=True)
print(data)

data.drop("Regio's", axis = 1, inplace=True)


data['Perioden'] = data['Perioden'].apply(lambda x: re.sub(r'[^0-9]', '', x))

print(data)


print(data)

def predict_missing_values(data):
    for column_name in data.columns:
        column = data[column_name]
        for row_index in column.index:
            value = column[row_index]
            if pd.isna(value):  # Check if the value is missing
                # Filter non-missing values from the current column
                non_missing_data = data[data[column_name].notna()]
                X = non_missing_data.index.to_numpy().reshape(-1, 1)
                y = non_missing_data[column_name].to_numpy()

                if len(X) > 0:  # Ensure there are non-missing values to fit the model
                    # Fit a linear regression model
                    model = LinearRegression()
                    model.fit(X, y)

                    # Predict the missing value
                    predicted_value = model.predict([[row_index]])[0]
                    data.at[row_index, column_name] = predicted_value

    return data

# Call the function to predict and replace missing values
data = predict_missing_values(data)

for column in data.columns:
    data[column] = data[column].astype(int)

data_province = data.iloc[:, 0:14]

data_province.set_index('Perioden', inplace=True)
print(data_province)
data.to_csv('CleanData.csv', index=False)
data_province.to_csv('data_province.csv', index=False)