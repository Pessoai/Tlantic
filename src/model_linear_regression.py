import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

csv_path = os.path.join('..', 'data', 'data_set_final_processed.csv')

df = pd.read_csv(csv_path, low_memory=False)

df.info()

# Keep the original dataframe separate
original_df = df.copy()

# Convert 'DATA_VENDA' to datetime and handle missing values
df['DATA_VENDA'] = pd.to_datetime(df['DATA_VENDA'], errors='coerce')
df['TIPO_FERIADO'] = df['TIPO_FERIADO'].fillna('None')

# Sort the DataFrame
df.sort_values(['LOJA', 'DATA_VENDA'], inplace=True)

# Create lag features and rolling means
df['lag_7'] = df.groupby('LOJA')['VALOR_VENDA'].shift(7)
df['rolling_mean_15'] = df.groupby('LOJA')['VALOR_VENDA'].shift(1).rolling(15).mean()

# Fill missing values in lag and rolling mean features
df.fillna({'lag_7': 0}, inplace=True)
df.fillna({'rolling_mean_15': df['rolling_mean_15'].mean()}, inplace=True)

# Drop unnecessary columns
columns_to_drop = ['DATA_VENDA','ABERTURA_LOJA', 'FECHO_LOJA', 'DATA_FERIADO', 'ITEMS', 'rolling_avg_items', 'TEMPO_ABERTURA', 'LOJA']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Check for null values
null_counts = df.isnull().sum()
print("Null values in each column:")
print(null_counts)

# Handle missing values in the rest of the dataset

# Drop columns with a high number of null values (threshold: 50% null values)
df = df.dropna(thresh=len(df) * 0.5, axis=1)

# Define features and target variable
X = df.drop(columns=['VALOR_VENDA'])
y = df['VALOR_VENDA']

# Fill null values with median for numeric columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())

# Fill any remaining null values in categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
X[categorical_columns] = X[categorical_columns].ffill()

# Ensure that 'lag_7' and 'rolling_mean_15' are included in the features
print("Features in X:")
print(X.columns)

# Ensure there are no remaining NaN values
X = X.fillna(0)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Features:")
print(X_train)
print("Test Features:")
print(X_test)
print("Training Target:")
print(y_train)
print("Test Target:")
print(y_test)

# Define preprocessing for numerical and categorical data
numeric_features = numeric_columns
categorical_features = categorical_columns

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Plotting the results

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Line of equality
plt.show()

# Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# Histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

# Create a DataFrame for y_test and y_pred to easily handle the dates
results_df = pd.DataFrame({
    'DATA_VENDA': original_df.loc[y_test.index, 'DATA_VENDA'],
    'Actual': y_test,
    'Predicted': y_pred
})

# Sort by date to ensure proper plotting
results_df.sort_values('DATA_VENDA', inplace=True)

# Plot actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(results_df['DATA_VENDA'], results_df['Actual'], label='Actual VALOR_VENDA')
plt.plot(results_df['DATA_VENDA'], results_df['Predicted'], label='Predicted VALOR_VENDA', linestyle='--')
plt.xlabel('Date')
plt.ylabel('VALOR_VENDA')
plt.title('Actual vs Predicted VALOR_VENDA')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='Actual', fill=True)
sns.kdeplot(y_pred, label='Predicted', fill=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Density Plot of Actual vs Predicted Values')
plt.legend()
plt.show()

# Plot 1: TIPO_FERIADO vs VALOR_VENDA
plt.figure(figsize=(12, 6))
sns.boxplot(x='TIPO_FERIADO', y='VALOR_VENDA', data=df)
plt.xticks(rotation=90)
plt.title('Impacto do TIPO_FERIADO no VALOR_VENDA')
plt.show()

# Plot 3: CIDADE vs VALOR_VENDA
plt.figure(figsize=(12, 6))
sns.boxplot(x='CIDADE', y='VALOR_VENDA', data=df)
plt.xticks(rotation=90)
plt.title('Impacto da CIDADE no VALOR_VENDA')
plt.show()

# Plot 4: REGIAO vs VALOR_VENDA
plt.figure(figsize=(12, 6))
sns.boxplot(x='REGIAO', y='VALOR_VENDA', data=df)
plt.xticks(rotation=90)
plt.title('Impacto da REGIAO no VALOR_VENDA')
plt.show()

# Ensure 'DATA_VENDA' is in datetime format
results_df['DATA_VENDA'] = pd.to_datetime(results_df['DATA_VENDA'], errors='coerce')

# Identify the last date in the DataFrame
last_date = results_df['DATA_VENDA'].max()

# Calculate the start date (one month before the last date)
start_date = last_date - pd.DateOffset(months=1)

# Filter the DataFrame to only include the last month's data
results_df_last_month = results_df[results_df['DATA_VENDA'] >= start_date]

# Plot actual vs predicted values for the last month
plt.figure(figsize=(14, 7))
plt.plot(results_df_last_month['DATA_VENDA'], results_df_last_month['Actual'], label='Actual VALOR_VENDA', marker='o')
plt.plot(results_df_last_month['DATA_VENDA'], results_df_last_month['Predicted'], label='Predicted VALOR_VENDA', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('VALOR_VENDA')
plt.title('Actual vs Predicted VALOR_VENDA (Last Month)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Calculate the absolute differences between actual and predicted values
results_df['Difference'] = abs(results_df['Actual'] - results_df['Predicted'])

# Sort the DataFrame by the difference in descending order
results_df_sorted = results_df.sort_values(by='Difference', ascending=False)

# Select the top N points where the difference is highest (e.g., top 10)
top_n = 10
results_df_top_n = results_df_sorted.head(top_n)

# Plot the top N points with the highest differences
plt.figure(figsize=(14, 7))
plt.plot(results_df_top_n['DATA_VENDA'], results_df_top_n['Actual'], label='Actual VALOR_VENDA', marker='o', linestyle='None')
plt.plot(results_df_top_n['DATA_VENDA'], results_df_top_n['Predicted'], label='Predicted VALOR_VENDA', marker='x', linestyle='None')
plt.xlabel('Date')
plt.ylabel('VALOR_VENDA')
plt.title('Top Differences Between Actual and Predicted VALOR_VENDA')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Checking feature importance

# Extract feature names after one-hot encoding
categorical_features_encoded = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_features = np.concatenate([numeric_features, categorical_features_encoded])

# Extract the coefficients
coefficients = model.named_steps['regressor'].coef_

# Create a DataFrame to hold feature names and their corresponding coefficients
feature_importances = pd.DataFrame({'Feature': all_features, 'Importance': coefficients})

# Sort the features by absolute value of their coefficients
feature_importances['Absolute_Importance'] = feature_importances['Importance'].abs()
feature_importances = feature_importances.sort_values(by='Absolute_Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(20, 70))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.title('Feature Importances')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()