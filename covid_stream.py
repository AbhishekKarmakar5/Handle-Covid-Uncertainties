import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Read the COVID-19 data
covid_data = pd.read_csv('WHO-COVID-19-global-data.csv')

# Data preprocessing
covid_data['Date_reported'] = pd.to_datetime(covid_data['Date_reported'], format='%d-%m-%Y')
covid_data['Month_Year'] = covid_data['Date_reported'].dt.strftime('%Y-%m')
covid_data = covid_data[['Country', 'Month_Year', 'New_cases', 'New_deaths','Date_reported']]

# Streamlit App
st.title('COVID-19 Analysis Dashboard')

# Select Country
select_country = st.selectbox('Select Country', covid_data['Country'].unique())

# Filter data based on the selected country
selected_country_data = covid_data[covid_data['Country'] == select_country]

# Plotting
st.subheader(f'COVID-19 Analysis for {select_country}')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Monthly Deaths
monthly_deaths = selected_country_data.groupby(['Month_Year'])['New_deaths'].sum()
moving_average_deaths = monthly_deaths.rolling(window=2).mean()
ax1.bar(monthly_deaths.index, monthly_deaths, color='red', label='Monthly Deaths')
ax1.plot(moving_average_deaths.index, moving_average_deaths, color='green', label='Moving Average Deaths (2 months)')
ax1.set_title('Monthly COVID-19 Death Totals with Moving Average')
ax1.set_xlabel('Month-Year')
ax1.set_ylabel('Sum of New Deaths')
ax1.legend()
ax1.tick_params(rotation=45)

# Monthly New Cases
monthly_new_cases = selected_country_data.groupby(['Month_Year'])['New_cases'].sum()
moving_average_cases = monthly_new_cases.rolling(window=2).mean()
ax2.bar(monthly_new_cases.index, monthly_new_cases, color='blue', label='Monthly Cases')
ax2.plot(moving_average_cases.index, moving_average_cases, color='orange', label='Moving Average Cases (2 months)')
ax2.set_title('Monthly COVID-19 Case Totals with Moving Average')
ax2.set_xlabel('Month-Year')
ax2.set_ylabel('Sum of New Cases')
ax2.legend()
ax2.tick_params(rotation=45)

plt.tight_layout()
st.pyplot(fig)

# Sudden Drop Detection
st.subheader('Sudden Drop Detection')

# Find the point of sudden drop in monthly deaths
deaths_values = moving_average_deaths.values
ratio_threshold_deaths = 50
ratios_deaths = []

for i in range(1, len(deaths_values) - 1):
    initial_deaths = deaths_values[i - 1]
    current_deaths = deaths_values[i]
    future_deaths = deaths_values[i + 1]

    ratio_deaths = (abs(initial_deaths - current_deaths)/initial_deaths) * (abs(future_deaths - current_deaths)/current_deaths)

    if np.isinf(ratio_deaths):
        ratio_deaths = 0

    ratios_deaths.append(ratio_deaths)

    if ratio_deaths > ratio_threshold_deaths:
        drop_point_deaths = monthly_deaths.index[i]
        st.warning(f"Sudden drop detected in Monthly Deaths at Month-Year: {drop_point_deaths}, Ratio: {ratio_deaths}")

# Find the point of sudden drop in monthly new cases
cases_values = moving_average_cases.values
ratio_threshold_cases = 30
ratios_cases = []

for i in range(1, len(cases_values) - 1):
    initial_cases = cases_values[i - 1]
    current_cases = cases_values[i]
    future_cases = cases_values[i + 1]

    ratio_cases = (abs(initial_cases - current_cases)/initial_cases) * (abs(future_cases - current_cases)/current_cases)
    if np.isinf(ratio_cases):
        ratio_cases = 0

    ratios_cases.append(ratio_cases)

    if ratio_cases > ratio_threshold_cases:
        drop_point_cases = monthly_new_cases.index[i]
        st.warning(f"Sudden drop detected in Monthly Cases at Month-Year: {drop_point_cases}, Ratio: {ratio_cases}")

# Elbow Method Plots
st.subheader('Elbow Method for Sudden Drop Detection')

# Elbow Method for Deaths
fig_deaths, ax_deaths = plt.subplots(figsize=(10, 4))
ax_deaths.plot(monthly_deaths.index[1:-1], ratios_deaths, marker='o', linestyle='-', color='red')
ax_deaths.set_title('Elbow Method for Sudden Drop Detection in Monthly Deaths')
ax_deaths.set_xlabel('Month-Year')
ax_deaths.set_ylabel('Ratio')
ax_deaths.tick_params(rotation=45)
st.pyplot(fig_deaths)

# Elbow Method for New Cases
fig_cases, ax_cases = plt.subplots(figsize=(10, 4))
ax_cases.plot(monthly_new_cases.index[1:-1], ratios_cases, marker='o', linestyle='-', color='blue')
ax_cases.set_title('Elbow Method for Sudden Drop Detection in Monthly Cases')
ax_cases.set_xlabel('Month-Year')
ax_cases.set_ylabel('Ratio')
ax_cases.tick_params(rotation=45)
st.pyplot(fig_cases)


### ---------------------------------------------- World Plot----------------------------------------------------
# Select Month
select_month = st.selectbox('Select Month', covid_data['Month_Year'].unique())

# Filter data based on the selected month
selected_month_data = covid_data[covid_data['Month_Year'] == select_month]

# Plotting Monthly Deaths and New Cases
st.subheader(f'COVID-19 Analysis for {select_month}')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Calculate the mean of new cases and deaths for each country and month
mean_new_cases = selected_month_data.groupby(['Country'])['New_cases'].mean().reset_index()
mean_new_deaths = selected_month_data.groupby(['Country'])['New_deaths'].mean().reset_index()

# Merge with a world map shapefile (e.g., Natural Earth shapefile)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.rename(columns={'name': 'Country'})

# Merge COVID-19 data with world map
merged_cases = pd.merge(world, mean_new_cases, how='left', on='Country')
merged_deaths = pd.merge(world, mean_new_deaths, how='left', on='Country')

# Plotting mean of new cases
fig_mean_cases, ax_mean_cases = plt.subplots(figsize=(15, 5))
merged_cases.plot(column='New_cases', cmap='OrRd', linewidth=0.8, ax=ax_mean_cases, edgecolor='0.8', legend=True)
ax_mean_cases.set_title(f'Mean of New Cases by Country - {select_month}')
ax_mean_cases.set_axis_off()
st.pyplot(fig_mean_cases)

# Plotting mean of deaths
fig_mean_deaths, ax_mean_deaths = plt.subplots(figsize=(15, 5))
merged_deaths.plot(column='New_deaths', cmap='OrRd', linewidth=0.8, ax=ax_mean_deaths, edgecolor='0.8', legend=True)
ax_mean_deaths.set_title(f'Mean of Deaths by Country - {select_month}')
ax_mean_deaths.set_axis_off()
st.pyplot(fig_mean_deaths)


###------------------------------------------- see the plot and find out the moth of select -------------------------------
# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Streamlit application
st.title('COVID-19 New Cases Over Time')

# Assuming 'select_country' is already defined in your Streamlit application
# select_country = st.selectbox('Select a Country', covid_data['Country'].unique())

# Filter data based on the selected country
selected_country_data = covid_data[covid_data['Country'] == select_country]

# Convert 'Date_reported' to datetime if it's not already
selected_country_data['Date_reported'] = pd.to_datetime(selected_country_data['Date_reported'])

# Plotting
plt.figure(figsize=(35, 9))  # Adjust the figsize for better visibility
plt.plot(selected_country_data['Date_reported'], selected_country_data['New_cases'], label=select_country, color='blue', linewidth=2)
plt.title(f'COVID-19 New Cases Over Time in {select_country}')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()

# Set x-axis limits for all months
plt.xlim(selected_country_data['Date_reported'].min(), selected_country_data['Date_reported'].max())

# Set the x-axis ticks to show all months with a larger gap
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonthday=-1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as year-month

st.pyplot(plt)

###----------------------take input of select month-year for each country -----------------------------------------------------------
# Input for selecting month-year for prediction
Select_month = st.selectbox('Select Month-Year for Prediction', covid_data['Month_Year'].unique())
Select_country = st.selectbox('Select Country for Prediction', covid_data['Country'].unique())

# Button to trigger the prediction
if st.button('Predict New Cases'):
    # Data preprocessing for the model
    train_test_data = covid_data[(covid_data['Month_Year'] < Select_month) & (covid_data['Country'] == Select_country)]
    train_data, test_data = train_test_split(train_test_data, test_size=0.2, shuffle=False)
    validation_data = covid_data[(covid_data['Month_Year'] >= Select_month) & (covid_data['Country'] == Select_country)]

    # Normalize the data
    scaler = MinMaxScaler()
    train_data_normalized = scaler.fit_transform(train_data[['New_cases']])
    test_data_normalized = scaler.transform(test_data[['New_cases']])
    validation_data_normalized = scaler.transform(validation_data[['New_cases']])

    # Create sequences for LSTM
    def create_sequences(data, sequence_length):
        sequences, targets = [], []
        for i in range(len(data) - sequence_length):
            sequence = data[i:i + sequence_length]
            target = data[i + sequence_length:i + sequence_length + 1]
            sequences.append(sequence)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    sequence_length = 10  # Adjust based on your data and model performance

    X_train, y_train = create_sequences(train_data_normalized, sequence_length)
    X_test, y_test = create_sequences(test_data_normalized, sequence_length)
    X_validation, y_validation = create_sequences(validation_data_normalized, sequence_length)

    # Build and train the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

    # Evaluate and make predictions
    validation_predictions = model.predict(X_validation)
    y_validation_reshaped = y_validation.reshape(-1, 1)
    validation_predictions = scaler.inverse_transform(validation_predictions)
    y_validation_actual = scaler.inverse_transform(y_validation_reshaped)

    # Calculate metrics
    mae = mean_absolute_error(y_validation_actual, validation_predictions)
    mse = mean_squared_error(y_validation_actual, validation_predictions)
    rmse = mean_squared_error(y_validation_actual, validation_predictions, squared=False)
    r_squared = r2_score(y_validation_actual, validation_predictions)
    n_observations = len(y_validation_actual)
    n_predictors = X_validation.shape[1]
    adjusted_r_squared = 1 - (1 - r_squared) * (n_observations - 1) / (n_observations - n_predictors - 1)

    # Display metrics
    st.write(f'Mean Squared Error on Validation Data: {mse}')
    st.write(f'Root Mean Squared Error on Validation Data: {rmse}')
    st.write(f'Mean Absolute Error on Validation Data: {mae}')
    st.write(f'R-squared on Validation Data: {r_squared}')
    st.write(f'Adjusted R-squared on Validation Data: {adjusted_r_squared}')

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(y_validation_actual, label='Actual')
    plt.plot(validation_predictions, label='Predicted')
    plt.title('Validation Data: Actual vs. Predicted New Cases')
    plt.xlabel('Time')
    plt.ylabel('New Cases')
    plt.legend()
    st.pyplot(plt)




