# Ex-08-Modeling-and-Forecasting-with-ARMA-Processes
## NAME : Paladugu Venkata Balaji
## REG NO : 2305001024
## Aim
To simulate an ARMA time series, fit ARMA models, forecast future values, and compare model performance using AIC, MAE, MSE, and RMSE.

## Objectives
1. To generate an ARMA time series.
2. To plot the series and its ACF and PACF.
3. To split data into training and testing sets.
4. To fit and compare different ARMA models.
5. To forecast future values.
6. To calculate MAE, MSE, and RMSE.
7. To generate an ARMA(2,1) process.
## Program
~~~python
# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 2: Generate ARMA(1,1) Data
np.random.seed(42)
ar = np.array([1, -0.7])
ma = np.array([1, 0.4])
data = ArmaProcess(ar, ma).generate_sample(200)
series = pd.Series(data)

# Step 3: Plot Time Series
plt.plot(series)
plt.title("ARMA(1,1) Time Series")
plt.show()

# Step 4: Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_acf(series, lags=20, ax=ax[0])
plot_pacf(series, lags=20, method="ywm", ax=ax[1])
plt.show()

# Step 5: Split Data
train = series[:160]
test = series[160:]
print("Training Data:", len(train))
print("Testing Data:", len(test))

# Step 6: Fit ARMA(1,1) Model
model = ARIMA(train, order=(1, 0, 1)).fit()
print(model.summary())

# Step 7: Forecast Future Values
forecast = model.forecast(steps=40)
print("Forecast:")
print(forecast)

# Step 8: Calculate Forecast Errors
print("MAE:", mean_absolute_error(test, forecast))
mse = mean_squared_error(test, forecast)
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))

# Step 9: Plot Actual and Forecast Values
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, forecast, label="Forecast")
plt.title("Actual vs Forecast")
plt.legend()
plt.show()

# Step 10: Compare Different ARMA Models
orders = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1)]

for order in orders:
    fit = ARIMA(train, order=order).fit()
    print("ARMA Order:", order)
    print("AIC:", fit.aic)
    print("BIC:", fit.bic)

# Step 11: Generate ARMA(2,1) Data
ar2 = np.array([1, -0.5, -0.2])
ma2 = np.array([1, 0.4])
data2 = ArmaProcess(ar2, ma2).generate_sample(200)

plt.plot(data2)
plt.title("ARMA(2,1) Time Series")
plt.show()

# Step 12: Fit ARMA(2,1) and Forecast
model2 = ARIMA(data2, order=(2, 0, 1)).fit()
forecast2 = model2.forecast(steps=10)

print("ARMA(2,1) Forecast:")
print(forecast2)

~~~

## Output
<img width="856" height="550" alt="image" src="https://github.com/user-attachments/assets/c1032e06-87e3-4117-9ef1-3bb0a030cdb8" />

<img width="1105" height="587" alt="image" src="https://github.com/user-attachments/assets/e700379a-f314-4b39-a947-6dcbb79e250b" />

<img width="1025" height="558" alt="image" src="https://github.com/user-attachments/assets/03f0287d-1fff-4faf-a5b2-07e3d01ef855" />
<img width="800" height="870" alt="image" src="https://github.com/user-attachments/assets/f402b139-c629-4aa9-b5a3-5b2fcdaad5ad" />
<img width="987" height="547" alt="image" src="https://github.com/user-attachments/assets/24183039-7a86-4fe1-9e10-bb97a328db8e" />
<img width="862" height="642" alt="image" src="https://github.com/user-attachments/assets/9b2d574b-c2fd-4e65-9cb2-826b7c11e7a8" />

## Result
Thus, ARMA processes were generated and analyzed, different ARMA models were fitted and compared using AIC and BIC, and forecasting performance was evaluated using MAE, MSE, and RMSE. The best forecasting model was identified based on the lowest RMSE, and an ARMA(2,1) process was successfully generated and forecasted.

