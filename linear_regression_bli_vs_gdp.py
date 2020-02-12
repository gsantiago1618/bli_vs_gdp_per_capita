import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# For Linear regression models
import sklearn.linear_model

# for KNN models
import sklearn.neighbors


# Load data
oecd_bli = pd.read_csv("oecd_bli_2019.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",
                             thousands=',',
                             delimiter='\t',
                             encoding='latin1', na_values="n/a")

# TODO: Prepare the data
def prepare_country_stats(bli, gdp):
    


country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select a linear model
# model_lr = sklearn.linear_model.LinearRegression()

# Select a KNN model
model_knn = sklearn.KNeighborsRegressor(n_neighbors=3)

# Train the model_lr or model_knn
# model_lr.fit(X, y)
model_knn.fit(X, y)

# Make a prediction for country A
sample_country_A_gdp = [[22587]]
print(model_knn.predict(sample_country_A_gdp))

