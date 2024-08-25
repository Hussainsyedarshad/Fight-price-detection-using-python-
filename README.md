# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, r2_score

from IPython.core.display import display, HTML

# %%
df = pd.read_csv("/kaggle/input/flight-price-data/flight_dataset.csv")

# %%
df.head()

# %%
df.shape

# %%
df.describe()

# %% [markdown]
# # Observing for NULL values

# %%
df.isna().sum()

# %% [markdown]
# # Creating all possible routes variations and total number of minutes for a flight

# %%
df["routes"] = df["Source"] + "->" + df["Destination"]
df["Total flight duration"] = df["Duration_hours"]*60 + df["Duration_min"]

# %% [markdown]
# # 10 most popular and used airlines, routes and months of travel

# %%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))

for i, j in enumerate(["Airline", "routes", "Month"]):
    count = df[j].value_counts()
    if j == "Month":
        count.plot(kind="bar", ax=axes[i])
    else:
        count[:10].plot(kind="bar", ax=axes[i])
    for container in axes[i].containers:
        axes[i].bar_label(container)
    axes[i].set_yticklabels(())
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].set_title(j.capitalize())
plt.tight_layout()
plt.show()

# %% [markdown]
# # Data distribution for prices, total flight duration in minutes and density for the days of months and time of the day that have the most amount of flights scheduled

# %%
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 6))

for i, j in enumerate(["Price", "Total flight duration", "Date", "Dep_hours"]):
    index = 0
    sns.histplot(df, x=j, ax=axes[index][i], kde=True)
    axes[index][i].set_xlabel("")
    axes[index][i].set_ylabel("")
    axes[index][i].set_title(j)
    
    index += 1
    
    sns.boxplot(df, x=j, ax=axes[index][i])
    axes[index][i].set_xlabel("")
    axes[index][i].set_ylabel("")
    axes[index][i].set_title("")
    
plt.tight_layout()
plt.show()

# %% [markdown]
# # Top 10 airlines with highest and lowest average flight prices

# %%
fig, axes = plt.subplots(ncols=2, figsize=(15, 6))

grouped = df.groupby("Airline")

mean = grouped["Price"].mean().sort_values(ascending=False)

mean[:10].plot(kind="bar", ax=axes[0])

for container in axes[0].containers:
    axes[0].bar_label(container, size=8)
    
axes[0].set_yticklabels(())
axes[0].set_ylabel("")
axes[0].set_xlabel("")

mean.sort_values(ascending=True)[:10].plot(kind="bar", ax=axes[1])

for container in axes[1].containers:
    axes[1].bar_label(container, size=8)
    
axes[1].set_yticklabels(())
axes[1].set_ylabel("")
axes[1].set_xlabel("")

plt.show()

# %% [markdown]
# # Routes rated from most expensive to the cheapest based on their average prices and average amount of stop required for these routes

# %%
fig, axes = plt.subplots(ncols=2, figsize=(15, 6))

index = 0

grouped = df.groupby("routes")

mean = grouped["Price"].mean().sort_values(ascending=False)

mean.plot(kind="bar", ax=axes[index])

for container in axes[index].containers:
    axes[index].bar_label(container, size=8)
    
axes[index].set_yticklabels(())
axes[index].set_ylabel("")
axes[index].set_xlabel("")

index += 1

mean = grouped["Total_Stops"].mean().sort_values(ascending=False)

mean.plot(kind="bar", ax=axes[index])

for container in axes[index].containers:
    axes[index].bar_label(container, size=8)
    
axes[index].set_yticklabels(())
axes[index].set_ylabel("")
axes[index].set_xlabel("")

plt.show()

# %% [markdown]
# # Average prices and tendencies for price ranges for different months

# %%
months = ["January", "February", "March",
         "April",  "May",  "June",
         "July", "August", "September",
         "October", "November", "December"]

grouped = df.groupby("Month")

fig, axes = plt.subplots(ncols=2, figsize=(15, 6))

index = 0

mean = grouped["Price"].mean()

mean.plot(kind="bar", ax=axes[index])
for container in axes[index].containers:
    axes[index].bar_label(container)

labels = [months[i-1] for i in mean.index]
axes[index].set_xticklabels(labels)
axes[index].set_yticklabels(())
axes[index].set_ylabel("")
axes[index].set_xlabel("")

index += 1

temp_df = df.copy()

temp_df["Month"] = temp_df["Month"].apply(lambda x: months[x-1])

sns.kdeplot(temp_df, x="Price", hue="Month", ax=axes[index])

axes[index].set_ylabel("")
axes[index].set_xlabel("")

plt.tight_layout()
plt.show()

# %%
df.head()

# %% [markdown]
# # Encoding categorical values

# %%
le = LabelEncoder()

for i in ["Airline", "Source", "Destination"]:
    df[i] = le.fit_transform(df[i].values)

# %% [markdown]
# # Selecting training features and scaling the whole dataset for training

# %%
x = df.drop(["Year", "routes", "Total flight duration", "Price"], axis=1).values
y = df.loc[:, "Price"].values.reshape(-1, 1)

data = np.hstack((x, y))

scaler = MinMaxScaler()

data = scaler.fit_transform(data)

x = data[:, :-1]
y = data[:, -1]

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

# %%
def training(model):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    r2 = r2_score(pred, y_test)
    mse = mean_squared_error(pred, y_test)
    
    return r2*100, mse

# %% [markdown]
# # Regression models defined

# %%
rfr = RandomForestRegressor()
etr = ExtraTreesRegressor()
gbr = GradientBoostingRegressor()
abr = AdaBoostRegressor()
lnr = LinearRegression()
svr = SVR()
xgb = XGBRegressor()
lgb = LGBMRegressor()

models = [rfr, etr, gbr, abr,
         lnr, svr, xgb, lgb]

names = ["Random Forest", "Extra Trees", "Gradient Boosting", "Ada Boost",
        "Linear Regression", "Support Vector Machine", "XGBoost", "LightGBM"]

# %% [markdown]
# # Training and saving evaluation metrics results of regression models defined earlier

# %%
r2s, mses = [], []

for i, j in zip(models, names):
    r2, mse = training(i)
    r2s += [r2]
    mses += [mse]

# %% [markdown]
# # Regression models rated based on their r2 and mean squared error scores

# %%
dd = pd.DataFrame({"scores": r2s, "mse": mses}, index=names)
dd = dd.sort_values("scores", ascending=False)
dd["scores"] = round(dd["scores"], 2)

fig, axes = plt.subplots(ncols=2, figsize=(15, 6))


index = 0

dd["scores"].plot(kind="bar", ax=axes[index])
for container in axes[index].containers:
    axes[index].bar_label(container)
axes[index].set_yticklabels(())
axes[index].set_ylabel("")
axes[index].set_xlabel("")

index += 1
dd = dd.sort_values("mse", ascending=True)
dd["mse"].plot(kind="bar", ax=axes[index])
for container in axes[index].containers:
    axes[index].bar_label(container)
axes[index].set_yticklabels(())
axes[index].set_ylabel("")
axes[index].set_xlabel("")

plt.tight_layout()
plt.show()


