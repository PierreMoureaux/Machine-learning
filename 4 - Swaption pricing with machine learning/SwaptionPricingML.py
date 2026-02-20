import pandas as pd
import numpy as np

from plotnine import *
from itertools import product
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
import math


def annuity(t: float, paymentDates: list, rateConvention: float, m: int, n: int, r: float) -> float:
    """Calculate annuity."""

    discountFactors = [math.exp(-r * (Ti - t)) for Ti in paymentDates]
    dcf = [(paymentDates[n] - paymentDates[n - 1]) / rateConvention for n in range(1, n)]
    dcf.insert(0, (paymentDates[0] - m) / rateConvention)
    return sum([x * y for x, y in zip(discountFactors, dcf)])


def black_swaption_price(payRec: int, t: float, swapRate: float, K: float, r: int, T_swaption: int, T_swap: int,
                         SwapPaymentDates: list, SwapRateConvention: float, sigma: float) -> float:
    """Calculate Black swaption price."""

    A = annuity(t, SwapPaymentDates, SwapRateConvention, T_swaption, T_swap, r)

    d1 = (math.log(swapRate / K) + (r + sigma ** 2 / 2) * T_swaption) / (sigma * math.sqrt(T_swaption))
    d2 = d1 - sigma * math.sqrt(T_swaption)

    return payRec * A * (swapRate * norm.cdf(payRec * d1) - K * norm.cdf(payRec * d2))

#Create seed, to reproduce results
random_state = 42
np.random.seed(random_state)

#Sample datas for swaption prices
swapRate = np.linspace(0.01, 0.1, 10)
K = np.linspace(0.01, 0.1, 5)
r = np.linspace(0.01, 0.2, 5)
T_Swaption = np.arange(1, 6)
T_Swap = np.arange(1, 6)
SwapPaymentDates = [np.arange(1, 6)] * 5
sigma = np.linspace(0.01, 0.25, 5)

#Dataframe for swaptions datas and resulting prices
swaption_prices = pd.DataFrame(
    product(swapRate, K, r, T_Swaption, T_Swap, SwapPaymentDates, sigma),
    columns=["swapRate", "K", "r", "T_Swaption", "T_Swap", "SwapPaymentDates", "sigma"]
)
swaption_prices["t"] = 0
swaption_prices["payRec"] = 1
swaption_prices["rateConvention"] = 1

def row_apply(row):
    return black_swaption_price(
        row["payRec"],
        row["t"],
        row["swapRate"],
        row["K"],
        row["r"],
        row["T_Swaption"],
        row["T_Swap"],
        row["SwapPaymentDates"],
        row["rateConvention"],
        row["sigma"],
    )

swaption_prices['black'] = swaption_prices.apply(row_apply, axis=1)

#Artificial observed prices, based on black prices + random white noise
swaption_prices = swaption_prices.assign(
    observed_price=lambda x: (x["black"] + np.random.normal(scale=0.05))
)

swaption_prices.drop("SwapPaymentDates", axis=1, inplace=True)
swaption_prices.drop("t", axis=1, inplace=True)
swaption_prices.drop("payRec", axis=1, inplace=True)
swaption_prices.drop("rateConvention", axis=1, inplace=True)

#Split datas versus training and testing data
train_data, test_data = train_test_split(
    swaption_prices, test_size=0.01, random_state=random_state
)

#Data normalization
preprocessor = ColumnTransformer(
    transformers=[
        (
            "normalize_predictors",
            StandardScaler(),
            ["swapRate", "K", "r", "sigma"],
        )
    ],
    remainder="drop",
)

#Multi-layer perceptron (one hidden layer)
max_iter = 100
nnet_model = MLPRegressor(
    hidden_layer_sizes=10, max_iter=max_iter, random_state=random_state
)
nnet_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", nnet_model)]
)
nnet_fit = nnet_pipeline.fit(
    train_data.drop(columns=["observed_price"]),
    train_data.get("observed_price"),
)

#Random forest
rf_model = RandomForestRegressor(
    n_estimators=50, min_samples_leaf=2000, random_state=random_state
)
rf_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", rf_model)]
)
rf_fit = rf_pipeline.fit(
    train_data.drop(columns=["observed_price"]),
    train_data.get("observed_price"),
)

#Multi-layer perceptron (three hidden layer), so-called deep learning
deepnnet_model = MLPRegressor(
    hidden_layer_sizes=(10, 10, 10),
    activation="logistic",
    solver="lbfgs",
    max_iter=max_iter,
    random_state=random_state,
)
deepnnet_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", deepnnet_model)]
)
deepnnet_fit = deepnnet_pipeline.fit(
    train_data.drop(columns=["observed_price"]),
    train_data.get("observed_price"),
)

#Polynomial regression, with lasso regularization
lm_pipeline = Pipeline(
    [
        (
            "polynomial",
            PolynomialFeatures(
                degree=5, interaction_only=False, include_bias=True
            ),
        ),
        ("scaler", StandardScaler()),
        ("regressor", Lasso(alpha=0.01)),
    ]
)
lm_fit = lm_pipeline.fit(
    train_data.get(["swapRate", "K", "r", "T_Swaption", "T_Swap", "sigma"]),
    train_data.get("observed_price"),
)

#Runs models and results
test_X = test_data.get(["swapRate", "K", "r", "T_Swaption", "T_Swap", "sigma"])
test_y = test_data.get("observed_price")
predictive_performance = (
    pd.concat(
        [
            test_data.reset_index(drop=True),
            pd.DataFrame(
                {
                    "Random forest": rf_fit.predict(test_X),
                    "Single layer": nnet_fit.predict(test_X),
                    "Deep NN": deepnnet_fit.predict(test_X),
                    "Lasso": lm_fit.predict(test_X),
                }
            ),
        ],
        axis=1,
    )
    .melt(
        id_vars=["swapRate", "K", "r", "T_Swaption", "T_Swap", "sigma", "black",
                 "observed_price"],
        var_name="Model",
        value_name="Predicted",
    )
    .assign(
        moneyness=lambda x: x["swapRate"] - x["K"],
        pricing_error=lambda x: np.abs(x["Predicted"] - x["black"]),
    )
)
predictive_performance_figure = (
        ggplot(
            predictive_performance,
            aes(x="moneyness", y="pricing_error")
        )
        + geom_point(alpha=0.05)
        + facet_wrap("Model")
        + labs(
    x="Moneyness (SwapRate - K)", y="Absolut prediction error (USD)",
    title="Prediction errors of swaptions for different models"
)
        + theme(legend_position="")
)
predictive_performance_figure.show()
