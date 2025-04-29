# Score Driven Models (SDM) in Python

## Overview

This repository contains a Python implementation of **Score Driven Models (SDMs)**, a class of dynamic models that use score functions to model time-varying parameters, such as volatility or trend. SDMs are widely used in time series forecasting, particularly in financial and economic data analysis. The implementation includes key functions for:

- **Training**: Fit the model to historical time series data.
- **Filtering**: Estimate the evolving state of the model.
- **Prediction**: Forecast future values based on the trained model.

Additionally, a test notebook demonstrates a standard **walk-forward validation** implementation, showcasing how to apply the model in a practical forecasting setting.

## Features

- **Training**: Fit the Score Driven Model to the data using maximum likelihood estimation.
- **Filtering**: Dynamically update the model’s parameters as new data becomes available.
- **Prediction**: Generate predictions for future time steps.
- **Walk-Forward Validation**: A rolling forecast implementation that re-trains the model at each step, useful for evaluating performance on out-of-sample data.

## Main Theory Reference
The theory behind the Score Driven Model is based on the following paper by Koopman et al.:

Score-Driven Time Series Models: https://sjkoopman.net/papers/OUPch2_final_upload.pdf

This paper provides a comprehensive overview of the underlying statistical framework and methodology used to construct SDMs.
