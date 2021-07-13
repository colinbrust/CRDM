# DroughtCast: A Machine Learning Forecast of the United States Drought Monitor
This repository contains a model framework for forecasting the United States Drought Monitor (USDM). The framework uses
a Seq2Seq model with encoder and decoder gated recurrent units. The model uses a combination of modeled meteorology and 
satellite observed soil moisture as input features, and forecasts a timeseries of USDM drought 1 to 12 weeks into the 
future. The repository is currently being adjusted to create operational drought forecasts for the United States, so the 
structure is likely to change in the near future. 

## Dependencies
- Python >= 3.8.3 
- PyTorch >= 1.9.0 (Python)
- R >= 4.0.2
- tidyverse >= 1.3.0 (R)

All other Python dependencies are listed in `requirements.txt`, and all other R dependencies will soon be added as part of
an R package for visualizing model results.