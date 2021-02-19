# DroughtCast - A Machine Learning-Based Model for Forecasting the United States Drought Monitor
DroughtCast uses a combination of satellite observations and modeled meteorology to forecast the United States Drought 
Monitor (USDM). Model inputs include observations of rootzone and surface soil moisture from the Soil Moisture Active Passive 
(SMAP) Level-4 soil moisture product, estimates of gross primary production from the SMAP Level-4 carbon product, estimates
of evaportanspiration from a SMAP-constrained MOD16 algorithm, and modeled meteorology from gridMet. The current model
uses a recurrent neural network architecture to process inputs and make a drought prediction. The DroughtCast model
forecasts drought up to 8 weeks in advance and can be used as a risk assessment tool to assess flash drought risk across
the United States. 

## Dependencies
### Python Dependencies
 - Python >= 3.8 
 - `PyTorch`
 - `NumPy`
 - `RasterIO`

### R Dependencies
- `Tidyverse`
- `Fasterize`
- `urbnmapr`
- `raster`

## Installation

At the moment, the package can only be installed through GitHub using the following steps:
- Run `git clone https://github.com/colinbrust/DroughtCast.git` in the terminal.
- `cd` to the DroughtCast directory.
- Run `git checkout develop` as the most up to date model is in the develop branch.
- Create project environment using `conda` or `venv`
- Install dependencies
- Install package using `python setup.py install`

As the project grows and nears completion, a more refined installation method will be developed. 

## Usage 

- The `fetch` module can be used to download model input training data and USDM maps.
- `R/GridmetToTif.R` can be used to convert training data from NetCDF format to GeoTiff format. 
- `utils/ToMemmap.py` can be used to convert model inputs to NumPy memory mapped files, which significantly speeds up read time. 
- `utils/MakeLSTMPixelTS.py` can be used to organize and arrange all training data so it can easily be fed into the DroughtCast model. 
- `classification/TrainLSTMCategorical.py` and `classification/TrainLSTMContinuous.py` can both be used train the model.
    - `TrainLSTMCategorical.py` treats USDM drought categories as classes.
    - `TrainLSTMContinuous.py` treats USDM drought as a continuous variable.
- `utils/ModelToMap.py` creates DroughtCast predictions once the model is trained. 
- The `R` module has various functions for plotting model predictions and model error. 