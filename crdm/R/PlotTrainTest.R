library(reticulate)
library(magrittr)
library(ggplot2)
use_condaenv("ml", conda = "/home/colin/miniconda3/bin/conda")
source_python('./crdm/utils/ReadPickle.py')
source('./crdm/R/PlotTheme.R')

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) 

f <-  '/mnt/e/PycharmProjects/DroughtCast/data/models/final_model/locs.p'
tmp <-  './data/tif_targets/20000104_USDM.tif' %>%
  raster::raster() %>%
  raster::rasterToPoints() %>%
  tibble::as_tibble() %>%
  dplyr::select(x, y) %>%
  tibble::rowid_to_column(var = 'idx')

dat <- read_pickle(f) 

base <- tibble::tibble(idx = 1:176648)
train <- tibble::tibble(train=unlist(dat$train)) %>%
  dplyr::mutate(idx=train)
test <- tibble::tibble(test=unlist(dat$test)) %>%
  dplyr::mutate(idx=test)

df <- dplyr::left_join(base, train) %>%
  dplyr::left_join(test) %>%
  dplyr::mutate(train = ifelse(is.na(train), 0, 1),
                test = ifelse(is.na(test), 0, 1)) %>% 
  dplyr::left_join(tmp) %>%
  tidyr::pivot_longer(cols=c(train, test), names_to='name') %>%
  dplyr::mutate(value = ifelse(value == 0, 'Training Set', 'Test Set'))

ggplot() + 
  geom_raster(aes(x=x, y=y, fill=value), data=df) +
  geom_sf(mapping = aes(), data=states, fill=NA) + 
  plot_theme() + 
  labs(x='', y='', fill='')
  
