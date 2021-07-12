library(reticulate)
library(magrittr)
library(ggplot2)
use_condaenv("gee", conda = "/opt/miniconda3/bin/conda")
source_python('./dc/utils/ReadPickle.py')
source('./R/PlotTheme.R')

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) 

f <-  './data/models/locs.p'

tmp <-  './data/out_classes/tif/20000104_USDM.tif' %>%
  raster::raster() %>%
  raster::mask(states) 

tmp[is.na(tmp)] <- -1
tmp[tmp != -1] <- 0
tmp <- tmp %>%
  raster::rasterToPoints() %>%
  tibble::as_tibble() %>%
  `names<-`(c('x', 'y', 'val')) %>%
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
  dplyr::mutate(value = ifelse(value == 0, 'Training Set', 'Test Set')) %>%
  dplyr::filter(val != -1)

ggplot() + 
  geom_raster(aes(x=x, y=y, fill=value), data=df) +
  geom_sf(mapping = aes(), data=states, fill=NA) + 
  plot_theme() + 
  labs(x='', y='', fill='')


