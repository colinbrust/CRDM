library(ggplot2)
library(magrittr)
source('./crdm/R/PlotTheme.R')

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) 

target_dir = './data/tif_targets'
f = './data/models/old/ensemble/model6/preds/20030805_preds_None.tif'

get_date_string <- function(f) {
  f %>% 
    basename() %>% 
    stringr::str_split('_') %>% 
    unlist() %>% 
    head(1)
}


get_baseline <- function(f, target_dir) {
  
  f %>% 
    get_date_string() %>%
    lubridate::as_date() %>% 
    {. - 7} %>%
    stringr::str_replace_all('-', '') %>%
    list.files(target_dir, full.names = T, pattern = .) %>%
    raster::raster()

}

get_targets <- function(target_dir, f, states) {
  
  day <- lubridate::as_date(get_date_string(f))
  
  dates <- seq(day, day + lubridate::weeks(11), by = 'weeks') %>% 
    stringr::str_replace_all('-', '') %>%
    paste(collapse = '|')
  
  target_dir %>%
    list.files(full.names = T, pattern=dates) %>%
    raster::stack()
}

