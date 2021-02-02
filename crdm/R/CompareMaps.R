library(ggplot2)
library(magrittr)
source('https://raw.githubusercontent.com/colinbrust/CRDM/develop/crdm/R/PlotTheme.R')

strip_text = function(x) {
  
  x %>%
    stringr::str_split('-') %>%
    lapply(magrittr::extract, -1) %>%
    as.character()
}

parse_fname <- function(df) {
  
  df %>%
    dplyr::mutate(base = basename(f)) %>%
    tidyr::separate(base, c('date', 'drop', 'epochs', 'batch', 'nMonths', 'hiddenSize', 'leadTime', 'remove', 'init', 'numLayers', 'drop2', 'type'), sep='_') %>%
    dplyr::select(-dplyr::starts_with('drop')) %>%
    dplyr::mutate_at(c('epochs', 'batch', 'nMonths', 'hiddenSize', 'leadTime', 'remove', 'init', 'numLayers'), strip_text) %>%
    dplyr::mutate(date = lubridate::as_date(date), 
                  type = stringr::str_replace(type, '.csv', ''))
}

map_to_tidy <- function(f, template) {
  
  f %>%
    read.csv(header = FALSE) %>% 
    as.matrix() %>%
    raster::raster() %>% 
    raster::`extent<-`(raster::extent(template)) %>% 
    raster::`crs<-`(value = raster::crs(template))  %>%
    raster::rasterToPoints() %>%
    tibble::as_tibble() %>%
    dplyr::rename(val = layer) %>% 
    dplyr::mutate(val = ifelse(val < 0.01, 0, val),
                  val = as.character(val),
                  val = dplyr::recode(
                    val, 
                    `0` = 'No Drought',
                    `1` = 'D0', 
                    `2` = 'D1',
                    `3` = 'D2',
                    `4` = 'D3',
                    `5` = 'D4'),
                  f = f) %>%
    parse_fname()
  
}

plot_fun <- function(df, template, states) {
  
  df = unname(df)[[1]]

  f_maps <- df$f %>%
    lapply(map_to_tidy, template=template)
  
  f_maps %>% 
    dplyr::bind_rows() %>%
    dplyr::mutate(leadTime = ifelse(type == 'real', 'USDM Drought', paste(leadTime, 'Week Forecast'))) %>%
    ggplot() + 
      geom_raster(aes(x=x, y=y, fill=val)) + 
      geom_sf(aes(), states, fill = NA) + 
      labs(x='', y='', fill='Drought\nCategory', paste('Drought Forecast for', unique(df$date))) +
      scale_fill_manual(values = c('No Drought' = NA,
                                   'D0' = '#FFFF00',
                                   'D1' = '#FCD37F',
                                   'D2' = '#FFAA00',
                                   'D3' = '#E60000',
                                   'D4' = '#730000')) + 
     facet_wrap(~leadTime, nrow=3) + 
     plot_theme() 
    
}

template = raster::raster('~/projects/CRDM/data/drought/template.tif')

states <- urbnmapr::get_urbn_map(sf = TRUE) %>% 
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933)

list.files('~/projects/CRDM/data/drought/model_results/weekly_maps', full.names = T) %>% 
  tibble::tibble(f = .) %>%
  dplyr::filter(basename(f) %>% stringr::str_sub(1, 4) %in% c('2015', '2017')) %>%
  parse_fname() %>%
  dplyr::group_by(date) %>% 
  dplyr::arrange(date, type) %>%
  dplyr::filter(dplyr::row_number() <= dplyr::n()/2 + 1) %>%
  split(.$date) %>%
  lapply(plot_fun)
  


    
