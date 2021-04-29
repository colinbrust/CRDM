library(ggplot2)
library(magrittr)
source('./crdm/R/PlotTheme.R')

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) 

clean_maps <- function(f, states) {
  
  f %>% 
    raster::stack() %>% 
    raster::mask(states)
}

# Have to include na.rm for compatibility with stackApply
mode_calc <- function(x, na.rm) {
    uniqv <- unique(x)
    uniqv[which.max(tabulate(match(x, uniqv)))]
}

average_ensemble <- function(day='20070102', holdout='None', f_list) {
  
  cleaned <- f_list %>%
    grep(day, ., value=TRUE) %>%
    grep(holdout, ., value=TRUE) %>%
    lapply(clean_maps, states=states) %>%
    raster::stack()
  
  len <- length(cleaned)
  
  raster::stackApply(cleaned, rep(1:12, len), fun=mode_calc, na.rm=T)
}

map_to_tidy <- function(stack, day) {
  
  stack %>% 
    raster::rasterToPoints() %>%
    tibble::as_tibble() %>%
    `names<-`(c('x', 'y', paste('lt', 1:12, sep='_'))) %>%
    tidyr::pivot_longer(
      dplyr::starts_with('lt'),
      names_to = 'lead_time',
      values_to = 'val'
    ) %>% 
    dplyr::mutate(
      day = lubridate::as_date(day),
      val = round(val),
      val = dplyr::recode(
        val,
        `0` = 'No Drought',
        `1` = 'D0',
        `2` = 'D1',
        `3` = 'D2',
        `4` = 'D3',
        `5` = 'D4'),
      lead_time = stringr::str_replace(lead_time, 'lt_', ''),
      lead_time = as.numeric(lead_time) - 1) 
}

label_model <- function(data) {
  
  data %>% 
    dplyr::mutate(
      label = paste0(day + lubridate::weeks(lead_time), ' Forecast ', '(', lead_time+1, ')'),
      label = factor(label, levels = stringr::str_sort(unique(label), numeric=TRUE))
    )
} 

label_targets <- function(data) {

    data %>% 
      dplyr::mutate(
        label = paste0(day + lubridate::weeks(lead_time), ' USDM Drought'),
        label = factor(label, levels = stringr::str_sort(unique(label), numeric=TRUE))
      )
}

get_targets <- function(f_dir, day, states) {
  
  day <- lubridate::as_date(day)
  
  dates <- seq(day, day + lubridate::weeks(11), by = 'weeks') %>% 
    stringr::str_replace_all('-', '') %>%
    paste(collapse = '|')
  
  f_dir %>%
    list.files(full.names = T, pattern=dates) %>%
    clean_maps(states = states)
}


plot_data <- function(data, states) {
  
  ggplot() + 
    geom_raster(data = data, mapping = aes(x=x, y=y, fill=val)) + 
    geom_sf(data = states, mapping = aes(), fill=NA, size = 0.5) +
    # coord_sf(crs = 6933, datum = NA) +
    facet_wrap(~label) +
    plot_theme() + 
    scale_fill_manual(values = c('No Drought' = NA,
                                 'D0' = '#FFFF00',
                                 'D1' = '#FCD37F',
                                 'D2' = '#FFAA00',
                                 'D3' = '#E60000',
                                 'D4' = '#730000')) +
    labs(x='', y='', fill='Drought\nCategory') + 
    scale_x_discrete(guide = guide_axis(check.overlap = TRUE)) +
    scale_y_discrete(guide = guide_axis(check.overlap = TRUE))
}

f_list <- list.files('./data/models', full.names=T, recursive = T, pattern='.tif') 

preds <- average_ensemble('20070116', 'None', f_list)
preds_tidy <- map_to_tidy(preds, '20070116')
preds_tidy <- label_model(preds_tidy)

targets <- get_targets('./data/tif_targets', '20070116', states)
targets_tidy <- map_to_tidy(targets, '20070116')
targets_tidy <- label_targets(targets_tidy)

plot_data(preds_tidy, states)
plot_data(targets_tidy, states)


test <- targets %>% dplyr::filter(val != 'No Drought')
