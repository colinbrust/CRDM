library(ggplot2)
library(magrittr)
source('./crdm/R/PlotTheme.R')

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) 

# Have to include na.rm for compatibility with stackApply
mode_calc <- function(x, na.rm) {
  uniqv <- unique(x)
  uniqv[which.max(tabulate(match(x, uniqv)))]
}

strip_date <- function(f) {
  f %>% 
    basename() %>% 
    stringr::str_split('_') %>% 
    unlist() %>% 
    head(1)
}

clean_maps <- function(f, states, agg = 1) {
  
  f %>% 
    raster::stack() %>% 
    raster::subset(subset = c(2, 4, 8, 12)) %>%
    raster::mask(states)
}

map_to_tidy <- function(stack, day, agg=1) {
  
  stack %>% 
    raster::rasterToPoints() %>%
    tibble::as_tibble() %>%
    `names<-`(c('x', 'y', 'lt_2', 'lt_4', 'lt_8', 'lt_12')) %>%
    tidyr::pivot_longer(
      dplyr::starts_with('lt'),
      names_to = 'lead_time',
      values_to = 'val'
    ) %>% 
    dplyr::mutate(
      day = lubridate::as_date(day),
      val =  dplyr::case_when(
        val <= 2 ~ round(val),
        TRUE ~ ceiling(val)
      ),
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
'./data/models/global_norm/model4/preds_87/20170627_preds_None.tif'

label_model <- function(data) {
  
  data %>% 
    dplyr::mutate(
      label = paste0(day + lubridate::weeks(lead_time), ' Drought ', '(', lead_time+1, ')'),
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
    facet_grid(rows = dplyr::vars(src), cols = dplyr::vars(label), switch = 'y') + 
    plot_theme() + 
    scale_fill_manual(values = c('No Drought' = NA,
                                 'D0' = '#FFFF00',
                                 'D1' = '#FCD37F',
                                 'D2' = '#FFAA00',
                                 'D3' = '#E60000',
                                 'D4' = '#730000')) +
    labs(x='', y='', fill='Drought\nCategory') + 
    scale_y_discrete(guide = guide_axis(check.overlap = TRUE)) + 
    theme(axis.text.x = element_text(angle = 45),
          strip.placement = "outside")
}


get_both <- function(f, target_dir, states, plot_target = TRUE) {
  
  day <- strip_date(f)
  
  model <- f %>% 
    clean_maps(states = states) %>%
    map_to_tidy(day = day) %>%
    label_model() %>%
    dplyr::mutate(src = 'Model')
  
  if (plot_target) {
    targets <- get_targets(target_dir, day, states) %>% 
      map_to_tidy(day = day) %>%
      label_model() %>%
      dplyr::mutate(src = 'Target')
  } else {
    targets <- tibble::tibble()
  }
  
  dplyr::bind_rows(model, targets) %>%
    dplyr::filter(val != 'No Drought') %>%
    plot_data(states = states)
}

plot_sd <- function(f, states) {
  
  d <- strip_date(f) %>% lubridate::as_date()
  
  dat <- f %>% 
    clean_maps(states = states) %>%
    raster::rasterToPoints() %>%
    tibble::as_tibble() %>%
    `names<-`(c('x', 'y', 'lt_2', 'lt_4', 'lt_8', 'lt_12')) %>%
    tidyr::pivot_longer(
      dplyr::starts_with('lt'),
      names_to = 'lead_time',
      values_to = 'val'
    ) %>%
    dplyr::mutate(
      lead_time = stringr::str_replace(lead_time, 'lt_', '') %>% as.numeric(), 
      lab = paste('Std. Dev. for', d + lubridate::weeks(lead_time-1)),
      lab = paste0(lab, ' (', lead_time, ')')
    )
  
    ggplot() + 
      geom_raster(aes(x=x, y=y, fill=val), data = dat) + 
      geom_sf(aes(), data = states, fill = NA) + 
      facet_wrap(~lab) +
      plot_theme() + 
      labs(x='', y='', fill='Std. Dev.') +
      scale_fill_continuous(type = 'viridis')
}
