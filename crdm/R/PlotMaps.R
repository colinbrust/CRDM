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

get_both <- function(f, target_dir, states, plot_target = TRUE) {
  
  day <- f %>% 
    basename() %>% 
    stringr::str_split('_') %>% 
    unlist() %>% 
    head(1)
  
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
    dplyr::filter(lead_time %in% c(1, 3, 7, 11),
                  val != 'No Drought') %>%
    plot_data(states = states)
}

plot_data <- function(data, states) {
  
  ggplot() + 
    geom_raster(data = d2, mapping = aes(x=x, y=y, fill=val)) + 
    geom_sf(data = states, mapping = aes(), fill=NA, size = 0.5) +
    # coord_sf(crs = 6933, datum = NA) +
    facet_grid(rows = dplyr::vars(src), cols = dplyr::vars(label), switch = 'y') + 
    # facet_wrap(src~label, nrow=2, switch = 'y') +
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
