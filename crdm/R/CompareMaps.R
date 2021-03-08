library(ggplot2)
library(magrittr)
source('https://raw.githubusercontent.com/colinbrust/DroughtCast/revert/crdm/R/PlotTheme.R')

make_info_df <- function(x) {
  
    x %>%
      list.files(full.names = T) %>% 
      tibble::tibble(f_path = .) %>%
      dplyr::mutate(f = basename(f_path)) %>%
      tidyr::separate(f, c('date', 'variable', 'lead_time'), '_') %>% 
      dplyr::mutate(date = lubridate::as_date(date),
                    variable = ifelse(variable == 'USDM.tif', 'true', variable),
                    lead_time = ifelse(is.na(lead_time), -1,
                                       stringr::str_replace(lead_time, '.tif', '') %>%
                      as.numeric()))
  
}

raster_to_tibble <- function(f, lead_time) {
  
  f %>%
    raster::raster() %>% 
    raster::rasterToPoints() %>%
    tibble::as_tibble() %>%
    `colnames<-`(c('x', 'y', 'val')) %>%
    dplyr::mutate(lead_time = lead_time)
  
}

plot_single_day <- function(dat, day,  states) {
  
  caption <- paste('Prediction for', day, 'USDM Drought')
  
  out_dat <- dat %>%
    dplyr::mutate(
      val = ifelse(lead_time == -1, val, val * 5),
      val = round(val),
      lead_time = as.character(lead_time), 
      lead_time = ifelse(lead_time == '-1', 
                         'USDM Drought',
                         paste(lead_time, 'Week Lead Time')
                         ), 
      val = dplyr::recode(
        val, 
        `0` = 'No Drought',
        `1` = 'D0', 
        `2` = 'D1',
        `3` = 'D2',
        `4` = 'D3',
        `5` = 'D4'),
      val = ifelse(val == 'NA', NA, val)
      )

  fig <- ggplot() + 
    geom_raster(data = out_dat, mapping = aes(x=x, y=y, fill = val)) +
    geom_sf(data = states, mapping = aes(), fill = NA, size=0.5) +
    # theme(aspect.ratio = 264/610) + 
    labs(x='', y='', fill='Drought\nCategory') +
    scale_fill_manual(values = c('No Drought' = NA,
                                 'D0' = '#FFFF00',
                                 'D1' = '#FCD37F',
                                 'D2' = '#FFAA00',
                                 'D3' = '#E60000',
                                 'D4' = '#730000')) + 
    facet_wrap(~lead_time, nrow = 3) + 
    labs(title=caption) + 
    plot_theme() 
}


save_all_maps <- function(pred_dir, true_dir, err) {
  
  states <- urbnmapr::get_urbn_map(sf = TRUE) %>% 
    dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
    sf::st_transform(6933)
  
  df <- lapply(c(pred_dir, true_dir), make_info_df) %>%
    dplyr::bind_rows() %>%
    dplyr::filter(if (!err) variable == 'full' | variable == 'true')
  
  if (err) {
    # do some stuff
  } else {
    df %>% 
      split(.$date) %>%
      Filter(function(x) NROW(x)  == 5, .) %>%
      lapply(function(x) {
        
        day <- unique(x$date)
        
        x %$%
          purrr::map2(f_path, lead_time, raster_to_tibble) %>% 
          dplyr::bind_rows() %>% 
          plot_single_day(day=day, states=states) 
      }) -> test
  }
}

ggsave(out_name, fig, width = 220, height = 195, units = 'mm',
       dpi = 300)
