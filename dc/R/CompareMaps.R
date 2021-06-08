library(ggplot2)
library(magrittr)
source('https://raw.githubusercontent.com/colinbrust/DroughtCast/revert/crdm/R/PlotTheme.R')

make_metadata_df <- function(x) {
  
    x %>%
      list.files(full.names = T, recursive = T, pattern = '.tif') %>% 
      tibble::tibble(f_path = .) %>%
      dplyr::mutate(f = basename(f_path)) %>%
      tidyr::separate(f, c('date', 'variable', 'lead_time'), '_') %>% 
      dplyr::mutate(date = lubridate::as_date(date),
                    variable = ifelse(variable == 'USDM.tif', 'true', variable),
                    lead_time = ifelse(is.na(lead_time), -1,
                                       stringr::str_replace(lead_time, '.tif', '') %>%
                      as.numeric()))
  
}

drought_palette <- c('#FFFF00', '#FCD37F', '#FFAA00', '#E60000', '#730000')

raster_to_tibble <- function(f, lead_time) {
  
  if (lead_time != -1) {
    f %>%
      raster::raster() %>% 
      raster::calc(function(x) {
        x = x * 5
        x[x < 0.3] = NA
        return(x)
      }) %>% 
      raster::rasterToPoints() %>%
      tibble::as_tibble() %>%
      `colnames<-`(c('x', 'y', 'val')) %>%
      dplyr::mutate(lead_time = lead_time) %>%
      return()
  } else {
    f %>%
      raster::raster() %>% 
      raster::rasterToPoints() %>%
      raster::calc(function(x) {
        x = x - 1
        x[x == -1] = NA
        return(x)
      }) %>% 
      raster::rasterToPoints() %>%
      tibble::as_tibble() %>%
      `colnames<-`(c('x', 'y', 'val')) %>%
      dplyr::mutate(lead_time = lead_time) %>%
      return()
  }
}

plot_single_day <- function(dat, day,  states, out_dir) {
  
  caption <- paste('Prediction for', day, 'USDM Drought')
  print(caption)
  out_dat <- dat %>%
    dplyr::mutate(
      lead_time = as.character(lead_time), 
      lead_time = ifelse(lead_time == '-1', 
                         'USDM Drought',
                         paste(lead_time, 'Week Lead Time')
                         )
      )
  
  fig <- ggplot() + 
    geom_raster(data = out_dat, mapping = aes(x=x, y=y, fill = val)) +
    geom_sf(data = states, mapping = aes(), fill = NA, size=0.5) +
    scale_fill_gradientn(colors=drought_palette, limits=c(0, 5)) +
    labs(x='', y='', fill='Drought\nCategory') +
    facet_wrap(~lead_time, nrow = 3) + 
    labs(title=caption) + 
    plot_theme() 
  
  
  out_name = file.path(out_dir, paste0(day, '.png'))
  ggsave(out_name, fig, width = 220, height = 195, units = 'mm',
         dpi = 300)
  
}


save_all_maps <- function(pred_dir, true_dir, out_dir, err) {
  
  states <- urbnmapr::get_urbn_map(sf = TRUE) %>% 
    dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
    sf::st_transform(6933)
  
  df <- lapply(c(pred_dir, true_dir), make_metadata_df) %>%
    dplyr::bind_rows() %>%
    dplyr::filter(if (!err) variable == 'full' | variable == 'true')
  
  if (err) {
    # do some stuff
  } else {
    df %>% 
      split(.$date) %>%
      Filter(function(x) NROW(x)  == 5, .) %>%
      tail(1) %>% 
      lapply(function(x) {
        
        day <- unique(x$date)
        
        x %$%
          purrr::map2(f_path, lead_time, raster_to_tibble) %>% 
          dplyr::bind_rows() %>% 
          plot_single_day(day=day, states=states, out_dir=out_dir) 
      }) 
  }
}

