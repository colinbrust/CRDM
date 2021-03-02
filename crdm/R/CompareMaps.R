library(ggplot2)
library(magrittr)
source('https://raw.githubusercontent.com/colinbrust/DroughtCast/revert/crdm/R/PlotTheme.R')

pred_dir = '~/projects/DroughtCast/data/model_results'
true_dir = '~/projects/DroughtCast/data/out_classes/out_tif'


make_info_df <- function(x, lt) {
    x %>%
      list.files(full.names = T) %>% 
      tibble::tibble(f_name = .) %>%
      dplyr::rowwise() %>%
      dplyr::mutate(date = basename(f_name) %>%
                      stringr::str_split('_')%>%
                      unlist() %>%
                      magrittr::extract(1) %>%
                      lubridate::as_date(),
                    lead_time = lt)
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
    facet_wrap(~lead_time, nrow = 2) + 
    labs(title=caption) + 
    plot_theme() 
}

df <- purrr::map2(
       .x = c('./data/model_results/out_maps2', './data/model_results/out_maps4', 
             './data/model_results/out_maps6', './data/model_results/out_maps8',
             './data/out_classes/out_tif'),
       .y = c(2, 4, 6, 8, -1),
       .f = make_info_df
       ) %>% 
  dplyr::bind_rows() %>% 
  split(.$date) %>% 
  Filter(function(y) NROW(y) >= 4, .) %>%
  lapply(function(x) {
    
    day <- unique(x$date)
    
    x %$%
      purrr::map2(f_name, lead_time, raster_to_tibble) %>% 
      dplyr::bind_rows() %>% 
      plot_single_day(day=day, states=states) 
  })
  
states <- urbnmapr::get_urbn_map(sf = TRUE) %>% 
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933)






map_to_tidy <- function(f, template) {
  
  type <- f %>%
    basename() %>%
    stringr::str_replace('.csv', '') %>%
    stringr::str_split('_') %>%
    unlist() %>% 
    tail(1)

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
                  val = ifelse(val == 'NA', NA, val),
                  type = type) 
  
}

plot_single_date <- function(pattern, f_list, out_dir, template, states) {
  
  print(pattern)
  out_name <- file.path(out_dir, paste0(pattern, '_map.png'))
  
  parts <- out_name %>% 
    basename() %>%
    stringr::str_split('_') %>% 
    unlist()
  
  day <- parts[1]
  lead <- parts[7] %>% stringr::str_replace('leadTime-', '')
  
  caption <- paste('Prediction for', lubridate::as_date(day), 'wtih', lead, 'Week Lead Time')
  
  dat <- f_list %>%
    grep(pattern, ., value = T) %>%
    lapply(map_to_tidy, template = template) %>%
    dplyr::bind_rows() %>%
    dplyr::mutate(type = ifelse(type == 'pred', 
                                'Modeled Drought', 
                                'USDM Drought')) 
  
  fig <- dat %>%
    ggplot() + 
    geom_raster(data = dat, mapping = aes(x=x, y=y, fill = val)) +
    geom_sf(data = states, mapping = aes(), fill = NA, size=0.5) +
    # theme(aspect.ratio = 264/610) + 
    labs(x='', y='', fill='Drought\nCategory', title = pattern) +
    scale_fill_manual(values = c('No Drought' = NA,
                                 'D0' = '#FFFF00',
                                 'D1' = '#FCD37F',
                                 'D2' = '#FFAA00',
                                 'D3' = '#E60000',
                                 'D4' = '#730000')) + 
    facet_wrap(~type, nrow=2) + 
    labs(title=caption) + 
    plot_theme() 
    # theme(axis.title.x=element_blank(),
    #       axis.text.x=element_blank(),
    #       axis.ticks.x=element_blank(),
    #       axis.title.y=element_blank(),
    #       axis.text.y=element_blank(),
    #       axis.ticks.y=element_blank()) 

  ggsave(out_name, fig, width = 220, height = 195, units = 'mm',
         dpi = 300)
}

save_all <- function(f_dir='/mnt/e/PycharmProjects/CRDM/data/model_results/weekly_maps',
                     out_dir='/mnt/e/PycharmProjects/CRDM/figures/maps/init', 
                     template='/mnt/e/PycharmProjects/CRDM/data/template.tif') {
  
  f_list <- list.files(f_dir, full.names = T, pattern = '.csv') 
  
  template <- raster::raster(template)
  
  states <- urbnmapr::get_urbn_map(sf = TRUE) %>% 
    dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
    sf::st_transform(6933)
  
  patterns <- f_list %>%
    basename() %>%
    stringr::str_replace('_pred.csv', '') %>%
    stringr::str_replace('_real.csv', '') %>%
    unique()
  
  parallel::mclapply(patterns, plot_single_date, f_list = f_list, out_dir = out_dir, 
         template = template, states = states, mc.cores = 10)
}



old_names = list.files('./data/in_features/weekly_mem', full.names = T, pattern = 'VOD') 
new_names = stringr::str_replace(old_names, '_VOD.dat', '_vod.dat')
