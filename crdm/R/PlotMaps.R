library(ggplot2)
library(magrittr)
source('https://raw.githubusercontent.com/colinbrust/CRDM/develop/crdm/R/PlotTheme.R')

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

tibble::as_tibble(a) %>% 
  `colnames<-`(c('x', 'y', paste('lt', 1:12, sep ='_'))) %>% 
  tidyr::pivot_longer(dplyr::starts_with('lt_')) %>% 
  dplyr::mutate(value = round(value),
                value = ifelse(value >5, 5, value),
                value = dplyr::recode(
                  value, 
                  `0` = 'No Drought',
                  `1` = 'D0', 
                  `2` = 'D1',
                  `3` = 'D2',
                  `4` = 'D3',
                  `5` = 'D4')) %>% 
  ggplot(aes(x=x, y=y, fill=value)) +
   geom_raster() + 
   facet_wrap(~name) + 
   scale_fill_manual(values = c('No Drought' = NA,
                               'D0' = '#FFFF00',
                               'D1' = '#FCD37F',
                               'D2' = '#FFAA00',
                               'D3' = '#E60000',
                               'D4' = '#730000')) + 
  plot_theme()
