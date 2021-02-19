library(ggplot2)
library(magrittr)
source('https://raw.githubusercontent.com/colinbrust/CRDM/develop/crdm/R/PlotTheme.R')



map_to_tidy_cat <- function(f, band) {
  
  f %>%
    raster::raster(band=band) %>% 
    raster::rasterToPoints() %>%
    tibble::as_tibble() %>%
    `colnames<-`(c('x', 'y', 'val')) %>% 
    dplyr::mutate(val = as.character(val),
                  val = dplyr::recode(
                    val, 
                    `0` = 'No Drought',
                    `1` = 'D0', 
                    `2` = 'D1',
                    `3` = 'D2',
                    `4` = 'D3',
                    `5` = 'D4'),
                   lead_time = band*2) 
}

map_to_tidy_con <- function(f, band) {
  
  f %>%
    raster::raster(band=band) %>% 
    raster::rasterToPoints() %>%
    tibble::as_tibble() %>%
    `colnames<-`(c('x', 'y', 'val')) %>% 
    dplyr::mutate(lead_time = band*2) 
}

plot_single_cat <- function(dat, states) {

  ggplot() + 
  geom_raster(aes(x=x, y=y, fill=val), dat) +
  geom_sf(aes(), states, fill = NA) +
    labs(x='', y='', fill='Drought\nCategory') +
    scale_fill_manual(values = c('No Drought' = NA,
                                 'D0' = '#FFFF00',
                                 'D1' = '#FCD37F',
                                 'D2' = '#FFAA00',
                                 'D3' = '#E60000',
                                 'D4' = '#730000')) +
    facet_wrap(~lab, nrow=2) +
    plot_theme()
}

plot_single_con <- function(dat, states) {
  
  ggplot() + 
    geom_raster(aes(x=x, y=y, fill=val), dat) +
    geom_sf(aes(), states, fill = NA) +
    labs(x='', y='', fill='Drought\nCategory') +
    scale_fill_manual(values = c('No Drought' = NA,
                                 'D0' = '#FFFF00',
                                 'D1' = '#FCD37F',
                                 'D2' = '#FFAA00',
                                 'D3' = '#E60000',
                                 'D4' = '#730000')) +
    facet_wrap(~lab, nrow=2) +
    plot_theme()
}


plot_fun <- function(dat, states, out_dir) {
  
  print(unique(dat$fname))
  
  f_maps <- dat %$%
    purrr::map2(fname, band, map_to_tidy_con) %>%
    dplyr::bind_rows() %>%
    dplyr::mutate(lab = paste(lead_time, 'Week Forecast')) %>%
    plot_single(states = states) +
    ggtitle(paste('Forecasts of USDM for', unique(dat$d))) +
    theme(legend.position = "none")
  
  true_map <- dat %$%
    unique(true_fname) %>%
    map_to_tidy(band=1) %>%
    dplyr::mutate(lab = 'True USDM Drought')  %>%
    plot_single(states = states)
  
  fig <- cowplot::plot_grid(f_maps, true_map, nrow=2, rel_heights = c(2, 1))
  out_name  = file.path(
    out_dir,
    paste0(unique(dat$d) %>% stringr::str_replace_all('-', ''), '_preds.png')
  )
  ggsave(out_name, fig, width = 220, height = 195, units = 'mm',
         dpi = 300)
}


save_all <- function(true_dir, map_dir, out_dir) {
  states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
    dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
    sf::st_transform(6933)
  
  true_dir <- true_dir %>%
    list.files(full.names = T) %>%
    tibble::tibble(true_fname = .) %>%
    dplyr::mutate(
      d = basename(true_fname) %>%
        stringr::str_sub(1, 8) %>%
        lubridate::as_date()
    )
  
  dfs <- list.files(map_dir,full.names = T) %>%
    expand.grid(., 1:4) %>%
    tibble::as_tibble() %>%
    `colnames<-`(c('fname', 'band')) %>%
    dplyr::mutate(
      fname = as.character(fname),
      d = basename(fname) %>%
        stringr::str_sub(1, 8) %>%
        lubridate::as_date(),
      d = d + band*2*7
    ) %>%
    dplyr::left_join(true_dir, by='d') %>%
    split(.$d) %>%
    Filter(function(x) {NROW(x) == 4}, .) 
  
  
    lapply(dfs, plot_fun, states = states, out_dir = out_dir)
}

true_dir = '/mnt/e/PycharmProjects/CRDM/data/out_classes/out_tif'
map_dir = '/mnt/e/PycharmProjects/CRDM/data/model_results/weekly_maps/mse_stateful_ensemble/model0'
out_dir = '/mnt/e/PycharmProjects/CRDM/data/model_results/weekly_maps/mse_stateful_ensemble/pred_maps'
 
# save_all(
#   '/mnt/e/PycharmProjects/CRDM/data/out_classes/out_tif',
#   '/mnt/e/PycharmProjects/CRDM/data/model_results/weekly_maps/mse_stateful_ensemble/model0',
#   '/mnt/e/PycharmProjects/CRDM/data/model_results/weekly_maps/mse_stateful_ensemble/pred_maps'
# )

