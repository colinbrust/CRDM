library(ggplot2)
library(magrittr)
source('./crdm/R/PlotTheme.R')

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv %in% c('MT', 'ND', 'SD', 'WY'))

strip_date <- function(f) {
  
  f %>%
    basename() %>%
    stringr::str_split('_') %>%
    unlist() %>%
    magrittr::extract(1) %>%
    lubridate::as_date()
}

get_tiffs <- function(indices, f_dir, day) {
  
  lapply(indices, function(x) {
    (day + lubridate::weeks(x)) %>%
      stringr::str_replace_all('-', '')
  }) %>% unlist() %>%
    stringr::str_replace_all('-', '') %>%
    paste(collapse = '|') %>%
    list.files(f_dir, pattern = ., full.names = T)
}

tidy_map <- function(r, name) {
  
  raster::rasterToPoints(r) %>%
    tibble::as_tibble() %>%
    `names<-`(c('x', 'y', name))
  
}

get_flash_bad <- function(f, target_dir='./data/models/pix_norm/model1/preds_48', 
                      n_weeks = 2, lead_time = 2) {
  
  day <- strip_date(f)
  pred_day <- day + lubridate::weeks(1)
  
  title <- paste('Observed and Forecasted USDM Change for', day + lubridate::weeks(2))

  obs <- get_tiffs(c(0, 2), dirname(f), day = day)
  pred <- get_tiffs(c(-2, -1, 0), target_dir, day = pred_day)
  
  change <- (raster::raster(obs[2]) - raster::raster(obs[1])) %>%
    raster::projectRaster(crs = new_crs, method = 'ngb') %>%
    raster::crop(states) %>%
    raster::mask(states) %>%
    tidy_map(name = 'val') %>%
    dplyr::mutate(lt = 'Observed Change (2 Weeks Prior)')
  
  pred <- c(raster::raster(pred[1], band = 4) - raster::raster(obs[1]),
            raster::raster(pred[2], band = 3) - raster::raster(obs[1]),
            raster::raster(pred[3], band = 2) - raster::raster(obs[1]))
  
  pred_change <- pred %>%
    lapply(function(x) {
      raster::projectRaster(x, crs=new_crs, method = 'ngb') %>%
        raster::crop(states) %>%
        raster::mask(states) %>%
        tidy_map(name = 'val')
    }) %>%
    dplyr::bind_rows(.id = 'lt') %>%
    dplyr::mutate(
      lt = dplyr::recode(
        lt,
        '1' = '4 Week Forecast',
        '2' = '3 Week Forecast',
        '3' = '2 Week Forecast'
      ))

  
  dat <- dplyr::bind_rows(change, pred_change) %>%
    dplyr::mutate(
      lt = factor(lt, levels = c('Observed Change (2 Weeks Prior)', '2 Week Forecast', 
                                 '3 Week Forecast', '4 Week Forecast'))
    )
  
  colr <- colorRampPalette(RColorBrewer::brewer.pal(7, 'RdBu'))
  
   ggplot() +
    geom_raster(data = dat, mapping = aes(x=x, y=y, fill=val)) +
    geom_sf(data = states, mapping = aes(), fill= NA) +
    plot_theme() +
    facet_wrap(~lt) +
    scale_fill_gradientn(na.value='grey26', colors = rev(colr(100)), limits = c(-3, 3)) +
    labs(x='', y='', fill='USDM\nCategory\nChange', title = title)
}

get_flash_new <- function(f, target_dir='./data/models/pix_norm/model1/preds_48', 
                          n_weeks = 2, lead_time = 2) {
  
  day <- strip_date(f)

  title <- paste('Observed and Forecasted USDM Change for', day + lubridate::weeks(2))
  
  obs <- get_tiffs(c(0, 1, 3, 7), dirname(f), day = day)
  pred <- get_tiffs(c(-2, -1, 0), target_dir, day = pred_day)
  
  change <- (raster::raster(obs[2]) - raster::raster(obs[1])) %>%
    raster::projectRaster(crs = new_crs, method = 'ngb') %>%
    raster::crop(states) %>%
    raster::mask(states) %>%
    tidy_map(name = 'val') %>%
    dplyr::mutate(lt = 'Observed Change (2 Weeks Prior)')
  
  pred_calc <- c(raster::raster(pred[1], band = 4),
            raster::raster(pred[2], band = 3) ,
            raster::raster(pred[3], band = 2) )
  
  pred_change <- pred_calc %>%
    lapply(function(x) {
      raster::projectRaster(x, crs=new_crs, method = 'ngb') %>%
        raster::crop(states) %>%
        raster::mask(states) %>%
        tidy_map(name = 'val')
    }) %>%
    dplyr::bind_rows(.id = 'lt') %>%
    dplyr::mutate(
      lt = dplyr::recode(
        lt,
        '1' = '4 Week Forecast',
        '2' = '3 Week Forecast',
        '3' = '2 Week Forecast'
      ))
  
  
  dat <- dplyr::bind_rows(change, pred_change) %>%
    dplyr::mutate(
      lt = factor(lt, levels = c('Observed Change (2 Weeks Prior)', '2 Week Forecast', 
                                 '3 Week Forecast', '4 Week Forecast'))
    )
  
  colr <- colorRampPalette(RColorBrewer::brewer.pal(7, 'RdBu'))
  
  ggplot() +
    geom_raster(data = dat, mapping = aes(x=x, y=y, fill=val)) +
    geom_sf(data = states, mapping = aes(), fill= NA) +
    plot_theme() +
    facet_wrap(~lt) +
    scale_fill_gradientn(na.value='grey26', colors = rev(colr(100)), limits = c(-3, 3)) +
    labs(x='', y='', fill='USDM\nCategory\nChange', title = title)
}