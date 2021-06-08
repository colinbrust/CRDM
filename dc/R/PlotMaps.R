library(ggplot2)
library(magrittr)
library(patchwork)
source('./crdm/R/PlotTheme.R')

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) 

strip_date <- function(f) {
  f %>% 
    basename() %>% 
    stringr::str_split('_') %>% 
    unlist() %>% 
    head(1)
}

clean_maps <- function(f, states) {
  
  f %>% 
    raster::stack() %>% 
    raster::subset(subset = c(2, 4, 8, 12)) %>%
    raster::mask(states)
}

map_to_tidy <- function(stack, day) {
  
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
      val = dplyr::case_when(
          val <= 3 ~ round(val),
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

label_model <- function(data, txt=' Forecast ') {
  
  lab <- txt
  
  data %>% 
    dplyr::mutate(
      label = paste0(day + lubridate::weeks(lead_time), lab, '(', lead_time+1, ')'),
      label = factor(label, levels = stringr::str_sort(unique(label), numeric=TRUE))
    )
} 

label_targets <- function(data) {

    data %>% 
      dplyr::mutate(
        label = paste0(day + lubridate::weeks(lead_time), ' USDM'),
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
  
  colorRampPalette(c('#ffffff','#FFFF00','#FCD37F','#FFAA00','#E60000','#730000')) -> pal
  
  ggplot() + 
    geom_raster(data = data, mapping = aes(x=x, y=y, fill=val)) + 
    geom_sf(data = states, mapping = aes(), fill=NA, size = 0.5) +
    facet_grid(rows = dplyr::vars(src), cols = dplyr::vars(label), switch = 'y') + 
    plot_theme() + 
    # scale_fill_gradientn(na.value='grey26', colors = pal(100), limits = c(0, 5)) + 
    scale_fill_manual(values = c('No Drought' = NA,
                                 'D0' = '#FFFF00',
                                 'D1' = '#FCD37F',
                                 'D2' = '#FFAA00',
                                 'D3' = '#E60000',
                                 'D4' = '#730000')) +
    labs(x='', y='', fill='Drought\nCategory') + 
    scale_y_discrete(guide = guide_axis(check.overlap = TRUE)) + 
    theme(axis.text.x = element_text(angle = 45),
          strip.placement = "outside",
          plot.margin= grid::unit(c(0, 0, 0, 0), "in"))
}

get_sd <- function(f, day, states) {
  
  f %>% 
    clean_maps(states = states) %>%
    raster::mask(states) %>% 
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
      lead_time = lead_time - 1,
      day = lubridate::as_date(day),
    )
}

plot_sd <- function(sd, states) {
  
  ggplot() +
    geom_raster(aes(x=x, y=y, fill=val), sd) + 
    geom_sf(aes(), states, fill = NA) + 
    facet_grid(rows = dplyr::vars(src), cols = dplyr::vars(label), switch = 'y') + 
    plot_theme() + 
    labs(x='', y='', fill='Std. Dev.\nDrought\nCategory') + 
    scale_y_discrete(guide = guide_axis(check.overlap = TRUE)) + 
    theme(axis.text.x = element_text(angle = 45),
          strip.placement = "outside",
          plot.margin= grid::unit(c(0, 0, 0, 0), "in")) + 
    scale_fill_viridis_c()
}

plot_diff <- function(difference, states) {
  
  pal <- colorRampPalette(RColorBrewer::brewer.pal(10, 'RdBu'))
  
  ggplot() +
    geom_raster(aes(x=x, y=y, fill=val), difference) + 
    geom_sf(aes(), states, fill = NA) + 
    facet_grid(rows = dplyr::vars(src), cols = dplyr::vars(label), switch = 'y') + 
    plot_theme() + 
    labs(x='', y='', fill='Difference') + 
    scale_fill_gradientn(na.value='grey26', colors = pal(100),limits = c(-3, 3)) + 
    theme(axis.text.x = element_text(angle = 45),
          strip.placement = "outside",
          plot.margin= grid::unit(c(0, 0, 0, 0), "in")) 
}

plot_all <- function(f, target_dir='./data/tif_targets', states) {

  day <- strip_date(f)
    
  model <- clean_maps(f, states = states) 
  
  targets <- get_targets(target_dir, day, states) 
  
  # sd <- file.path(pred_dir, 'sd', base) %>%
  #   get_sd(day, states) %>%
  #   label_model(txt = ' Std. Dev. ') %>%
  #   dplyr::mutate(src = 'Std. Dev.')
  
  p1 <- targets %>%
    map_to_tidy(day = day) %>%
    label_targets() %>%
    dplyr::mutate(src = 'Target') %>%
    dplyr::filter(val != 'No Drought') %>%
    plot_data(states = states) 
    
  
  p2 <- model %>%
    map_to_tidy(day = day) %>%
    label_model(txt = ' Forecast ') %>%
    dplyr::mutate(src = 'Model') %>%
    dplyr::filter(val != 'No Drought') %>%
    plot_data(states = states)
  
  p1/p2
  
  # p3 <- plot_sd(sd, states) 
  # p4 <- plot_diff(difference, states)
  # 
  # (p1)/(p2)/(p3)/(p4)
  
}

plot_flash <- function(f, target_dir='./data/tif_targets', states) {
  
  flash_states <- states %>%
    dplyr::filter(state_abbv %in% c('MT', 'WY', 'SD', 'ND'))
  
  day <- strip_date(f)
  
  model <- clean_maps(f, states = flash_states) 
  
  max <- file.path(dirname(dirname(f)), 'max') %>%
    list.files(pattern = day, full.names = T) %>%
    clean_maps(states = flash_states)
  
  targets <- get_targets(target_dir, day, flash_states) 
  
  sd <- file.path(dirname(dirname(f)), 'sd') %>%
    list.files(pattern = day, full.names = T) %>%
    get_sd(day, flash_states) %>%
    label_model(txt = ' Std. Dev. ') %>%
    dplyr::mutate(src = 'Std. Dev.')
  
  p1 <- targets %>%
    map_to_tidy(day = day) %>%
    label_targets() %>%
    dplyr::mutate(src = 'Target') %>%
    dplyr::filter(val != 'No Drought') %>%
    plot_data(states = flash_states) 
  
  
  p2 <- model %>%
    map_to_tidy(day = day) %>%
    label_model(txt = ' Forecast ') %>%
    dplyr::mutate(src = 'Model') %>%
    dplyr::filter(val != 'No Drought') %>%
    plot_data(states = flash_states)
  
  p3 <- plot_sd(sd, flash_states) 
  
  p4 <- max %>%
    map_to_tidy(day = day) %>%
    label_model(txt = ' Ensemble Max ') %>%
    dplyr::mutate(src = 'Ens. Max') %>%
    dplyr::filter(val != 'No Drought') %>%
    plot_data(states = flash_states)
  
  p1/p2/p4/p3
  
  
}

# lead time on x axis