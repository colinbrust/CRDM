tidy_drought_plot <- function(f) {
  
  states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
    dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
    sf::st_transform(6933) 
  
  dat <- f %>%
    raster::raster() %>%
    raster::rasterToPoints() %>%
    `colnames<-`(c('x', 'y', 'val')) %>%
    tibble::as_tibble() %>%
    dplyr::mutate(val = factor(val))
    
  
  ggplot() + 
    geom_raster(data = dat, mapping = aes(x=x, y=y, fill=val)) + 
    geom_sf(data = states, mapping = aes(), fill=NA, size = 0.5) +
    plot_theme() + 
    scale_fill_manual(values = c(`0` = NA,
                                 `1` = '#FFFF00',
                                 `2` = '#FCD37F',
                                 `3` = '#FFAA00',
                                 `4` = '#E60000',
                                 `5` = '#730000')) +
    labs(x='', y='', fill='Drought\nCategory') + 
    scale_y_discrete(guide = guide_axis(check.overlap = TRUE)) 
}


colorRampPalette(c('#ffffff','#FFFF00','#FCD37F','#FFAA00','#E60000','#730000')) -> pal

dat <- tidy_raster('./data/models/ensemble_101/preds_37/20070828_preds_None.tif', T) %>%
  tidyr::pivot_longer(dplyr::starts_with('lt')) %>% 
  dplyr::filter(name %in% c('lt_2', 'lt_4', 'lt_8', 'lt_12')) 

library(ggplot2)

ggplot() + 
  geom_raster(aes(x=x, y=y, fill=value), data = dat2) + 
  geom_sf(aes(), data = states, fill=NA) + 
  scale_fill_gradientn(na.value='grey26', colors = pal(100), limits = c(0, 5)) + 
  facet_wrap(~name) +
  plot_theme() +
  labs(x='', y='')
  
dat %>%
  dplyr::mutate(
    test = dplyr::case_when(
      value <= 2 ~ round(value),
      TRUE ~ ceiling(value)
    )
  ) -> dat2

r