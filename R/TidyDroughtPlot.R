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