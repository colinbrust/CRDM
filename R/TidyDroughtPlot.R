library(ggplot2)
library(magrittr)
library(patchwork)
source('./R/PlotTheme.R')


tidy_drought_plot <- function(f) {
  print(f)
  
  states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
    dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
    sf::st_transform(6933) %>%
    dplyr::filter(state_abbv %in% c('TN', 'NC', 'SC', 'GA', 'AL', 'MS', 'FL'))
  
  dat <- f %>%
    raster::raster() %>%
    raster::mask(states) %>%
    raster::rasterToPoints() %>%
    `colnames<-`(c('x', 'y', 'val')) %>%
    tibble::as_tibble() %>%
    dplyr::mutate(val = factor(val))
    
  out_name = stringr::str_replace(basename(f), '.tif', '.png')
  
  out <- ggplot() + 
    geom_raster(data = dat, mapping = aes(x=x, y=y, fill=val)) + 
    geom_sf(data = states, mapping = aes(), fill=NA, size = 0.5) +
    plot_theme() + 
    scale_fill_manual(values = c(`0` = NA,
                                 `1` = '#FFFF00',
                                 `2` = '#FCD37F',
                                 `3` = '#FFAA00',
                                 `4` = '#E60000',
                                 `5` = '#730000')) +
    labs(x='', y='', fill='Drought\nCategory', title=out_name) + 
    scale_y_discrete(guide = guide_axis(check.overlap = TRUE)) 

  ggsave(glue::glue('./data/plot_data/animations/{out_name}'), height=3,
         width=5, unit = 'in', dpi=320)
}


library(foreach)
library(doParallel)

cl <- makeCluster(9) #not to overload your computer
registerDoParallel(cl)

f_list <- list.files('./data/out_classes/tif', full.names = T, pattern = '2007') 

foreach(i=1:length(f_list), .packages=c('magrittr', 'glue', 'ggplot2', 'dplyr', 'sf', 'raster', 'urbnmapr')) %dopar% {
  tidy_drought_plot(f_list[i])
}
#stop cluster
stopCluster(cl)

