library(magrittr)

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) 

tidy_raster <- function(r, stack = TRUE, band = 1) {
  print(r)
  
  if (stack) {
    r <- raster::stack(r)
  } else {
    r <- raster::raster(r, band = band)
  }
  
  if (raster:: nlayers(r) == 12) {
    cnames <- paste0('lt_', seq(1:raster::nlayers(r)))
  } else {
    cnames <- 'val'
  }
  
  r %>%
    raster::mask(states) %>% 
    raster::rasterToPoints() %>%
    tibble::as_tibble() %>%
    `colnames<-`(c('x', 'y', cnames)) 
}
