library(magrittr)


tidy_raster <- function(r, stack = TRUE, band = 1, states) {
  
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
