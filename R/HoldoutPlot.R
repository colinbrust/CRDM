map_to_tidy <- function(stack) {
  stack %>% 
    raster::rasterToPoints() %>%
    tibble::as_tibble() %>%
    `names<-`(c('x', 'y', paste('lt', 1:12, sep='_'))) %>%
    tidyr::pivot_longer(
      dplyr::starts_with('lt'),
      names_to = 'lead_time',
      values_to = 'val'
    )
}

none <- raster::stack('./data/err_maps/annual/err_None.tif')
pr <- raster::stack('./data/err_maps/annual/err_pr.tif')
vpd <- raster::stack('./data/err_maps/annual/err_vpd.tif')
srad <- raster::stack('./data/err_maps/annual/err_srad.tif')
rz <- raster::stack('./data/err_maps/annual/err_sm-rootzone.tif')
surf <- raster::stack('./data/err_maps/annual/err_sm-surface.tif')

