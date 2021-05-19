get_ts <- function(data_dir) {
  test <- list.files(
    './data/models/ensemble_101/preds_37', full.names = T, pattern = 'None'
  ) %>% raster::stack() 
}