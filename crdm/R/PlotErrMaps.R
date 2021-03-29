library(magrittr)

rast_mean <- function(rast_list) {
  
  rast_list %>% 
    lapply(raster::raster) %>% 
    raster::mean()
}

diff_maps <- function(f_dir, variable) {
  
  var_maps <- list.files(f_dir, pattern = variable, full.names = T) %>%
    grep('preds.tif', ., invert = T, value = T) %>%
    tibble::tibble(fname = .) 

  true_maps = list.files(f_dir, pattern = 'full', full.names = T) %>% 
    tibble::tibble(fname = .)
  
  map_list <- dplyr::bind_rows(var_maps, true_maps) %>%
    dplyr::rowwise() %>% 
    dplyr::mutate(
      date = basename(fname) %>%
        stringr::str_sub(1, 8) %>% 
        lubridate::as_date(),
      
      variable = basename(fname) %>% 
        stringr::str_split('_') %>%
        unlist() %>% 
        tail(1) %>% 
        stringr::str_replace('.tif', '')
    ) %>% 
    janitor::get_dupes(date) %>%
    dplyr::filter(dupe_count == 2)
  
  var_maps <- map_list %>% 
    dplyr::filter(variable == !!variable) %>%
    dplyr::arrange(date) %$%
    raster::stack(fname)
  
  true_maps <-  map_list %>% 
    dplyr::filter(variable == 'full') %>%
    dplyr::arrange(date) %$%
    raster::stack(fname)
  
  raster::mean(var_maps - true_maps)
}
