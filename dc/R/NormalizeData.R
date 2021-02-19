library(magrittr)
args <- commandArgs(trailingOnly=TRUE)

# Stack rasters of same variable, normalize between -1 and 1, write out
normalize_across_domain <- function(variable, in_dir, out_dir) {

  f_list <- list.files(in_dir, pattern = variable, full.names = T) 
  
  rnge <- f_list %>% 
    raster::stack() %>%
    raster::values() %>%
    range(na.rm = T)
  
  for (i in 1:length(f_list)) {
    
    f = f_list[i]
    print(f)
    
    f %>% 
      raster::raster() %>%
      raster::calc(function(x) {(2*(x - rnge[1])/(rnge[2] - rnge[1])) - 1}) %>%
      raster::writeRaster(
        filename = file.path(out_dir, basename(f)), 
        overwrite = T
      )
    
  }
} 

norm_calc <- function(x) {
  
  (2*(x - min(x, na.rm = T))/(max(x, na.rm = T) - min(x, na.rm = T))) - 1
  
}

normalize_across_pixel <- function(variable, in_dir, out_dir) {
  
  f_list <- list.files(in_dir, pattern = variable, full.names = T) 
  
  stk <-  f_list %>%
    raster::stack() %>%
    raster::calc(norm_calc) %>%
    raster::as.list()
  
  out_names <- file.path(out_dir, basename(f_list))
  
  mapply(raster::writeRaster, x=stk, filename=out_names, overwrite = T)
}  

variables = c('pr', 'rmax', 'rmin', 'sm-rootzone', 'sm-surface', 'srad', 'tmmn', 
              'tmmx', 'vpd', 'vs', 'fw', 'vapor', 'vod') %>% paste0('.tif')

# variables = c('gpp', 'ET') %>% paste0('.tif')

variables %>%
  parallel::mclapply(normalize_across_pixel, in_dir = args[1], out_dir = args[2], mc.cores = 13)

  