library(magrittr)
library(foreach)
library(doParallel)
cl <- makeCluster(10)
registerDoParallel(cl)
args <- commandArgs(trailingOnly=TRUE)

# Stack rasters of same variable, normalize between -1 and 1, write out
normalize <- function(variable, in_dir, out_dir) {

  f_list <- list.files(in_dir, pattern = variable, full.names = T) 
  
  rnge <- f_list %>% 
    raster::stack() 
    raster::`NAvalue<-`(-9999) %>% 
    raster::values() %>%
    range(na.rm = T)
  
  foreach(i=1:length(f_list), .packages=c('raster', 'magrittr')) %dopar% {
    
    f = f_list[i]
    
    f %>% 
      raster::raster() %>%
      raster::calc(function(x) {(2*(x - rnge[1])/(rnge[2] - rnge[1])) - 1}) %>%
      raster::writeRaster(
        filename = file.path(out_dir, basename(f)), 
        overwrite = T
      )
    
  }
    raster::calc(function(x) {(2*(x - rnge[1])/(rnge[2] - rnge[1])) - 1}) %>%
    raster::as.list()
  
}  

variables = c('pr', 'rmax', 'rmin', 'sm-rootzone', 'sm-surface', 'srad', 'tmmn', 
              'tmmx', 'vpd', 'vs', 'fw', 'VOD') %>% paste0('.tif')

# variables = c('gpp', 'ET') %>% paste0('.tif')

variables %>%
  lapply(normalize, in_dir = args[1], out_dir = args[2])
