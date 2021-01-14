args <- commandArgs(trailingOnly=TRUE)

# Normalize data between -1 and 1
norm_calc <- function(x) {
  
  (2*(x - min(x, na.rm = T))/(max(x, na.rm = T) - min(x, na.rm = T))) - 1
  
}

# Stack rasters of same variable, normalize between -1 and 1, write out
normalize <- function(variable, in_dir, out_dir) {

  f_list <- list.files(in_dir, pattern = variable, full.names = T) %>% head(20)
  
  stk <-  f_list %>%
    raster::stack() %>%
    raster::calc(norm_calc) %>%
    raster::as.list()
  
  out_names <- file.path(out_dir, basename(f_list))
  
  mapply(raster::writeRaster, x=stk, filename=out_names)
}

variables = c('fw', 'pr', 'rmax', 'rmin', 'sm-surface', 'sm-rootzone', 
              'srad', 'tmmn', 'tmmx', 'VOD', 'vpd', 'vs') %>% paste0('.tif')

variables %>%
  parallel::mclapply(normalize, in_dir = args[[1]], out_dir = args[[2]], mc.cores=12)