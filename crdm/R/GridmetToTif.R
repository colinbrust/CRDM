library(foreach)
library(doParallel)

args <- commandArgs(trailingOnly=TRUE)

possible_dates <- seq(as.Date('2000-01-04'), as.Date(Sys.time()), by='1 week')
template <- raster::raster(args[[4]])

gm_to_tif <- function(gm_dir='/mnt/e/PycharmProjects/CRDM/data/raw/gridmet/ncdf',
                      out_dir='/mnt/e/PycharmProjects/a;sldgfh',
                      variable='tmmn') {
  
  f_list <-   list.files(gm_dir, pattern=variable, full.names = T)
  
  cl = makeCluster(20)
  registerDoParallel(cl)
  
  foreach(i=1:(length(possible_dates) - 1), 
          .packages=c('lubridate', 'raster', 'magrittr')) %dopar% {
    
    start <- possible_dates[i]
    end <- possible_dates[i+1] - 1
    
    name <- paste0(end %>% stringr::str_replace_all('-', ''), 
                   '_', variable, '.tif')
    
    print(name)
    if (lubridate::year(start) == lubridate::year(end)) {
      
      dat <- f_list %>% 
        grep(lubridate::year(start) %>% as.character(), ., value = T)
      
      if (length(dat) == 0) {next}
      else {
        dat %>%
          raster::stack(bands=lubridate::yday(start):lubridate::yday(end), quick=T) %>%
          raster::mean() %>%
          raster::projectRaster(template) %>% 
          raster::writeRaster(file.path(out_dir, name))
      }
    } else {
    
      dat <- f_list %>% 
        grep(lubridate::year(start) %>% as.character(), ., value = T)
      
      if (length(dat) == 0) {next}
      else {
        first <- dat %>%
          raster::stack()
        
        first <- first[[lubridate::yday(start):raster::nlayers(first)]]
        
        second <- f_list %>% 
          grep(lubridate::year(end) %>% as.character(), ., value = T) %>%
          raster::stack(bands=1:lubridate::yday(end))
        
        raster::stack(first, second) %>%
          raster::mean() %>%
          raster::projectRaster(template) %>% 
          raster::writeRaster(file.path(out_dir, name))
      }
    }
  }
}

gm_to_tif(gm_dir = args[[1]], out_dir = args[[2]], variable = args[[3]])