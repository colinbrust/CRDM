# library(raster)
library(magrittr)
args = commandArgs(trailingOnly = TRUE)

f = args[1]
tmp = args[2]
new_name = args[3]

tmp <- raster::raster(tmp)

dat <- f %>%
  raster::raster() %>% 
  raster::`crs<-`(value='+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ') %>%
  raster::projectRaster(tmp) %>%
  raster::crop(tmp)

dat[dat < 0] <- NA

raster::writeRaster(dat, new_name, overwrite = T)
