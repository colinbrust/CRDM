get_usdm <- function(start_date, end_date, out_dir) {
  
  possible_dates <- seq(as.Date('2000-01-04'), as.Date(Sys.time()), by='1 week')
  start_date <- as.Date(start_date)
  end_date <- as.Date(end_date)
  
  if (!(start_date %in% possible_dates)) {
    print('No USDM forecast for given start date. Using closest valid date.')
    start_date <- possible_dates[which.min(abs(start_date - possible_dates))]
  }

  if (!(end_date %in% possible_dates)) {
    print('No USDM forecast for given end date. Using closest valid date.')
    end_date <- possible_dates[which.min(abs(end_date - possible_dates))]
  }

  if (start_date > end_date) return("Error: start_date is larger than end_date")
  
  base <- "https://droughtmonitor.unl.edu/data/shapefiles_m/USDM_"
  
  possible_dates <- possible_dates[possible_dates >= start_date &
                                   possible_dates < end_date]
  
  possible_dates %>%
    as.character() %>%
    stringr::str_replace_all('-', '') %>%
    paste0(base, ., '_M.zip') %>%
    lapply(function(x) {
      temp <- tempfile()
      download.file(x, temp)
      unzip(temp, exdir=out_dir)
      file.remove(temp)
    })
}

# Replace NA values with zero 
na_replace <- function(x) {
  x[is.na(x)] <- 0
  x
}

dm_to_raster <- function(x, template, out_dir) {
  
  x2 <- basename(x) %>%
    stringr::str_replace('.shp', '') %>%
    stringr::str_split('_') %>% 
    unlist() %>%
    {paste0(.[2], '_', .[1], '.tif')}
  
  name <- file.path(out_dir, x2)
  print(name)
  
  x %>%
    sf::read_sf() %>%
    sf::st_transform(6933) %>%
    fasterize::fasterize(template, field='DM') %>%
    raster::calc(function(x) x + 1) %>%
    na_replace() %>%
    raster::writeRaster(name, overwrite = T)
}

usdm_to_raster <- function(basemap, f_dir = "/mnt/e/Data/USDM/shapefiles", 
                           out_dir='/mnt/e/PycharmProjects/CRDM/data/out_classes') {
  
  basemap <- raster::raster(basemap)
  
  f_dir %>%
    list.files(full.names = T, pattern = '.shp') %>%
    grep('.xml', ., value = T, invert = T) %>% tail(24) %>%
    lapply(dm_to_raster, template=basemap, out_dir=out_dir)
}
# get_usdm('2000-01-01', '2020-01-01', '/Users/cbandjelly/Workspace/data/usdm')

usdm_to_raster(basemap = '~/projects/CRDM/data/drought/template.tif')


