library(magrittr)
library(raster)
library(fst)

raster_to_fst(f_dir = '~/Box/school/Data/drought/in_features/monthly', 
              out_dir = '~/Box/school/Data/drought/in_fst/monthly', 
              variable = 'pr') {
  
  list.files(f_dir, full.names = T, pattern = variable) %>%
    raster::stack() %>%
    raster::as.data.frame(t, na.rm = F) %>%
    `names<-`(as.character(1:ncol(.))) %>%
    fst::write_fst(file.path(out_dir, paste0(variable, '.fst')), compress = 100)
}


tdf %<>% `names<-`(as.character(1:ncol(.)))


from_fst <- function() {fst::read_fst(file.path(out_dir, paste0(variable, '.fst')), columns='1') %>%
  unlist() %>%
  matrix(nrow=264, ncol=610, byrow = T) 
}


from_tif <- function() {raster::raster('~/Box/school/Data/drought/in_features/monthly/20020101_pr.tif') }
  