library(reticulate)
library(magrittr)
library(ggplot2)
use_condaenv("gee", conda = "/opt/miniconda3/bin/conda")
source_python('~/projects/DroughtCast/crdm/utils/ReadPickle.py')
source('https://raw.githubusercontent.com/colinbrust/DroughtCast/revert/crdm/R/PlotTheme.R')


read_metadata <- function(f) {
  
  read_pickle(f) %>% 
    lapply(function(x) paste(x, collapse=', ')) %>%
    tibble::as_tibble() %>% 
    dplyr::mutate(model_class = basename(dirname(f)), 
                  model_id = basename(f) %>% 
                    stringr::str_split('_') %>% 
                    unlist() %>% 
                    magrittr::extract(2)) 
}

read_error <- function(f) {
  
  read_pickle(f) %>%
    lapply(function(x) lapply(x, mean)) %>%
    lapply(tibble::as_tibble) %>%
    dplyr::bind_rows() %>%
    dplyr::mutate(model_class = basename(dirname(f)), 
                  model_id = basename(f) %>% 
                    stringr::str_split('_') %>% 
                    unlist() %>% 
                    magrittr::extract(2)) %>%
    tibble::rowid_to_column() %>%
    dplyr::rename(epoch=rowid)
  
}

read_all <- function(pth) {
   
  err <-  list.files(pth, recursive = T, full.names = T, pattern = 'err_') %>%
    lapply(read_error) %>%
    dplyr::bind_rows() %>%
    dplyr::mutate(model_id = as.numeric(model_id))
  
  metadata <- list.files(
    pth, recursive = T, full.names = T, pattern = 'metadata_'
    ) %>% 
    lapply(read_metadata) %>% 
    dplyr::bind_rows()  %>%
    dplyr::select(-c(dirname, criterion, pix_mask, model_type, early_stop, 
                     in_features, out_classes, seq, clip)) %>%
    dplyr::mutate(dplyr::across(!dplyr::ends_with('class'), as.numeric))
  
  
  dplyr::left_join(err, metadata, by=c('model_id', 'model_class'))

}

plot_all <- function(pth, ...) {
  

  read_all(pth) %>%
    tidyr::pivot_longer(c(train, test), names_to = 'set', values_to = 'err') %>% 
    dplyr::mutate(model_id = factor(model_id)) %>%
    dplyr::filter(set == 'test', 
                  model_class %in% c('seq32', 'seq64', 'seq128')) %>% 
    ggplot(aes(x=epoch, y=err, color=model_class)) + 
      geom_line() +
    facet_wrap(~model_id)
    
    
}

