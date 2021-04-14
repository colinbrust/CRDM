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
    dplyr::mutate(across(!dplyr::starts_with('feats'), as.numeric)) %>%
    dplyr::mutate(model_id = basename(dirname(f))) 
}

read_error <- function(f) {
  
  read_pickle(f) %>%
    lapply(function(x) lapply(x, mean)) %>%
    lapply(tibble::as_tibble) %>%
    dplyr::bind_rows() %>%
    dplyr::mutate(model_id = basename(dirname(f))) %>%
    tibble::rowid_to_column() %>%
    dplyr::rename(epoch=rowid)
  
}

read_all <- function(pth) {
  
  err <-  list.files(pth, recursive = T, full.names = T, pattern = 'err.p') %>%
    lapply(read_error) %>%
    dplyr::bind_rows()
  
  metadata <- list.files(
    pth, recursive = T, full.names = T, pattern = 'metadata.p'
    ) %>% 
    lapply(read_metadata) %>% 
    dplyr::bind_rows()
  
  dplyr::left_join(err, metadata, by='model_id')

}

plot_all <- function(pth, ...) {
  

  read_all(pth) %>%
    tidyr::pivot_longer(c(train, test), names_to = 'set', values_to = 'err') %>% 
    dplyr::mutate(model_id = stringr::str_replace(model_id, 'model', '') %>%
                    as.numeric() %>% 
                    factor()) %>%
    dplyr::filter(set == 'train') %>% 
    ggplot(aes(x=epoch, y=err, color=model_id)) + 
      geom_line() 
    
    
}

read_file <- function() {
  
  read_pickle(f) %>%
    lapply(function(x) lapply(x,mean)) %>%
    lapply(tibble::as_tibble) %>%
    dplyr::bind_rows() %>%
    tibble::rowid_to_column() %>%
    dplyr::mutate(f = basename(f)) %>%
}

plot_all <- function(f_dir='~/projects/DroughtCast/data/model_results/unet/crop16/') {
  
  f_dir %>%
    list.files(full.names = T, pattern = 'err.p') %>%
    lapply(read_file) %>%
    dplyr::bind_rows() %>%
    dplyr::arrange(rerun, rowid) %>%
    dplyr::select(-rowid) %>% 
    tibble::rowid_to_column() %>% 
    tidyr::pivot_longer(c(train, test), names_to='set') %>%
    dplyr::mutate(batch = factor(batch),
                  hiddenSize = factor(hiddenSize),
                  nWeeks = factor(nWeeks)) %>%
    ggplot(aes(x=rowid, y=value, color=set)) + 
     geom_line() +
     labs(x='Epoch', y='MSE Loss', color='Train/Test Set') + 
     plot_theme()
}

