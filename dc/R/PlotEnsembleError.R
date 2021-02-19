library(reticulate)
library(magrittr)
library(ggplot2)
use_condaenv("gee", conda = "/home/colin/miniconda3/condabin/conda")
source_python('./crdm/utils/ReadPickle.py')
source('./crdm/R/PlotTheme.R')

strip_text = function(x) {
  
  x %>%
    stringr::str_split('-') %>%
    lapply(magrittr::extract, -1) %>%
    as.character()
}

read_file <- function(f) {
  
  tmp <- f %>% 
    stringr::str_split('/') %>% 
    unlist() 
    
  ens <- tmp %>%
    tail(2) %>% 
    magrittr::extract(1) %>%
    stringr::str_sub(-1, -1) %>% 
    as.numeric()
  
  err <- tmp %>%
    tail(3) %>% 
    magrittr::extract(1) %>% 
    stringr::str_split('_') %>% 
    unlist() %>% 
    magrittr::extract(1)
  
  read_pickle(f) %>%
    lapply(function(x) lapply(x,mean)) %>%
    lapply(tibble::as_tibble) %>%
    dplyr::bind_rows(.id = 'epoch') %>%
    dplyr::mutate(f = basename(f)) %>%
    tidyr::separate(
      f,
      c('epochs', 'batch', 'nWeeks', 'hiddenSize', 
        'rmFeatures', 'init', 'numLayers', 'stateful', 'fType'), 
      sep='_'
    ) %>%
    dplyr::mutate_at(
      c('epochs', 'batch', 'nWeeks', 'hiddenSize', 
        'rmFeatures', 'init', 'numLayers', 'stateful', 'fType'),
      strip_text
    ) %>%
    dplyr::mutate_at(c('epoch', 'batch', 'nWeeks', 'hiddenSize',
                       'numLayers'), as.numeric
    ) %>%
    dplyr::mutate(ensemble = ens,
                  err_type = err)
    
}

plot_all <- function(f_dir='./data/model_results/weekly_results') {
  
  f_dir %>%
    list.files(full.names = T, pattern = 'err.p', recursive = T) %>%
    grep('old_norm_methods', ., value = T, invert = T) %>%
    lapply(read_file) %>%
    dplyr::bind_rows() %>%
    tidyr::pivot_longer(c(train, test), names_to='set') %>%
    dplyr::mutate(batch = factor(batch),
                  hiddenSize = factor(hiddenSize),
                  nWeeks = factor(nWeeks),
                  numLayers = factor(numLayers),
                  ensemble = factor(ensemble)) -> a
    
    a %>%
     dplyr::filter(set == 'train', err_type == 'cce') %>% 
     ggplot(aes(x=epoch, y=value, color=set)) + 
     geom_line() +
     facet_wrap(~ensemble) + 
     labs(x='Epoch', y='Loss', color='Set') + 
     plot_theme()
}




