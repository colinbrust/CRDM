library(reticulate)
library(magrittr)
library(ggplot2)
use_condaenv("gee", conda = "/opt/miniconda3/bin/conda")
source_python('~/projects/DroughtCast/crdm/utils/ReadPickle.py')
source('https://raw.githubusercontent.com/colinbrust/DroughtCast/revert/crdm/R/PlotTheme.R')

strip_text = function(x) {
  
  x %>%
    stringr::str_split('-') %>%
    lapply(magrittr::extract, -1) %>%
    unlist()
}

read_file <- function(f) {
  
  read_pickle(f) %>%
    lapply(function(x) lapply(x,mean)) %>%
    lapply(tibble::as_tibble) %>%
    dplyr::bind_rows() %>%
    tibble::rowid_to_column() %>%
    dplyr::mutate(f = basename(f)) %>%
    tidyr::separate(
      f,
      c('epochs', 'batch', 'nWeeks', 'hiddenSize', 'leadTime',
        'rmFeatures', 'init', 'rerun', 'fType'), 
      sep='_'
    ) %>%
    dplyr::select(-dplyr::starts_with('drop')) %>%
    dplyr::mutate_at(
      c('epochs', 'batch', 'nWeeks', 'hiddenSize', 'leadTime', 
        'rmFeatures', 'init', 'rerun', 'fType'), strip_text
    ) %>%
    dplyr::mutate_at(
      c('epochs', 'batch', 'nWeeks', 'hiddenSize', 'rerun'), as.numeric
    ) 
}

plot_all <- function(f_dir='~/projects/DroughtCast/data/model_results/lt_all/') {
  
  f_dir %>%
    list.files(full.names = T, pattern = 'err.p') %>%
    lapply(read_file) %>%
    dplyr::bind_rows() %>%
    dplyr::group_by(rerun) %>% 
    dplyr::mutate(epochs = max(rowid))
  
  
    tidyr::pivot_longer(c(train, test), names_to='set') %>%
    dplyr::mutate(batch = factor(batch),
                  hiddenSize = factor(hiddenSize),
                  nWeeks = factor(nWeeks)) %>%
    ggplot(aes(x=rowid, y=value, color=set)) + 
     geom_line() +
     labs(x='Epoch', y='MSE Loss', color='Train/Test Set') + 
     plot_theme()
}

