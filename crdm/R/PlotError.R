library(reticulate)
library(magrittr)
library(ggplot2)
use_condaenv("gee", conda = "/opt/miniconda3/bin/conda")
source_python('~/projects/CRDM/crdm/utils/ReadPickle.py')

strip_text = function(x) {
  
  x %>%
    stringr::str_split('-') %>%
    lapply(magrittr::extract, -1) %>% 
    as.numeric()
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
      c('drop1', 'epochs', 'batch', 'nMonths', 'hiddenSize', 'leadTime', 'drop2'), 
      sep='_'
    ) %>%
    dplyr::select(-dplyr::starts_with('drop')) %>%
    dplyr::mutate_at(
      c('epochs', 'batch', 'nMonths', 'hiddenSize', 'leadTime'),
      strip_text
    )
}

plot_all <- function(f_dir) {
  
  f_dir %>%
    list.files(full.names = T, pattern = 'err.p') %>%
    lapply(read_file) %>%
    dplyr::bind_rows() %>%
    tidyr::pivot_longer(c(train, test), names_to='set') %>%
    dplyr::mutate(batch = factor(batch),
                  hiddenSize = factor(hiddenSize)) %>%
    dplyr::filter(set == 'train') %>%
    ggplot(aes(x=rowid, y=value, color=batch)) + 
     geom_line() +
     facet_wrap(~hiddenSize) + 
     labs(x='Epoch', y='Cross-Entropy Loss', color='Batch Size') + 
     theme_bw()
}