library(reticulate)
library(magrittr)
library(ggplot2)
use_condaenv("ml", conda = "/home/colin/miniconda3/bin/conda")
source_python('./crdm/utils/ReadPickle.py')
source('https://raw.githubusercontent.com/colinbrust/DroughtCast/revert/crdm/R/PlotTheme.R')


read_metadata <- function(f) {
  
  read_pickle(f) %>% 
    lapply(function(x) paste(x, collapse=', ')) %>%
    tibble::as_tibble() %>% 
    dplyr::mutate(model_class = basename(dirname(f)), 
                  model_id = basename(f) %>% 
                    stringr::str_split('_') %>% 
                    unlist() %>% 
                    magrittr::extract(2)%>%
                    stringr::str_replace('.p', '') %>%
                    as.numeric()) 
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
                    magrittr::extract(2) %>%
                    stringr::str_replace('.p', '') %>%
                    as.numeric()) %>%
    tibble::rowid_to_column() %>%
    dplyr::rename(epoch=rowid)
  
}

read_all <- function(pth) {
   
  err <-  list.files(pth, recursive = T, full.names = T, pattern = 'err_') %>%
    grep('old', ., value = T, invert = T) %>% 
    lapply(read_error) %>%
    dplyr::bind_rows() %>%
    dplyr::mutate(model_id = as.numeric(model_id))
  
  metadata <- list.files(
    pth, recursive = T, full.names = T, pattern = 'metadata_'
    ) %>% 
    lapply(read_metadata) %>% 
    dplyr::bind_rows()  %>%
    dplyr::select(-c(dirname, pix_mask, model_type, early_stop, 
                     in_features, out_classes)) %>%
    dplyr::mutate(dplyr::across(!dplyr::ends_with('class'), as.numeric))
  
  
  dplyr::left_join(err, metadata, by=c('model_id', 'model_class'))

}

plot_all <- function(pth) {
  

  read_all(pth) %>%
    tidyr::pivot_longer(c(train, test), names_to = 'set', values_to = 'err') %>% 
    dplyr::mutate(batch_size = factor(batch_size),
                  hidden_size = factor(hidden_size),
                  model_class = factor(model_class),
                  model_id = factor(model_id)) %>%
    dplyr::filter(set == 'train') %>% 
    ggplot(aes(x=epoch, y=err, color=model_class)) + 
      geom_line() +
    facet_wrap(~model_id) 
    
}

good_ex_fig <- './data/models/global_norm/model4/preds_87/20170627_preds_None.tif'

readr::read_csv('./data/ensemble_results.csv') %>%
  tidyr::separate(pred, c('date', 'drop', 'holdout'), sep = '_') %>%
  dplyr::mutate(model = dirname(name)) %>%
  dplyr::group_by(lead_time, model) %>%
  dplyr::summarise(mse = mean(mse), r2 = mean(r2)) %>%
  dplyr::group_by(lead_time) %>%
  dplyr::slice(which.max(r2))
library(reticulate)
library(magrittr)
library(ggplot2)
use_condaenv("ml", conda = "/home/colin/miniconda3/bin/conda")
source_python('./crdm/utils/ReadPickle.py')
source('https://raw.githubusercontent.com/colinbrust/DroughtCast/revert/crdm/R/PlotTheme.R')


read_metadata <- function(f) {
  
  read_pickle(f) %>% 
    lapply(function(x) paste(x, collapse=', ')) %>%
    tibble::as_tibble() %>% 
    dplyr::mutate(model_class = basename(dirname(f)), 
                  model_id = basename(f) %>% 
                    stringr::str_split('_') %>% 
                    unlist() %>% 
                    magrittr::extract(2)%>%
                    stringr::str_replace('.p', '') %>%
                    as.numeric()) 
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
                    magrittr::extract(2) %>%
                    stringr::str_replace('.p', '') %>%
                    as.numeric()) %>%
    tibble::rowid_to_column() %>%
    dplyr::rename(epoch=rowid)
  
}

read_all <- function(pth) {
  
  err <-  list.files(pth, recursive = T, full.names = T, pattern = 'err_') %>%
    grep('old', ., value = T, invert = T) %>% 
    lapply(read_error) %>%
    dplyr::bind_rows() %>%
    dplyr::mutate(model_id = as.numeric(model_id))
  
  metadata <- list.files(
    pth, recursive = T, full.names = T, pattern = 'metadata_'
  ) %>% 
    lapply(read_metadata) %>% 
    dplyr::bind_rows()  %>%
    dplyr::select(-c(dirname, pix_mask, model_type, early_stop, 
                     in_features, out_classes)) %>%
    dplyr::mutate(dplyr::across(!dplyr::ends_with('class'), as.numeric))
  
  
  dplyr::left_join(err, metadata, by=c('model_id', 'model_class'))
  
}

plot_all <- function(pth) {
  
  
  read_all(pth) %>%
    tidyr::pivot_longer(c(train, test), names_to = 'set', values_to = 'err') %>% 
    dplyr::mutate(batch_size = factor(batch_size),
                  hidden_size = factor(hidden_size),
                  model_class = factor(model_class),
                  model_id = factor(model_id)) %>%
    dplyr::filter(set == 'train') %>% 
    ggplot(aes(x=epoch, y=err, color=model_class)) + 
    geom_line() +
    facet_wrap(~model_id) 
  
}

good_ex_fig <- './data/models/global_norm/model4/preds_87/20170627_preds_None.tif'

readr::read_csv('./data/models/big_test/selection_2/ensemble_results.csv') %>%
  tidyr::separate(pred, c('date', 'drop', 'holdout'), sep = '_') %>%
  dplyr::mutate(model = dirname(name)) %>%
  dplyr::group_by(model) %>%
  dplyr::summarise(mse = mean(mse)) %>%
  dplyr::slice_min(mse, n=10)


