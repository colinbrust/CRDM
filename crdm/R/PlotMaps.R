library(ggplot2)
library(magrittr)
source('https://raw.githubusercontent.com/colinbrust/CRDM/develop/crdm/R/PlotTheme.R')

map_to_tidy <- function(f) {
  
  type <- f %>%
    basename() %>%
    stringr::str_replace('.csv', '') %>%
    stringr::str_split('_') %>%
    unlist() %>% 
    tail(1)

  f %>%
    readr::read_csv(col_names = F, col_types = readr::cols()) %>%
    tibble::rowid_to_column() %>%
    tidyr::pivot_longer(-rowid, 
                        names_to = 'col',
                        values_to = 'val') %>%
    dplyr::mutate(col = as.numeric(stringr::str_replace(col, 'X', '')),
                  rowid = rowid * -1,
                  val = ifelse(val < 0.01, 0, val),
                  val = as.character(val),
                  val = dplyr::recode(
                    val, 
                    `0` = 'No Drought',
                    `1` = 'D0', 
                    `2` = 'D1',
                    `3` = 'D2',
                    `4` = 'D3',
                    `5` = 'D4'),
                  val = ifelse(val == 'NA', NA, val),
                  type = type) 
  
}

plot_single_date <- function(pattern, f_list, out_dir) {
  
  print(pattern)
  out_name <- file.path(out_dir, paste0(pattern, '_map.png'))
  
  parts <- out_name %>% 
    basename() %>%
    stringr::str_split('_') %>% 
    unlist()
  
  day <- parts[1]
  lead <- parts[8] %>% stringr::str_replace('leadTime-', '')
  
  caption <- paste('Prediction for', lubridate::as_date(day), 'wtih', lead, 'Month Lead Time')
  
  fig <- f_list %>%
    grep(pattern, ., value = T) %>%
    lapply(map_to_tidy) %>%
    dplyr::bind_rows() %>%
    dplyr::mutate(type = ifelse(type == 'pred', 
                                'Modeled Drought', 
                                'USDM Drought')) %>%
    ggplot(aes(x=col, y=rowid, fill=val)) + 
    geom_raster() +
    theme(aspect.ratio = 264/610) + 
    labs(x='', y='', fill='Drought\nCategory', title = pattern) +
    scale_fill_manual(values = c('No Drought' = '#d3d3d3',
                                 'D0' = '#FFFF00',
                                 'D1' = '#FCD37F',
                                 'D2' = '#FFAA00',
                                 'D3' = '#E60000',
                                 'D4' = '#730000')) + 
    facet_wrap(~type, nrow=2) + 
    labs(title=caption) + 
    plot_theme() +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank()) 

  ggsave(out_name, fig, width = 7, height = 5, units = 'in',
         dpi = 300)
}

save_all <- function(f_dir='~/projects/CRDM/data/drought/model_results/pred_maps',
                     out_dir='~/projects/CRDM/figures') {
  
  f_list <- list.files(f_dir, full.names = T, pattern = '.csv') 
  
  patterns <- f_list %>%
    basename() %>%
    stringr::str_replace('_pred.csv', '') %>%
    stringr::str_replace('_real.csv', '') %>%
    unique()
  
  lapply(patterns, plot_single_date, f_list = f_list, out_dir = out_dir)
}
