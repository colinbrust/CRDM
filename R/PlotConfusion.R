library(magrittr)
library(ggplot2)
source('/mnt/e/PycharmProjects/DroughtCast/crdm/R/PlotTheme.R')


plot_confusion <- function(data_dir, lead_time, set, rm=TRUE) {

  lt_use = lead_time - 1
  
  set_use = dplyr::recode(
    set,
    'test' = 'Spatial Holdout',
    'train' = 'Training Set',
    'val' = 'Temporal Holdout'
  )
  
  dat <- list.files(
    data_dir, 
    pattern = glue::glue('conf_{lt_use}_{set}'),
    full.names = T
  ) %>%
    lapply(function(x) {
      
      data_source = ifelse(
        stringr::str_detect(x, 'base'),
        'Baseline Predictions',
        'Model Predictions'
      ) 
      
      readr::read_csv(
        x, 
        col_names=c('No Drought', 'D0', 'D1', 'D2', 'D3', 'D4'),
        col_types = readr::cols()
      ) %>%
        dplyr::mutate(Class = c('No Drought', 'D0', 'D1', 'D2', 'D3', 'D4')) %>%
        tidyr::pivot_longer(
          -Class,
          names_to = 'Prediction',
          values_to = 'count'
        ) %>%
        dplyr::mutate(
          Class = factor(
            Class, 
            levels = rev(c('No Drought', 'D0', 'D1', 'D2', 'D3', 'D4'))
          ),
          Prediction = factor(
            Prediction, 
            levels = c('No Drought', 'D0', 'D1', 'D2', 'D3', 'D4')
          ),
          count = round(count),
          src = data_source
        )   
   }) %>%
    dplyr::bind_rows()
  
  if (rm) {
    dat %<>%
      dplyr::filter(Class != 'No Drought', Prediction != 'No Drought')
  }
  
  
  pal <- colorRampPalette(RColorBrewer::brewer.pal(10, 'Spectral'))
  
  # True = Rows, Predicted = Cols
  ggplot(dat, aes(x = Prediction, y = Class)) + 
    geom_tile(aes(fill = count), color = 'white', size = 2) + 
    geom_text(aes(label = scales::comma(count)), vjust = 1, color='white') + 
    scale_fill_gradientn(
      na.value='grey26', colors = pal(100), labels=scales::comma
    ) + 
    labs(
      fill = 'Pixel\nCount', 
      title = glue::glue(
        'Confusion Matrix of {set_use} for {lead_time} Week Lead Time'
      )
    ) + 
    plot_theme() + 
    coord_fixed() + 
    facet_wrap(~src, ncol=2) 
  
}
