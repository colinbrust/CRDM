library(magrittr)
library(kableExtra)

readr::read_csv('./data/plot_data/tables/complete_err.csv') %>% 
  dplyr::mutate(`Lead Time` = glue::glue('{1:12} Week')) %>%
  dplyr::mutate(across(dplyr::ends_with('cor'), ~ .x**2),
                across(where(is.numeric), ~round(.x, 4))) %>%
  dplyr::select(c(`Lead Time`, train_mse, test_mse, val_mse, train_cor, test_cor, val_cor)) %>%
  dplyr::rename(`Train MSE` = train_mse, `Spatial MSE` = test_mse,
                `Temporal MSE` = val_mse, `Train Corr.` = train_cor, 
                `Spatial Corr.` = test_cor, `Temporal Corr.` = val_cor) %>%
  kbl('latex') %>%
  kable_classic() %>%
  add_header_above(c(" " = 1, "MSE (USDM Categories)" = 3, 'Correlation (R^2)' = 3))
  
  
  tidyr::pivot_wider(names_from = lead_time, values_from = value) %>%
  dplyr::mutate(
    set = dplyr::recode(
      set,
      'train' = 'Training Set',
      'test' = 'Spatial Holdout',
      'val' = 'Temporal Holdout'
    ),
    metric = dplyr::recode(
      metric,
      'cor' = 'R2',
      'mse' = 'MSE'
    )
  )
