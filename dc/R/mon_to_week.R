library(magrittr)
library(rlang)

mon_to_week <- function(mon_dir, week_dir, out_dir) {
  
  mon_df = fname_to_df(mon_dir) %>%
    dplyr::group_by(.data$variable)
  
  week_df = fname_to_df(week_dir)
  
  match_dates <- as.character(unique(week_df$day))
  
  for (d in match_dates) {
    
    d = as.Date(d)
    print(d)
    
    out <-  mon_df %>%
      dplyr::slice(which.min(abs(d - .data$day))) %>%
      dplyr::filter(abs(d - .data$day) <= 31) %>%
      dplyr::mutate(out_name = file.path(
        out_dir,
        paste0(stringr::str_replace_all(as.character(d), '-', ''), '_', .data$variable))
      )
    
    file.copy(out$fname, out$out_name)
    
  }
}

fname_to_df <- function(f_list) {
  
  f_list %>%
    list.files(full.names = T) %>%
    tibble::tibble(fname = .) %>%
    dplyr::mutate(
      'base' = basename(.data$fname)
    ) %>%
    tidyr::separate(.data$base, c('day', 'variable'), '_') %>%
    dplyr::mutate(
      'day' = lubridate::as_date(.data$day)
    )
}

mon_to_week('~/data/in_features/global_norm/monthly_mem', '~/data/in_features/global_norm/weekly_mem', '~/data/in_features/global_norm/weekly_mem')
