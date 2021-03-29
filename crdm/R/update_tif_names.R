list.files('~/data/in', full.names = T) %>%
  tibble::tibble(fname = .) %>%
  dplyr::mutate(base = basename(fname)) %>%
  tidyr::separate(base, c('date', 'v'), sep = '_') %>%
  dplyr::mutate(date = lubridate::as_date(date) + 1,
                date = as.character(date) %>% stringr::str_replace_all('-', ''),
                out = file.path(dirname(fname), paste0(date, '_', v))) -> a

file.rename(a$fname, a$out)

library(lubridate)
list.files('~/Box/school/Data/drought/in_features/fst/monthly', full.names = T) %>%
  tibble::tibble(fname = .) %>%
  dplyr::mutate(base = basename(fname)) %>%
  tidyr::separate(base, c('date', 'v'), sep = '_') %>%
  dplyr::mutate(date = (lubridate::as_date(date) %m+% months(1)) - 1,
                date = as.character(date) %>% stringr::str_replace_all('-', ''),
                out = file.path(dirname(fname), paste0(date, '_', v))) -> a

file.rename(a$fname, a$out)
