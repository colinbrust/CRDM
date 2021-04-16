library(magrittr)
library(lubridate)

list.files('~/data/in_features/pix_norm/weekly_mem/', full.names = T) %>%
  grep('gpp.dat|ET.dat', ., value=T, invert=T) %>%
  tibble::tibble(fname = .) %>%
  dplyr::mutate(base = basename(fname)) %>%
  tidyr::separate(base, c('date', 'v'), sep = '_') %>%
  dplyr::mutate(date = lubridate::as_date(date) + 1,
                date = as.character(date) %>% stringr::str_replace_all('-', ''),
                out = file.path(dirname(fname), paste0(date, '_', v))) -> a

print(head(a$fname))
print(head(a$out))
file.rename(a$fname, a$out)

# list.files('~/Box/school/Data/drought/in_features/memmap/monthly_mem', full.names = T) %>%
#   tibble::tibble(fname = .) %>%
#   dplyr::mutate(base = basename(fname)) %>%
#   tidyr::separate(base, c('date', 'v'), sep = '_') %>%
#   dplyr::mutate(date = (lubridate::as_date(date) %m+% months(1)) - 1,
#                 date = as.character(date) %>% stringr::str_replace_all('-', ''),
#                 out = file.path(dirname(fname), paste0(date, '_', v))) -> a

# file.rename(a$fname, a$out)
