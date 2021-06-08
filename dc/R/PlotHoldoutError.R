library(magrittr)
library(ggplot2)
source('https://raw.githubusercontent.com/colinbrust/DroughtCast/revert/crdm/R/PlotTheme.R')

read_files <- function(f) {
readr::read_csv(f, col_types = readr::cols()) %>%
  janitor::clean_names() %>% 
  tidyr::pivot_longer(
    -c(date, lead_time),
    names_to = 'variable',
    values_to = 'value'
  ) %>%
  dplyr::mutate(date = lubridate::as_date(as.character(date))) 
}


list.files('./data/model_results', recursive = T, pattern = 'err.csv', 
           full.names = T) %>% 
  lapply(read_files) %>% 
  dplyr::bind_rows() %>%
  dplyr::mutate(month = lubridate::month(date)) %>%
  dplyr::group_by(month, lead_time, variable) %>% 
  dplyr::summarise(err = mean(value)) %>% 
  dplyr::filter(variable != 'usdm') %>%
  dplyr::ungroup() %>% 
  tidyr::pivot_wider(names_from = variable, values_from = err) %>% 
  tidyr::pivot_longer(
    -c(month, lead_time, full),
    names_to = 'variable', 
    values_to = 'err'
  ) %>%
  dplyr::mutate(
    variable = dplyr::recode(
      variable, 
      'et' = 'ET',
      'fw' = 'AMSR-FW',
      'gpp' = 'GPP',
      'pr' = 'PPT',
      'rmax' = 'RH-MAX',
      'rmin' = 'RH-MIN',
      'sm_rootzone' = 'SM-Rootzone',
      'sm_surface' = 'SM-Surface',
      'srad' = 'Solar-Rad',
      'tmmn' = 'TMIN',
      'tmmx' = 'TMAX',
      'vapor' = 'AMSR-Vapor',
      'vod' = 'AMSR-VOD',
      'vpd' = 'VPD',
      'vs' = 'Windspeed'
    ), 
    month = month.abb[month],
    month = factor(month, levels = month.abb)
  ) %>% 
  dplyr::mutate(degrade = err - full,
                lead_time = paste0(lead_time, ' Week Lead Time')) %>% 
  dplyr::filter(variable != 'full') %>% 
  ggplot(aes(x=month, y=degrade, fill=variable)) + 
    geom_bar(stat='identity', position='dodge') + 
    scale_fill_viridis_d() + 
    facet_wrap(~lead_time, nrow=3) +
    plot_theme() + 
    labs(x='Month', y='Error Change', fill = 'Variable')

  
  readr::read_tsv('https://psl.noaa.gov/enso/mei/data/meiv2.data', skip=1, head)
  