library(magrittr)
library(ggplot2)
source('./crdm/R/PlotTheme.R')

plot_ts_density <- function(f = './data/err_maps/bin_counts.csv', lead_time=1) {
  
  dat <- f  %>%
    readr::read_csv() %>%
    dplyr::filter(category != 6) %>%
    dplyr::mutate(day = lubridate::as_date(as.character(day))) %>%
    tidyr::pivot_longer(
      c(pred, targ),
      values_to = 'val', 
      names_to = 'name'
    ) %>%
    dplyr::mutate(category = dplyr::recode(
      category,
      `0` = 'No Drought',
      `1` = 'D0',
      `2` = 'D1',
      `3` = 'D2',
      `4` = 'D3',
      `5` = 'D4'),
      category = factor(category, 
                        levels = c('No Drought', 'D0','D1','D2','D3','D4'))) %>%
    dplyr::group_by(lt, day, name) %>%
    dplyr::mutate(pct = 100 * (val / sum(val))) %>%
    dplyr::ungroup() %>%
    dplyr::filter(lt %in% c(1, 3, 7, 11)) %>%
    dplyr::mutate(
      lt = lt + 1,
      lt = paste(lt, 'Week Lead Time'), 
      name = ifelse(name == 'pred', 'Model', 'USDM'),
      lt = factor(lt, levels=c('2 Week Lead Time', '4 Week Lead Time', 
                                   '8 Week Lead Time', '12 Week Lead Time')))
  
  dateRanges <- tibble::tibble(
    from=as.Date(c('2007-01-01', '2014-01-01', '2017-01-01')), 
    to=as.Date(c('2007-12-31', '2014-12-31', '2017-12-31'))
  )
  
  ggplot() + 
    geom_bar(
      data = dat, 
      mapping = aes(x=day, y=pct, fill=category), 
      stat='identity',
      width = 7.5) +
    geom_rect(
      data = dateRanges,
      mapping = aes(xmin = from, xmax = to, ymin = 0, ymax = 100),
      alpha = 0.4
    ) +
    scale_fill_manual(values = c('No Drought' = NA,
                                 'D0' = '#FFFF00',
                                 'D1' = '#FCD37F',
                                 'D2' = '#FFAA00',
                                 'D3' = '#E60000',
                                 'D4' = '#730000')) + 
    facet_grid(rows = dplyr::vars(lt), cols = dplyr::vars(name)) + 
    plot_theme() + 
    labs(x='', y='Percent Coverage', fill = 'USDM\nCategory')
    
}