library(magrittr)
library(ggplot2)
source('./R/PlotTheme.R')

plot_ts_density <- function(f = './data/plot_data/bin_counts_median.csv') {
  
  dat <- f %>%
    readr::read_csv() %>%
    dplyr::filter(category != 6) %>%
    dplyr::mutate(day = lubridate::as_date(as.character(day)),
                  day = day + lubridate::weeks(lt - 1)
                  ) %>%
    dplyr::filter(lt %in% c(2, 4, 8, 12)) %>%
    tidyr::pivot_wider(names_from = lt, values_from = pred) %>%
    tidyr::pivot_longer(
      c('targ', `2`, `4`, `8`, `12`),
      names_to = 'name', 
      values_to = 'val'
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
    dplyr::group_by(day, name) %>%
    dplyr::mutate(pct = (val / sum(val))) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      name = dplyr::recode(name, 
        'targ' = 'USDM Drought',
        '2' = '2 Week Lead Time (Model)',
        '4' = '4 Week Lead Time (Model)',
        '8' = '8 Week Lead Time (Model)',
        '12' = '12 Week Lead Time (Model)'
      ),
      name = factor(
        name, 
        levels = c('USDM Drought', '2 Week Lead Time (Model)',
                   '4 Week Lead Time (Model)', '8 Week Lead Time (Model)',
                   '12 Week Lead Time (Model)'))
    )

  dateRanges <- tibble::tibble(
    from=as.Date(c('2007-01-01', '2014-01-01', '2017-01-01')), 
    to=as.Date(c('2007-12-31', '2014-12-31', '2017-12-31'))
  )
  
  out <- ggplot() + 
    geom_bar(
      data = dat, 
      mapping = aes(x=day, y=pct, fill=category), 
      stat='identity',
      width = 7.5) +
    geom_rect(
      data = dateRanges,
      mapping = aes(xmin = from, xmax = to, ymin = 0, ymax = 1),
      alpha = 0.4,
      show.legend = F
    ) +
    scale_fill_manual(values = c('No Drought' = NA,
                                 'D0' = '#FFFF00',
                                 'D1' = '#FCD37F',
                                 'D2' = '#FFAA00',
                                 'D3' = '#E60000',
                                 'D4' = '#730000')) + 
    facet_wrap(~name, nrow=5) + 
    plot_theme() + 
    labs(x='', y='Percent Coverage', fill = 'USDM\nCategory',
         title = 'CONUS Areal Drought Coverage (%)') +
    scale_y_continuous(labels = scales::percent_format()) 
  
  ggplot2::ggsave('./data/plot_data/final_figures/areal_coverage.png', out, 
                  width = 7.5, height = 7.5, units = 'in', dpi = 320)
    
}