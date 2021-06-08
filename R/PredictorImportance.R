library(magrittr)
library(ggplot2)
library(patchwork)
source('./crdm/R/PlotTheme.R')

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) 

dat <- fst::read_fst('./data/plot_data/monthly_holdouts.fst') %>% 
  tibble::as_tibble() 

spatial_importance <- function(dat) {
  
  out <- dat %>%
    tidyr::pivot_longer(
      cols = dplyr::starts_with('lt_')
    ) %>%
    dplyr::group_by(x, y, holdout) %>%
    dplyr::summarise(value = mean(value)) %>% 
    tidyr::pivot_wider(
      names_from = holdout, 
      values_from = value
    ) %>% 
    tidyr::pivot_longer(
      cols = -c(x, y, None),
      names_to = 'holdout'
    ) %>%
    dplyr::mutate(
      diff = None - value
    ) %>%
    dplyr::group_by(x, y) %>%
    dplyr::arrange(x, y, diff) %>%
    dplyr::mutate(rank = 1:dplyr::n(),
                  holdout = dplyr::recode(
                    holdout,
                    'gpp' = 'GPP',
                    'ET' = 'ET',
                    'pr' = 'PPT',
                    'rmax' = 'RH Max',
                    'rmin' = 'RH Min',
                    'sm-rootzone' = 'RZSM',
                    'sm-surface' = 'SFSM',
                    'srad' = 'SRAD', 
                    'tmmn' = 'TMIN',
                    'tmmx' = 'TMAX',
                    'vpd' = 'VPD',
                    'vs' = 'WS'
                  )) 
  
  pal <- colorRampPalette(RColorBrewer::brewer.pal(10, 'Spectral'))
  
  ggplot() + 
    geom_raster(aes(x=x, y=y, fill=rank), data = out) + 
    geom_sf(aes(), data = states, fill = NA) + 
    facet_wrap(~holdout) + 
    scale_fill_gradientn(colors = rev(pal(12))) + 
    plot_theme() + 
    labs(x='', y='', fill='Importance\nRank', title = 'Predictor Importance in Forecasting USDM') 
  
}

lead_time_importance <- function(dat) {
  
  out <- dat %>%
    tidyr::pivot_longer(
      cols = dplyr::starts_with('lt_'),
      names_to = 'lead_time'
    )  %>%
    dplyr::group_by(holdout, lead_time) %>% 
    dplyr::summarise(value = mean(value)) %>%
    tidyr::pivot_wider(names_from = holdout, values_from = value) %>%
    tidyr::pivot_longer(
      -c(lead_time, None), 
      names_to = 'holdout', 
      values_to = 'value'
    ) %>% 
    dplyr::mutate(
      diff = None - value
    ) 
  
  out %>% 
    dplyr::arrange(lead_time, diff) %>%
    dplyr::group_by(lead_time) %>%
    dplyr::mutate(rank = factor(1:dplyr::n()), 
                  lead_time = stringr::str_replace(lead_time, 'lt_', ''),
                  lead_time = factor(as.numeric(lead_time)),
                  holdout = dplyr::recode(
                    holdout,
                    'gpp' = 'GPP',
                    'ET' = 'ET',
                    'pr' = 'PPT',
                    'rmax' = 'RH Max',
                    'rmin' = 'RH Min',
                    'sm-rootzone' = 'RZSM',
                    'sm-surface' = 'SFSM',
                    'srad' = 'SRAD', 
                    'tmmn' = 'TMIN',
                    'tmmx' = 'TMAX',
                    'vpd' = 'VPD',
                    'vs' = 'WS'
                  )) %>%
    ggplot(aes(x=lead_time, y= rank)) + 
    geom_text(aes(label=holdout)) + 
    labs(x = 'Lead Time (Weeks)', y = '(More Important) -------------------------------- Predictor Importance -------------------------------- (Less Important)') +
    plot_theme()

}

