source('./R/TidyRaster.R')
source('./crdm/R/PlotTheme.R')
library(magrittr)
library(patchwork)
library(ggplot2)

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933)

parse_monthly_data <- function(d) {
  
  d %>%
    list.files(full.names = T, pattern = '.tif') %>%
    grep('drought|mei', ., value = TRUE, invert = TRUE) %>%
    parallel::mclapply(function(x) {
      
      x %>%
        tidy_raster(stack = TRUE) %>%
        dplyr::mutate(
           holdout = !!x,
           holdout = basename(stringr::str_replace(holdout, '.tif', ''))
        ) %>%
        tidyr::separate(holdout, c('drop', 'holdout', 'month'), sep = '_') %>%
        dplyr::select(-drop)
    }, mc.cores = 12) %>%
    dplyr::bind_rows() -> a
}

states <- sf::st_union(states)

mon_plot <- function(dat, lead_time=1, 
                     variable = 'pr') {
  
  lt <- glue::glue('lt_{lead_time}')
  
  out <- dat %>% 
    dplyr::rename(val = !!lt) %>%
    dplyr::select(c('x', 'y', 'holdout', 'month', 'val')) %>%
    # dplyr::filter(holdout %in% c(!!variable, 'None')) %>%
    tidyr::pivot_wider(
      names_from = holdout, 
      values_from = val
    ) %>%
    dplyr::mutate(diff = None - !!rlang::sym(variable),
                  month = factor(month.abb[as.numeric(month)], levels = month.abb),
                  diff = ifelse(diff > 0, 0, diff), 
                  diff = ifelse(abs(diff - median(diff)) > 2*sd(diff), median(diff) - 2*sd(diff), diff)) 
  
  
  pal <- colorRampPalette(RColorBrewer::brewer.pal(10, 'Spectral'))
  
  var_use <- dplyr::recode(
    variable,
    'gpp' = 'GPP',
    'ET' = 'ET',
    'pr' = 'PPT',
    'rmax' = 'RH Max',
    'rmin' = 'RH Min',
    'sm-rootzone' = 'SM',
    'sm-surface' = 'SFSM',
    'srad' = 'SRAD', 
    'tmmn' = 'TMIN',
    'tmmx' = 'TMAX',
    'vpd' = 'VPD',
    'vs' = 'WS'
  )
  
  fig <- ggplot() + 
    geom_raster(aes(x=x, y=y, fill=diff), out) + 
    geom_sf(aes(), states, fill=NA, size = 0.5) + 
    facet_wrap(~month) + 
    plot_theme() + 
    scale_fill_gradientn(colors = pal(10)) + 
    labs(x='', y='', fill='MSE\n(USDM Categories)', 
         title = glue::glue('Increase in Model Error With {var_use} Removed ({lead_time} Week Lead Time)')) +
    theme(axis.text.x = element_text(angle = 45))
  
  
  ggplot2::ggsave(
    glue::glue('./data/plot_data/figs/monthly_err/{variable}_{lead_time}.png'),
    fig, 
    width = 8,
    height = 5,
    units = 'in'
  )
  
}

plot_spatial_error <- function(f_dir = './data/plot_data/figs/spatial_err') {
  out <- list.files(f_dir, full.names = T) %>%
    lapply(function(x) {
      n <- stringr::str_replace(basename(x), '.tif', '')
      
      x %>% 
        tidy_raster(stack = FALSE) %>%
        dplyr::mutate(f = n)
    }) %>%
    dplyr::bind_rows() %>%
    tidyr::separate(f, c('metric', 'set'), sep = '_') %>%
    dplyr::mutate(
        val = ifelse(metric == 'r', val ** 2, val),
        metric = dplyr::recode(
          metric,
          'mse' = 'Mean Squared Error',
          'r' = 'R-Squared'
        ),
        set = dplyr::recode(
          set,
          'train' = 'Training Set',
          'val' = 'Temporal Holdout'
        )
    )
  
  pal <- colorRampPalette(RColorBrewer::brewer.pal(10, 'Spectral'))
  
  p1 <- ggplot() +
    geom_raster(aes(x=x, y=y, fill=val), data = dplyr::filter(out, metric == 'R-Squared')) + 
    geom_sf(aes(), data = states, fill = NA) + 
    facet_grid(rows = dplyr::vars(metric), cols = dplyr::vars(set), switch = 'y') +
    plot_theme() + 
    labs(x='', y='', fill='')
  
  
  p2 <- ggplot() +
    geom_raster(aes(x=x, y=y, fill=val), data = dplyr::filter(out, metric == 'Mean Squared Error')) + 
    geom_sf(aes(), data = states, fill = NA) + 
    facet_grid(rows = dplyr::vars(metric), cols = dplyr::vars(set), switch = 'y') +
    plot_theme() +
    scale_fill_gradientn(colors = rev(pal(100))) +
    labs(x='', y='', fill='')
    
  p1/p2
  

}