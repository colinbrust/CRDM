source('./R/TidyRaster.R')
source('./crdm/R/PlotTheme.R')
library(magrittr)
library(ggplot2)

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) %>%
  sf::st_union()

parse_annual_data <- function(d, mon='*', lead_time=2) {
  
  d %>%
    list.files(full.names = T) %>%
    grep(mon, ., value = T) %>%
    lapply(function(x) {
      
      variable <- basename(x) %>%
        stringr::str_split('_') %>%
        unlist() %>%
        magrittr::extract(2) %>%
        stringr::str_replace('.tif', '')
      
      x %>%
        tidy_raster(stack = FALSE, band = lead_time) %>%
        dplyr::mutate(holdout = variable)
    }) %>%
    dplyr::bind_rows() %>% 
    tidyr::pivot_wider(names_from = holdout, values_from = val) %>% 
    tidyr::pivot_longer(
      cols = !c(x, y, None),
      names_to = 'holdout', 
      values_to = 'val'
    ) %>% 
    dplyr::mutate(diff = None - val)
}

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









ann_map_plot <- function(r) {
  
  dat <- tidy_raster(r, TRUE) %>% 
    tidyr::pivot_longer(
      dplyr::starts_with('lt_')
    ) %>% 
    dplyr::mutate(
      name = stringr::str_replace(name, 'lt_', '') %>% as.numeric()
    ) %>%
    dplyr::filter(name %in% c(2, 4, 6, 8, 10, 12))
  
  ggplot() + 
    geom_raster(aes(x=x, y=y, fill=value), data = dat) + 
    geom_sf(aes(), data = states, fill=NA) + 
    facet_wrap(~name) + 
    plot_theme() + 
    labs(x='', y='', fill='MSE\n(USDM Categories)') + 
    scale_fill_continuous(type = 'viridis')
}

ann_map_all <- function(ann_dir, lead_time) {
  
  dat <- parse_annual_data(ann_dir)
  
  filt <- dplyr::filter(dat, holdout != 'pr')
  ggplot() + 
    geom_raster(aes(x=x, y=y, fill=diff), data = filt) + 
    geom_sf(aes(), data = states, fill = NA) + 
    facet_wrap(~holdout) + 
    plot_theme() + 
    scale_fill_continuous(type = 'viridis') + 
    labs(x='', y='', fill='MSE\n(USDM Categories)') 
    
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

dat <- fst::read_fst(f) %>% tibble::as_tibble()


library(foreach)
library(doParallel)

cl <- makeCluster(12)
registerDoParallel(cl)
foreach(i = 1:length(unique(dat$holdout))) %dopar% {
  for (l in 1:12) {
    v = unique(dat$holdout)[i]
    print(v)
    print(l)
    mon_plot(dat, lead_time = l, variable = v)
  }
}

mon_plot_v2 <- function(f = './data/plot_data/monthly_holdouts.fst', lead_time=1, 
                     variable = 'pr') {
  
  lt <- glue::glue('lt_{lead_time}')
  
  dat <- fst::read_fst(f, columns = c('x', 'y', 'holdout', 'month')) %>%
    tibble::as_tibble() %>%
    dplyr::rename(val = !!lt)
  
  
  pal <- colorRampPalette(RColorBrewer::brewer.pal(10, 'Spectral'))
  
  out <- dat %>% 
    # dplyr::filter(holdout %in% c(!!variable, 'None')) %>%
    tidyr::pivot_wider(
      names_from = holdout, 
      values_from = val
    ) %>%
    tidyr::pivot_longer(
      cols = -c(x, y, month, None)
    ) %>%
    dplyr::mutate(
      diff = None - value,
      diff = ifelse(diff > 0, 0, diff),
      binned = dplyr::ntile(diff, 10),
      name = dplyr::recode(
        name,
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
      ),
      month = factor(month.abb[as.numeric(month)], levels = month.abb)
    ) %>%
    dplyr::filter(month == 'Jan')
  
  ggplot() + 
    geom_raster(aes(x=x, y=y, fill=binned), out) + 
    geom_sf(aes(), states, fill=NA, size = 0.5) + 
    facet_wrap(~name) + 
    plot_theme() + 
    scale_fill_gradientn(colors = pal(10)) + 
    labs(x='', y='', fill='MSE\n(USDM Categories)', 
         title = glue::glue('Increase in Model Error for Jan ({lead_time} Week Lead Time)')) +
    theme(axis.text.x = element_text(angle = 45))
  
  
}

ts_plot <- function(f = './data/plot_data/monthly_holdouts.fst', lead_time=1, 
                        variable = 'pr') {
  
  lt <- glue::glue('lt_{lead_time}')
  
  dat <- fst::read_fst(f, columns = c('x', 'y', lt, 'holdout', 'month')) %>%
    tibble::as_tibble() %>%
    dplyr::rename(val = !!lt) %>%
    tidyr::pivot_wider(
      names_from = holdout, 
      values_from = val
    ) %>%
    tidyr::pivot_longer(
      cols = -c(x, y, month, None)
    ) %>%
    dplyr::mutate(
      diff = None - value,
      diff = ifelse(diff > 0, 0, diff)
    )
  
  out <- dat %>%
    dplyr::group_by(month, name) %>%
    dplyr::summarise(diff = mean(diff)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(month = factor(month.abb[as.numeric(month)], levels = month.abb)) 
    
  ggplot(out, aes(x=month, y=diff, color=name)) + 
      geom_point() + 
      plot_theme()
}