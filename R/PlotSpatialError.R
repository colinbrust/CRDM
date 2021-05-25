source('./R/TidyRaster.R')
source('./crdm/R/PlotTheme.R')


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

mon_plot <- function(f = './data/plot_data/monthly_holdouts.fst', lead_time=1, 
                     variable = 'pr') {
  
  lt <- glue::glue('lt_{lead_time}')
  
  dat <- fst::read_fst(f, columns = c('x', 'y', lt, 'holdout', 'month')) %>%
    tibble::as_tibble() %>%
    dplyr::rename(val = !!lt) 
  
    
  
  out <- dat %>% 
    dplyr::filter(holdout %in% c(!!variable, 'None')) %>%
    tidyr::pivot_wider(
      names_from = holdout, 
      values_from = val
    ) %>%
    dplyr::mutate(diff = None - !!rlang::sym(variable),
                  month = factor(month.abb[as.numeric(month)], levels = month.abb),
                  diff = ifelse(diff > 0, 0, diff))
  
  colorRampPalette(RColorBrewer::brewer.pal(10, 'Reds')) -> pal
  ggplot() + 
    geom_raster(aes(x=x, y=y, fill=diff), out) + 
    geom_sf(aes(), states, fill=NA) + 
    facet_wrap(~month) + 
    plot_theme() + 
    scale_fill_gradientn(colors = rev(pal(10)), limits = c(-2.5, 0)) + 
    labs(x='', y='', fill='MSE\n(USDM Categories)', title = 'Increase in Model Error When PPT Removed (4 Week Lead Time)') +
    theme(axis.text.x = element_text(angle = 45))
    
}

dat %>% fst::write_fst('./data/err_maps/monthly_holdouts.fst')
