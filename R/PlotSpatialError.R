source('./R/TidyRaster.R')
source('./crdm/R/PlotTheme.R')


parse_data <- function(d, mon='*') {
  
  d %>%
    list.files(full.names = T) %>%
    grep(mon, ., value = T) %>%
    lapply(function(x) {
      
      variable <- basename(x) %>%
        stringr::str_split('_') %>%
        unlist() %>%
        tail(1) %>%
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
  
  dat <- parse_data(ann_dir)
  
  filt <- dplyr::filter(dat, holdout != 'pr')
  ggplot() + 
    geom_raster(aes(x=x, y=y, fill=diff), data = filt) + 
    geom_sf(aes(), data = states, fill = NA) + 
    facet_wrap(~holdout) + 
    plot_theme() + 
    scale_fill_continuous(type = 'viridis') + 
    labs(x='', y='', fill='MSE\n(USDM Categories)') 
    
}

mon_plot <- function(mon_dir, mon) {
  
  dat <- mon_dir %>%
    parse_data(mon = '*_05.tif')
}

ann_map_plot("./data/err_maps/annual/err_pr.tif")
