library(ggplot2)
library(magrittr)
source('./crdm/R/PlotTheme.R')

f <- './data/models/model3/preds_1/20170704_preds_None.tif'
target_dir = './data/tif_targets'
states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) 

get_date_string <- function(f) {
  f %>% 
    basename() %>% 
    stringr::str_split('_') %>% 
    unlist() %>% 
    head(1)
}

get_baseline <- function(f, target_dir) {
  
  f %>% 
    get_date_string() %>%
    lubridate::as_date() %>% 
    {. - 7} %>%
    stringr::str_replace_all('-', '') %>%
    list.files(target_dir, full.names = T, pattern = .) %>%
    raster::raster()

}

get_targets <- function(target_dir, f, states) {
  
  day <- lubridate::as_date(get_date_string(f))
  
  dates <- seq(day, day + lubridate::weeks(11), by = 'weeks') %>% 
    stringr::str_replace_all('-', '') %>%
    paste(collapse = '|')
  
  target_dir %>%
    list.files(full.names = T, pattern=dates) %>%
    raster::stack()
  
}

calc_error <- function(f, target_dir, states, out_dir) {
  

  targets <- get_targets(target_dir, f, states)
  preds <- raster::stack(f) %>% raster::mask(states)
  
  out_name <- file.path(out_dir, basename(f))
  print(out_name)
  preds <- (preds - targets) ** 2
  
  raster::writeRaster(preds, out_name, overwrite = T)
}

list.files('./data/models/model3/err_1', full.names = T) %>% 
  rev() %>% 
  parallel::mclapply(function(f) {
    calc_error(f, './data/tif_targets', states, './data/models/model3/err_1')
  })



tidy_map <- function(r, name) {
  
  raster::rasterToPoints(r) %>%
    tibble::as_tibble() %>%
    `names<-`(c('x', 'y', paste('lt', 1:12, sep='_'))) %>%
    tidyr::pivot_longer(
      cols = dplyr::starts_with('lt'),
      values_to = name,
      names_to = 'lead_time')
  
}

rmse = function(m, o) {
  sqrt(mean((m - o)^2, na.rm = T))
}


rsquared = function(m, o) {
  cor(m, o, use='complete.obs') ^ 2
}

calc_reduce_error <- function(preds, targets, baseline, states) {
  
  pred_df <- tidy_map(preds, 'preds')
  targ_df <- tidy_map(targets, 'targets')
  base_df <- tidy_map(baseline, 'base')
  
  dplyr::left_join(pred_df, targ_df, by=c('x', 'y', 'lead_time')) %>%
    dplyr::left_join(base_df, by=c('x', 'y', 'lead_time')) %>%
    dplyr::group_by(lead_time) %>%
    dplyr::summarise(
      pred_rmse = rmse(preds, targets),
      base_rmse = rmse(base, targets),
      pred_r = rsquared(preds, targets),
      base_r = rsquared(base, targets)
    )
  
}

read_metadata <- function(f) {
  
  read_pickle(f) %>% 
    lapply(function(x) paste(x, collapse=', ')) %>%
    tibble::as_tibble() %>% 
    dplyr::mutate(model_class = basename(dirname(f)), 
                  model_id = basename(f) %>% 
                    stringr::str_split('_') %>% 
                    unlist() %>% 
                    magrittr::extract(2)) 
}

states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) 

target_dir = './data/tif_targets'
pred_dir = './data/models/model0'
model_number = 0

expand.grid(c('./data/models/model0', './data/models/model1', 
              './data/models/model2', './data/models/model3',
              './data/models/model4'),
            c(0, 1), stringsAsFactors = FALSE) -> a


models <- a$Var1
num <- a$Var2

calc_all_error <- function(pred_dir, model_num, target_dir, states, holdout='None') {
  6
  
  print(basename(pred_dir))
  list.files(file.path(pred_dir, paste0('preds_', model_num)),
             pattern = holdout, full.names = T) %>%
    lapply(function(f) {
      
      print(f)
      day <- get_date_string(f)
      
      preds  <- raster::stack(f) %>% raster::mask(states)
      targets <- get_targets(target_dir, f, states) 
      baseline <- raster::stack(replicate(12, get_baseline(f, target_dir))) 
      
      df <- calc_reduce_error(preds, targets, baseline, states) %>%
        dplyr::mutate(day = day)
      
    }) %>% 
    dplyr::bind_rows() %>%
    dplyr::mutate(model_num = model_num,
                  dir_num = basename(pred_dir))
  
}

parallel::mcmapply(calc_all_error, models, num, MoreArgs=list(target_dir = target_dir, states = states), mc.cores = 10) -> a

a %>%
  dplyr::bind_rows() %>% 
  dplyr::mutate(lead_time = as.numeric(stringr::str_replace(lead_time, 'lt_', '')),
                day = lubridate::as_date(day),
                model_num = factor(model_num),
                lead_time = factor(lead_time)) %>%
  tidyr::pivot_longer(
    cols = c(pred_rmse, base_rmse, pred_r, base_r) 
  ) %>%
  dplyr::filter(name %in% c('pred_rmse')) %>%
  dplyr::group_by(lead_time) %>%
  dplyr::slice(which.min(value))
  ggplot(aes(x=day, y=value, color=lead_time)) + 
  geom_line() + 
  facet_wrap(model_num~dir_num)

  
  
x <- runif(n = 100, min=0, max=1)
  true <- runif(n=100, min=0, max=1)
  
  rmse = function(m, o) {
    sqrt(mean((m - o)^2, na.rm = T))
  }
  rmse(preds, true)  == sqrt(mean(out ** 2))

  mse = function(m, o) {
    mean((m - o)^2, na.rm = TRUE)
  }
  
  out = rep(0, 100)
  for (i in 1:length(x)) {
      out[i] = mse(x[i], true[i])  
  }
  

  
  mse(x, true) - mean(c(mean(out[1:50]), mean(out[51:100])))
  