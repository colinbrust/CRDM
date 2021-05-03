target_dir = './data/tif_targets'
f = './data/models/ensemble/model6/preds/20030805_preds_None.tif'


get_baseline <- function(f, target_dir) {
  
  day <- f %>% 
    basename() %>% 
    stringr::str_split('_') %>% 
    unlist() %>% 
    head(1) %>% 
    lubridate::as_date()
}