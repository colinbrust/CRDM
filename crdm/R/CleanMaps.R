states <- urbnmapr::get_urbn_map(sf = TRUE) %>%
  dplyr::filter(state_abbv != 'AK', state_abbv != 'HI') %>%
  sf::st_transform(6933) 

tidy_map <- function(f, states) {
  
  f %>%
    raster::stack() %>% 
    raster::mask(states) %>% 
    raster::rasterToPoints() %>% 
    tibble::as_tibble() %>%
    `names<-`(c('x', 'y', paste('lt', 1:12, sep='_'))) %>%
    tidyr::pivot_longer(
      dplyr::starts_with('lt'),
      names_to = 'lead_time',
      values_to = 'val'
    ) %>% 
    dplyr::mutate(val = round(val),
                  val = dplyr::recode(
                    val,
                    `0` = 'No Drought',
                    `1` = 'D0',
                    `2` = 'D1',
                    `3` = 'D2',
                    `4` = 'D3',
                    `5` = 'D4'))
  
}