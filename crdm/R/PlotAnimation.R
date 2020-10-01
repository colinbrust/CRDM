PlotAnimation <- function(
  f_dir = '~/projects/CRDM/figures/',
  out_name = '~/projects/CRDM/figures/preds_7mons.gif') {
  
  f_dir %>%
    list.files(full.names = T, pattern = '.png') %>%
    purrr::map(magick::image_read) %>%
    magick::image_join() %>%
    magick::image_animate(fps = 10) %>%
    magick::image_write(out_name)
}
