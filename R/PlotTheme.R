plot_theme <- function() {
  
  list(
    theme_minimal(),
    theme(plot.title = element_text(hjust = 0.5, colour = "gray15", face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, colour = "gray20", face = "bold"),
          axis.title.x =  element_text(colour = "gray26", face = "bold", size=11),
          axis.title.y =  element_text(colour = "gray26", face = "bold", size=11),
          axis.text.x =  element_text(colour = "gray26", size=10),
          axis.text.y =  element_text(colour = "gray26", size=10),
          legend.title =  element_text(hjust = 0.5, colour="gray15", face = "bold",
                                       size = 10),
          legend.text =   element_text(colour="gray26", face = "bold", size = 10),
          strip.text =    element_text(family = "sans", size = 11, face = "bold", hjust = 0.5,
                                       vjust = 1)
    )
  )
}