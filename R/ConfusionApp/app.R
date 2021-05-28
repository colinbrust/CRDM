#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
source('/mnt/e/PycharmProjects/DroughtCast/R/PlotConfusion.R')
library(shiny)

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("DroughtCast Confusion Matrix"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
            sliderInput("lt",
                        "Lead Time:",
                        min = 1,
                        max = 12,
                        value = 5),
            
            selectInput("set", 
                        "Validation Set:",
                        c('Spatial Holdout'='test',
                          'Training Set'='train',
                          'Temporal Holdout'='val')),
            
            checkboxInput('rm', "Include 'No Drought' Category", TRUE)
        ),

        # Show a plot of the generated distribution
        mainPanel(
           plotOutput("confPlot")
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {

    output$distPlot <- renderPlot({
        plot_confusion(
            '/mnt/e/PycharmProjects/DroughtCast/data/plot_data/tables',
            input$lt,
            input$set, 
            input$rm
        ) 
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
