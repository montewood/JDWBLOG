library(shiny)
library(bslib)
library(bsicons)
library(jsonlite)
library(dplyr)
library(stringr)
library(purrr)
library(glue)
library(ggplot2)
library(shinycssloaders)
library(shinyjs)
library(scales)

ui <- fluidPage(
  useShinyjs(),  # 🔹 shinyjs 활성화
  titlePanel(HTML("<b>About This blog</b>")),

  # 🔹 설정 아이콘을 화면 우상단에 고정
  absolutePanel(
    top = 5, right = 20, fixed = TRUE,
    actionButton("settings_btn", label = bsicons::bs_icon("gear-fill"),
                 style = "font-size: 64px; background: none; border: none; cursor: pointer; color: #555;")

  ),

  # 메인 레이아웃
  fluidRow(
    column(12,
           card(
             height = "70vh",
             card_body(
               style = "padding-top: 4vh; padding-left: 2vh; padding-right: 2vh;",
               withSpinner(
                 plotOutput("visit_plot", height = "64vh", width = "100%"),
                 type = 3,
                 color = "#F5F5F5",
                 color.background = "#F5F5F5"
               )
             )
           )
    ),
    column(12,
           value_box(
             title = "총 방문자 수",
             value = uiOutput("row_count_ui"),
             showcase = bsicons::bs_icon("people-fill"),
             theme = "custom",
             style = "background-color: #A67B5B; color: white;",
             height = "12vh"
           )
    )
  )
)

server <- function(input, output, session) {

  # 설정 버튼 클릭 시 팝업 UI 표시
  observeEvent(input$settings_btn, {
    showModal(modalDialog(
      title = h2(HTML("<b>방문자 수</b>")),
      dateRangeInput("date_range", "조회 기간 설정",
                     start = "2024-01-01",
                     end = "2024-02-28"),
      easyClose = TRUE,
      footer = tagList(
        modalButton("취소"),
        actionButton("apply_settings", "적용")
      )
    ))
  })

  # "적용" 버튼 클릭 시 팝업 닫기
  observeEvent(input$apply_settings, {
    removeModal()
  })

  # 데이터 로딩 상태 관리
  data_reactive <- reactiveVal(NULL)
  loading_state <- reactiveVal(FALSE)

  # 🔹 "방문자 수 조회" 버튼 클릭 시 데이터 로딩 (팝업 "적용" 버튼도 트리거)
  observeEvent({ input$visit_btn; input$apply_settings }, {
    req(input$date_range)

    loading_state(TRUE)
    showNotification("Loading Data...", type = "message", duration = 3)

    # 선택한 날짜 범위의 데이터를 로드
    date_seq <- seq(from = input$date_range[1], to = input$date_range[2], by = "1 days")

    result_data <- map_df(date_seq, function(selected_date) {
      selected_date_string <- format(selected_date, "%Y%m%d")
      url <- paste0("https://raw.githubusercontent.com/montewood/gh-action/refs/heads/main/output/GA-",
                    selected_date_string, ".json")

      tryCatch({
        fromJSON(url) %>% fromJSON() %>% as_tibble()
      }, error = function(e) {
        NULL
      })
    })

    data_reactive(result_data)
    loading_state(FALSE)
    showNotification("Data Loading Complete", type = "message", duration = 2)
  })

  # 총 방문자 수 UI 출력
  output$row_count_ui <- renderUI({
    if (loading_state()) {
      span("Loading...", style = "font-size: 20px; font-weight: bold; color: gray;")
    } else {
      df <- data_reactive()
      if (is.null(df) || nrow(df) == 0) {
        span("No Data", style = "font-size: 20px; font-weight: bold; color: gray;")
      } else {
        span(paste0(sum(df$activeUsers, na.rm = TRUE), " 명"),
             style = "font-size: 30px; font-weight: bold; color: black;")
      }
    }
  })

  # 방문자 수 변화 차트 출력
  output$visit_plot <- renderPlot({
    req(data_reactive())
    if (loading_state()) return(NULL)

    df <- data_reactive()

    agg_df <- df %>%
      mutate(parsed_year_month = str_sub(parsed_year_month, 1, 7)) %>%
      group_by(parsed_year_month) %>%
      summarise(activeUsers = sum(activeUsers, na.rm = TRUE))

    ggplot(agg_df, aes(x = parsed_year_month, y = activeUsers)) +
      geom_col() +
      geom_text(aes(label = activeUsers), vjust = -0.5, color = "black") +
      scale_y_continuous(expand = c(0, 10)) +
      labs(title = "JDW BLOG Monthly Visitor Chart",
           subtitle = glue("{input$date_range[1]} ~ {input$date_range[2]}"),
           x = "Year-Months", y = "Total Visitors") +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 18, face = "bold", color = "black"),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14)
      )
  })
}

shinyApp(ui, server)
