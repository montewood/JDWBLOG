library(shiny)

default_end <- Sys.Date() - 1
default_start <- default_end - 89

ui <- fluidPage(
  tags$script(src = "fetch-analytics.js"),
  tags$style(HTML("
    :root {
      color-scheme: light dark;
      font-family: system-ui, sans-serif;
    }
    body {
      margin: 0;
      background: Canvas;
      color: CanvasText;
    }
    .dashboard-shell {
      padding: 1rem;
    }
    .dashboard-header {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 1rem;
    }
    .dashboard-header h1 {
      margin: 0;
      font-size: 1.35rem;
    }
    .dashboard-header p {
      max-width: 44rem;
      margin: 0.35rem 0 0;
      color: GrayText;
      font-size: 0.82rem;
    }
    .dashboard-settings {
      min-height: 2.5rem;
      white-space: nowrap;
    }
    .dashboard-summary {
      margin: 1rem 0 0;
      padding: 0.8rem 1rem;
      border: 1px solid color-mix(in srgb, CanvasText 15%, transparent);
      border-radius: 0.7rem;
    }
    .dashboard-summary span {
      display: block;
      color: GrayText;
      font-size: 0.78rem;
    }
    .dashboard-summary strong {
      display: block;
      margin-top: 0.2rem;
      font-size: 1.6rem;
    }
  ")),
  div(
    class = "dashboard-shell",
    div(
      class = "dashboard-header",
      div(
        h1("JDW Blog R Dashboard"),
        p(
          "날짜별 고유 Active Users 추이입니다. ",
          "백필 전 구형 파일은 분 단위 관측 합계로 표시될 수 있습니다."
        )
      ),
      actionButton("settings_btn", "기간 설정", class = "dashboard-settings")
    ),
    div(
      class = "dashboard-summary",
      span("선택 기간 최근일 Active Users"),
      strong(textOutput("observation_total", inline = TRUE))
    ),
    p(textOutput("load_status")),
    plotOutput("visit_plot", height = "360px", width = "100%")
  )
)

server <- function(input, output, session) {
  selected_range <- reactiveVal(c(default_start, default_end))

  observeEvent(input$settings_btn, {
    current_range <- selected_range()
    showModal(modalDialog(
      title = "조회 기간",
      dateRangeInput(
        "date_range",
        "기간 설정",
        start = current_range[1],
        end = current_range[2],
        min = default_start,
        max = default_end
      ),
      easyClose = TRUE,
      footer = tagList(
        modalButton("취소"),
        actionButton("apply_settings", "적용")
      )
    ))
  })

  observeEvent(input$apply_settings, {
    req(input$date_range)
    selected_range(as.Date(input$date_range))
    removeModal()
  })

  all_analytics <- reactive({
    req(input$analytics_payload)
    payload <- input$analytics_payload

    data.frame(
      date = as.Date(
        as.character(unlist(payload[["dates"]], use.names = FALSE)),
        "%Y%m%d"
      ),
      active_users = as.numeric(
        unlist(payload[["values"]], use.names = FALSE)
      ),
      status = as.character(
        unlist(payload[["statuses"]], use.names = FALSE)
      ),
      stringsAsFactors = FALSE
    )
  })

  analytics_data <- reactive({
    date_range <- selected_range()
    data <- all_analytics()
    data[
      data$date >= date_range[1] &
        data$date <= date_range[2] &
        data$status == "ok",
      ,
      drop = FALSE
    ]
  })

  output$load_status <- renderText({
    if (is.null(input$analytics_payload)) {
      progress <- input$analytics_progress
      if (is.null(progress)) return("일별 데이터를 준비하는 중입니다.")
      return(sprintf(
        "일별 데이터 %d/%d 불러오는 중",
        progress$completed,
        progress$total
      ))
    }

    data <- all_analytics()
    sprintf(
      "%d일 수집 완료 · %d일 데이터 없음 또는 요청 실패",
      sum(data$status == "ok"),
      sum(data$status != "ok")
    )
  })

  output$observation_total <- renderText({
    data <- analytics_data()
    if (nrow(data) == 0) return("데이터 없음")
    format(tail(data$active_users, 1), big.mark = ",")
  })

  output$visit_plot <- renderPlot({
    data <- analytics_data()
    validate(need(nrow(data) > 0, "선택한 기간에 표시할 데이터가 없습니다."))

    bar_positions <- barplot(
      height = data$active_users,
      names.arg = FALSE,
      col = "#4f83cc",
      border = NA,
      main = "Daily Active Users",
      xlab = "Date",
      ylab = "Active Users",
      las = 1
    )

    label_indices <- unique(round(seq(1, nrow(data), length.out = min(8, nrow(data)))))
    axis(
      side = 1,
      at = bar_positions[label_indices],
      labels = format(data$date[label_indices], "%m-%d"),
      las = 2,
      cex.axis = 0.75
    )
  })
}

shinyApp(ui, server)
