############################################################
# KENYA EV STOCK FORECAST
# Auto-ARIMA Baseline + Prediction Intervals
# Policy-informed Conservative and Accelerated Scenarios
# Ribbon extension included so intervals visually reach 2030
############################################################

library(forecast)
library(ggplot2)
library(dplyr)
library(scales)

############################################################
# 1. OBSERVED DATA
############################################################

year <- 2019:2025

ev_stock <- c(
  194,
  300,
  584,
  1059,
  3753,
  5294,
  9200
)

obs_df <- data.frame(
  Year = year,
  EV_Stock = ev_stock,
  Type = "Observed EV stock"
)

last_obs <- tail(ev_stock, 1)

############################################################
# 2. AUTO-ARIMA BASELINE FORECAST
############################################################

ev_ts <- ts(
  ev_stock,
  start = 2019,
  frequency = 1
)

fit <- auto.arima(
  ev_ts,
  seasonal = FALSE,
  stepwise = FALSE,
  approximation = FALSE
)

summary(fit)

fc <- forecast(
  fit,
  h = 5,
  level = c(50, 95)
)

future_years <- 2026:2030

############################################################
# 3. EXTRACT FORECAST INTERVALS
############################################################

Q25 <- as.numeric(fc$lower[, 1])
Q75 <- as.numeric(fc$upper[, 1])

Q05 <- as.numeric(fc$lower[, 2])
Q95 <- as.numeric(fc$upper[, 2])

Q50 <- as.numeric(fc$mean)

Q05 <- pmax(Q05, last_obs)
Q25 <- pmax(Q25, last_obs)
Q50 <- pmax(Q50, last_obs)
Q75 <- pmax(Q75, last_obs)
Q95 <- pmax(Q95, last_obs)

############################################################
# 4. POLICY-INFORMED SCENARIOS
############################################################

conservative <- c(
  10100,
  11000,
  12000,
  13000,
  14000
)

accelerated <- c(
  13000,
  17000,
  23000,
  29000,
  36000
)

############################################################
# 5. FORECAST DATAFRAME, ANCHORED AT 2025
############################################################

forecast_df <- data.frame(
  Year = c(2025, future_years),
  Q05 = c(last_obs, Q05),
  Q25 = c(last_obs, Q25),
  Q50 = c(last_obs, Q50),
  Q75 = c(last_obs, Q75),
  Q95 = c(last_obs, Q95),
  Conservative = c(last_obs, conservative),
  Accelerated = c(last_obs, accelerated)
)

# Extend ribbons and forecast lines slightly beyond 2030
forecast_df_plot <- rbind(
  forecast_df,
  forecast_df[nrow(forecast_df), ]
)

forecast_df_plot$Year[nrow(forecast_df_plot)] <- 2030.15

############################################################
# 6. PLOT
############################################################

p <- ggplot() +

  geom_ribbon(
    data = forecast_df_plot,
    aes(
      x = Year,
      ymin = Q05,
      ymax = Q95,
      fill = "95% prediction interval"
    ),
    alpha = 0.25
  ) +

  geom_ribbon(
    data = forecast_df_plot,
    aes(
      x = Year,
      ymin = Q25,
      ymax = Q75,
      fill = "50% prediction interval"
    ),
    alpha = 0.45
  ) +

  geom_line(
    data = obs_df,
    aes(
      x = Year,
      y = EV_Stock,
      colour = Type
    ),
    linewidth = 1.6
  ) +

  geom_point(
    data = obs_df,
    aes(
      x = Year,
      y = EV_Stock,
      colour = Type
    ),
    size = 3.6
  ) +

  geom_line(
    data = forecast_df_plot,
    aes(
      x = Year,
      y = Q50,
      colour = "Auto-ARIMA baseline forecast"
    ),
    linewidth = 1.8,
    linetype = "dashed"
  ) +

  geom_point(
    data = forecast_df,
    aes(
      x = Year,
      y = Q50,
      colour = "Auto-ARIMA baseline forecast"
    ),
    size = 3
  ) +

  geom_line(
    data = forecast_df_plot,
    aes(
      x = Year,
      y = Conservative,
      colour = "Conservative scenario"
    ),
    linewidth = 1.4,
    linetype = "dotted"
  ) +

  geom_line(
    data = forecast_df_plot,
    aes(
      x = Year,
      y = Accelerated,
      colour = "Accelerated scenario"
    ),
    linewidth = 1.5,
    linetype = "longdash"
  ) +

  geom_vline(
    xintercept = 2025,
    linetype = "dashed",
    linewidth = 0.9
  ) +

  annotate(
    "text",
    x = 2022,
    y = max(forecast_df_plot$Accelerated) * 0.86,
    label = "Observed evidence window",
    fontface = "bold",
    size = 5
  ) +

  annotate(
    "text",
    x = 2028,
    y = max(forecast_df_plot$Accelerated) * 0.86,
    label = "Forecast scenario window",
    fontface = "bold",
    size = 5
  ) +

  scale_fill_manual(
    values = c(
      "95% prediction interval" = "skyblue",
      "50% prediction interval" = "dodgerblue"
    )
  ) +

  scale_colour_manual(
    values = c(
      "Observed EV stock" = "black",
      "Auto-ARIMA baseline forecast" = "blue",
      "Conservative scenario" = "firebrick",
      "Accelerated scenario" = "darkgreen"
    )
  ) +

  scale_x_continuous(
    breaks = 2019:2030,
    limits = c(2019, 2030)
  ) +

  scale_y_continuous(labels = comma) +
  coord_cartesian(
    ylim = c(0, max(forecast_df_plot$Q95, forecast_df_plot$Accelerated) * 1.10)
  ) +

  labs(
    title = "Kenya Electric Vehicle Stock Forecast (2019–2030)",
    subtitle = "Auto-ARIMA baseline with forecast uncertainty and policy-informed adoption scenarios",
    x = "Year",
    y = "Total Registered EV Stock",
    colour = NULL,
    fill = NULL,
    caption =
      "Baseline: Auto-ARIMA median forecast. Bands: 50% and 95% prediction intervals. Scenarios: policy-informed conservative and accelerated pathways."
  ) +

  theme_bw(base_size = 14) +

  theme(
    plot.title = element_text(face = "bold", size = 20),
    plot.subtitle = element_text(size = 13),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom",
    legend.text = element_text(size = 10),
    panel.grid.minor = element_blank()
  )

print(p)

############################################################
# 7. FORECAST TABLE
############################################################

forecast_table <- forecast_df %>%
  filter(Year >= 2026) %>%
  mutate(across(where(is.numeric), round))

print(forecast_table)

write.csv(
  forecast_table,
  "Kenya_EV_Final_Forecast_Table.csv",
  row.names = FALSE
)

ggsave(
  "Kenya_EV_Final_Forecast.png",
  p,
  width = 12,
  height = 7,
  dpi = 300
)
