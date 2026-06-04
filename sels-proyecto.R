sels <-  tibble::tribble(~sel, ~prop,
                         01, 33.0,
                         02,  3.6,
                         03,  7.6,
                         04, 13.3,
                         05,  5.9,
                         06,  6.7,
                         07, 10.2,
                         08,  9.8,
                         09, 12.9,
                         10,  0.9,
                         11,  0.3,
                         12,  0.2)

sels$color <- c(1,2,2,2,2,2,2,2,2,3,3,3)
library(ggplot2)

ggplot(sels, aes(x = reorder(sel, -prop), y = prop, group = 1, color = color)) +
  geom_point(show.legend = F) +
  geom_line(show.legend = F) +
  scale_color_viridis_c(begin = 0.0, end = 0.8, option = "D")
