# clear workspace
rm(list = ls())

library(ggplot2)
library(cowplot)
library(dplyr)
library(PupillometryR)


scores <- read.csv("results/all_method_scores_by_run.csv")
task_id_lists <- unique(scores$task_id)
task_id_lists

methods <- unique(scores$method)
methods

SHAPE <- c(21,24,22,25)
cb_palette <- c('#D81B60','#1E88E5','#FFC107','#004D40')

# declare theme
p_theme <- theme(
  plot.title = element_text( face = "bold", size = 22, hjust=0.5),
  panel.border = element_blank(),
  panel.grid.minor = element_blank(),
  legend.title=element_text(size=22),
  legend.text=element_text(size=23),
  axis.title = element_text(size=23),
  axis.text = element_text(size=19),
  # legend.position="bottom",
  legend.position = "none",
  panel.background = element_rect(fill = "#f1f2f5",
                                  colour = "white",
                                  linewidth = 0.5, linetype = "solid")
)

plots <- list() # empty list to hold plots

for (i in seq_along(task_id_lists)) {
    task_data <- filter(scores, task_id == task_id_lists[1])
    plot <- ggplot(task_data, aes(x = method, y = test_score, color = method, fill = method, shape = method)) +
    geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
    geom_boxplot(color = 'black', width = .08, outlier.shape = NA, alpha = 0.0, linewidth = 0.8, position = position_nudge(x = .15, y = 0)) +
    geom_point(position = position_jitter(width = .015, height = .0001), size = 2.0, alpha = 1.0) +
    scale_y_continuous(
        name = "Accuracy %",
        breaks = c(.74, .78, .82, .86),
        labels = scales::percent
    ) +
    scale_x_discrete(name = "Method") +
    scale_shape_manual(values = SHAPE) +
    scale_colour_manual(values = cb_palette) +
    scale_fill_manual(values = cb_palette) +
    ggtitle(paste('Task', task_id_lists[i])) +
    p_theme

    plots[[i]] <- plot
}

# extract shared legend from first plot
legend <- get_legend(
    plots[[1]] + theme(legend.position = "bottom")
)

# remove legends from individual plots
# plots_nolegend <- lapply(plots, function(p) p + theme(legend.position = "none"))

final_plot <- plot_grid(plotlist = plots, ncol = 2)
ggsave("all_tasks.pdf", final_plot, width=12, height=30)

# task_1 <- filter(scores, task_id == task_id_lists[1]) %>%
#   ggplot(., aes(x = method, y = test_score, color = method, fill = method, shape = method)) +
#   geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
#   geom_boxplot(color = 'black', width = .08, outlier.shape = NA, alpha = 0.0, linewidth = 0.8, position = position_nudge(x = .15, y = 0)) +
#   geom_point(position = position_jitter(width = .015, height = .0001), size = 2.0, alpha = 1.0) +
#   scale_y_continuous(
#     name="Accuracy %",
#     breaks=c(.74,.78,.82,.86),
#     labels = scales::percent

#   ) +
#   scale_x_discrete(
#     name="Method"
#   )+
#   scale_shape_manual(values=SHAPE)+
#   scale_colour_manual(values = cb_palette, ) +
#   scale_fill_manual(values = cb_palette) +
#   ggtitle('Random Forest Test Accuracy')+
#   p_theme


# save_plot(
#   paste(filename ="accuracy.pdf"),
#   task_1,
#   base_width=10,
#   base_height=17
# )