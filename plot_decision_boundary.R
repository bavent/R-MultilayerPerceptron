###################################################
## Decision boundary plotting function definition
###################################################

# Plots a decision boundary for 2-D data across the grid region from (x_min, y_min) to
# (x_max, y_max) with an optionally-specified resolution
# Overlays input points from X colored by their predictions from Y_predicted
plot_decision_boundary = function(x_min, y_min, x_max, y_max, X, Y_predicted, resolution=0.01) {
  library(ggplot2)
  # Create grid (of specified resolution) of points to create predictions for (across specified range)
  grid = as.matrix(expand.grid(seq(x_min, x_max, by = resolution), seq(y_min, y_max, by = resolution)))
  # Feed each point from the grid into the neural network to generate predictions
  Z = predict(grid, trained_weights)
  
  # Format original data to be overlaid on decision boundary
  data = data.frame(X, Y, Y_predicted)
  colnames(data) = c("x", "y", "label")
  
  # Run plotting code
  ggplot()+
    geom_tile(aes(x = grid[,1],y = grid[,2],fill=as.character(Z)), alpha = 0.3, show.legend = F)+ 
    geom_point(data = data, aes(x=x, y=y, color = as.character(label)), size = 2) + theme_bw(base_size = 15) +
    ggtitle('Neural Network Decision Boundary') +
    coord_fixed(ratio = 0.8) + 
    theme(axis.ticks=element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
          axis.text=element_blank(), axis.title=element_blank(), legend.position = 'none')
}