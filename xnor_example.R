###################################################
## Import and format XNOR data
###################################################

# XNOR data:
X = matrix(c(0,0,0,1,1,0,1,1), ncol=2, byrow=T)
Y = matrix(c(1,0,0,1), ncol=1)

m = dim(X)[1]
n = dim(X)[2]


###################################################
## Run network with XNOR data
###################################################

net_structure = c(2, 2, 1) # number of nodes in each layer (not including bias node)
learning_rate = .3 # learning rate for gradient descent
regularization_factor = 0 # regularization parameter for gradient descent
max_epochs = 15000 # maximum number of epochs (passes through the data set)

trained_weights = train(X, Y, net_structure, learning_rate, regularization_factor, max_epochs)

Y_predicted = predict(X, trained_weights)
acc = compute_accuracy(Y_predicted, Y)
cat("XNOR network test accuracy rate: ", 100*acc, "%", sep="")
