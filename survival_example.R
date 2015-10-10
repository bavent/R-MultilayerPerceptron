###################################################
## Import and format "Haberman's Survival Data Set"
###################################################

# "Haberman's Survival Data Set": https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival
# Abstract: Dataset contains cases from study conducted on the survival of patients who had undergone surgery for breast cancer
# Goal: Predict whether the patient will surive 5 years or longer following surgery
survival_data = data.matrix(read.csv("haberman.data", header=F))
X = scale(survival_data[,-4])
Y = survival_data[,4]-1

m = dim(X)[1]
n = dim(X)[2]

# Create training (90%) and test sets (10%)
samples = sample(m, .9*m)
X_train = X[samples,]
Y_train = matrix(Y[samples], ncol=1)
X_test = X[-samples,]
Y_test = matrix(Y[-samples], ncol=1)


###################################################
## Run network with "Haberman's Survival Data Set"
###################################################

# number of nodes in each layer [do NOT include bias nodes here]
net_structure = c(n, 3, 1)
# consider this example net_structure: c(10, 6, 4, 1)
# this "structure" has:
# - 10 input nodes
# - 6 nodes in hidden layer #1
# - 4 nodes in hidden layer #2
# - 1 node in output layer

# learning rate for gradient descent
learning_rate = .3

# regularization parameter for gradient descent
regularization_factor = .1

# maximum number of epochs (passes through the data set)
max_epochs = 500

## Train network on training data
trained_weights = train(X_train,
                        Y_train,
                        net_structure,
                        learning_rate,
                        regularization_factor,
                        max_epochs)

# trained_weights = gradient_descent(X_train, Y_train)
Y_predicted = predict(X_test, trained_weights)
acc = compute_accuracy(Y_predicted, Y_test)
cat("Haberman's survival network test accuracy rate: ", 100*acc, "%", sep="")
