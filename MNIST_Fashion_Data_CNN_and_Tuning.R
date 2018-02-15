##################### R INTERFACE TO KERAS BEGIN #################

# Keras is a high-level neural networks API developed with a focus 
# on enabling fast experimentation. Being able to go from idea to 
# result with the least possible delay is key to doing good research. 
# Keras has the following key features:

#################### INSTALLATION ########################

# First, install the keras R package from GitHub as follows:
devtools::install_github("rstudio/keras")

# The Keras R interface uses the TensorFlow backend engine by default. 
# To install both the core Keras library as well as the 
# TensorFlow backend use the install_keras() function:
library(keras)
install_keras()

################### MNIST Example #########################

# PREPARING THE DATA

# The MNIST dataset is included with Keras and can be 
# accessed using the dataset_mnist() function. Here we 
# load the dataset then create variables for our test 
# and training data:

library(keras)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Fashion data
mnist <- dataset_fashion_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# The x data is a 3-d array (images,width,height) of grayscale 
# values . To prepare the data for training we convert the 
# 3-d arrays into matrices by reshaping width and height 
# into a single dimension (28x28 images are flattened 
# into length 784 vectors). Then, we convert the grayscale 
# values from integers ranging between 0 to 255 into 
# floating point values ranging between 0 and 1:

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784)) 
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

# Note that we use the array_reshape() function rather 
# than the dim<-() function to reshape the array. 
# This is so that the data is re-interpreted using 
# row-major semantics (as opposed to R’s default 
# column-major semantics), which is in turn compatible 
# with the way that the numerical libraries called by 
# Keras interpret array dimensions.

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

################### DEFINING THE MODEL ###################

# The core data structure of Keras is a model, a way to 
# organize layers. The simplest type of model is the 
# Sequential model, a linear stack of layers.

# We begin by creating a sequential model and then 
# adding layers using the pipe (%>%) operator:

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = 'softmax')

# The input_shape argument to the first layer specifies 
# the shape of the input data (a length 784 numeric 
# vector representing a grayscale image). The final 
# layer outputs a length 10 numeric vector (probabilities 
# for each digit) using a softmax activation function.

summary(model)

# Next, compile the model with appropriate loss 
# function, optimizer, and metrics:

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

####################### TRAINING AND EVALUATION ####################

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
); plot(history)

# Evaluate the model’s performance on the test data:
model %>% evaluate(x_test, y_test)

# Generate predictions on new data:
model %>% predict_classes(x_test)

# Keras provides a vocabulary for building deep learning 
# models that is simple, elegant, and intuitive. Building a 
# question answering system, an image classification model, a 
# neural Turing machine, or any other model is just as straightforward.


####################### TUNING NEURAL NETWORK #############################

keras.network <- function(
  layer_1 = 128,
  layer_2 = 256,
  layer_3 = 128
){
  
  ################### DEFINING THE MODEL ###################
  
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = layer_1, activation = 'relu', input_shape = c(784)) %>% 
    layer_dropout(rate = 0.4) %>% 
    layer_dense(units = layer_2, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = layer_3, activation = 'relu') %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 10, activation = 'softmax')
  
  # The input_shape argument to the first layer specifies 
  # the shape of the input data (a length 784 numeric 
  # vector representing a grayscale image). The final 
  # layer outputs a length 10 numeric vector (probabilities 
  # for each digit) using a softmax activation function.
  
  summary(model)
  
  # Next, compile the model with appropriate loss 
  # function, optimizer, and metrics:
  
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )
  
  ####################### TRAINING AND EVALUATION ####################
  
  history <- model %>% fit(
    x_train, y_train, 
    epochs = 30, batch_size = 128, 
    validation_split = 0.2
  ); plot(history)
  
  # Evaluate the model’s performance on the test data:
  model %>% evaluate(x_test, y_test)
} # End of function

# Test
keras.network()

# Tuning Keras Network
tuning.keras.network <- function(
  default.layer_1 = 128,
  default.layer_2 = 256,
  default.layer_3 = 128
){
    
    # Default:
    #default.a1 = 128
    #default.a2 = 64
    #default.a3 = 2
  
    
    # Tune the number of hidden neurons in the 1st hidden layer:
    tune.a1 <- NULL
    tune.a1.interval <- seq(128,128*2,128)
    for (i in c(tune.a1.interval)) {
      tune.a1 <- cbind(tune.a1, keras.network(
        layer_1 = i,
        layer_2 = default.layer_2,
        layer_3 = default.layer_3
        )$acc
      )
      print(c("Finished tuning a1 with rounds", i))
    }; tune.a1; tune.a1.interval
    tune.a1.mat <- data.frame(rbind(
      tune.a1.interval, tune.a1
    )); t(tune.a1.mat)
    print("Finished tuning the first hidden layer!")
    inds.a1 <- which(tune.a1.mat == max(tune.a1), arr.ind = TRUE)
    i <- tune.a1.interval[inds.a1[1,2]]
    
    # Tune the number of hidden neurons in the 2nd hidden layer:
    tune.a2 <- NULL
    tune.a2.interval <- seq(64,64*2,64)
    for (j in c(tune.a2.interval)) {
      tune.a2 <- cbind(tune.a2, keras.network(
        layer_1 = i,
        layer_2 = j,
        layer_3 = default.layer_3
      )$acc
      )
      print(c("Finished tuning a2 with rounds", j))
    }; tune.a2; tune.a2.interval
    tune.a2.mat <- data.frame(rbind(
      tune.a2.interval, tune.a2
    )); t(tune.a2.mat)
    print("Finished tuning the second hidden layer!")
    inds.a2 <- which(tune.a2.mat == max(tune.a2), arr.ind = TRUE)
    j <- tune.a2.interval[inds.a2[1,2]]
    
    # Tune the number of hidden neurons in the 3rd hidden layer:
    tune.a3 <- NULL
    tune.a3.interval <- seq(64,64*2,64)
    for (k in c(tune.a3.interval)) {
      tune.a3 <- cbind(tune.a3, keras.network(
        layer_1 = i,
        layer_2 = j,
        layer_3 = default.layer_3
      )$acc
      )
      print(c("Finished tuning a2 with rounds", k))
    }; tune.a3; tune.a3.interval
    tune.a3.mat <- data.frame(rbind(
      tune.a3.interval, tune.a3
    )); t(tune.a3.mat)
    print("Finished tuning the second hidden layer!")
    inds.a3 <- which(tune.a3.mat == max(tune.a3), arr.ind = TRUE)
    k <- tune.a3.interval[inds.a3[1,2]]
    
    # Return:
    return(list(
      i,j,k
    ))
} # End of function

# Test
tuning.keras.network(
  default.layer_1 = i,
  default.layer_2 = j,
  default.layer_3 = k
)

##################### END #################
