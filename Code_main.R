### Capstone: Analisys for Abalone dataset that can be found at:
# https://archive.ics.uci.edu/ml/datasets/Abalone

if (!require(tidyverse)) install.packages('tidyverse')
library(tidyverse)
if (!require(caret)) install.packages('caret')
library(caret)
if (!require(gam)) install.packages('gam')
library(gam)

### There are two ways to get the dataset
## Way number 1 (From the original repository at UCI Machine Learning):
abalone <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", header = FALSE)
colnames(abalone) <- c("sex","length","diameter","height","whole_weight",
                       "shucked_weight","viscera_weight","shell_weight","rings")

## Way number 2 from my repo:
abalone <- read.csv("https://raw.githubusercontent.com/vhpenagarcia/Abalone/main/abalone.csv")

## Creating a RMSE function
RMSE <- function(true_rings, predicted_rings){
  sqrt(mean((true_rings - predicted_rings)^2))
}

## Creating a second metric for regression: R^2
Rsquared <- function(true_rings, predicted_rings){
  1-(sum((true_rings - predicted_rings)^2)/sum((true_rings - mean(true_rings))^2))
}

## Pre-processing

## Checking for useless (zero or close standard deviation) variables:
abalone_mx <- as.matrix(abalone)
index_nzv <- nearZeroVar(abalone_mx) # No useless predictors

## Checking distribution of predictors
dist_predictors <- lapply(abalone[,2:8], function(X){
  abalone %>% ggplot(aes(x=X)) +
    geom_histogram(color = "black", fill = "gray70")
})

dist_predictors$height ## It shows outliers that can bias predictions

height_outliers <- abalone %>% ggplot(aes(height)) + ##Creating a plot to show outliers
  geom_boxplot(fill = "gray70") + coord_flip() +
  ylim(-0.7,0.7)+
  ylab("") +
  geom_text(aes(label=ifelse(height > 0.3, as.character(height),''), y=0),hjust=-0.4,vjust=0.5, col="blue", size=4)

## Codes to change outliers for less noisy value: the mean
index_outliers <- which(abalone$height >0.3) # Outliers are higher than 0.3
abalone$height[index_outliers]
mean_without_outliers <- mean(abalone$height[-index_outliers]) # Extracting the mean
abalone$height[index_outliers] <- mean_without_outliers # replacing the outliers
abalone$height[index_outliers] # checking the correct value

## some exploration
# Amount of individuals according to number of rings in the dataset
abalone %>% group_by(rings) %>% summarize(n = n()) %>% # Checking distribution of outcome variable
  ggplot(aes(rings, n)) +
  geom_bar(color = "black", fill = "gray70", stat = "identity")
range(abalone$rings)

# splitting the individuals according to sex
abalone %>% group_by(rings, sex) %>% summarize(n = n()) %>%
  ggplot(aes(rings, n)) +
  geom_bar(color = "black", fill = "gray70", stat = "identity") +
  facet_wrap(.~sex)



## Splitting dataset: creating validation and train datasets
set.seed(1783, sample.kind = "Rounding")
validation_index <- createDataPartition(abalone$rings, times = 1, p=0.2, list = FALSE)
validation <- abalone[validation_index,]
train <- abalone[-validation_index,]

# Further split of train: Creating training and test sets
set.seed(1709, sample.kind = "Rounding")
test_index <- createDataPartition(train$rings, times = 1, p=0.2, list = FALSE)
test_set <- train[test_index,]
train_set <- train[-test_index,]


### Training ML algorithm-based models

### KNN logarithm
modelLookup("knn") # checking algorithms
control <- trainControl(method = "cv", number = 10, p = .9) ## Control to be used

knn_model <- train(rings~., method="knn", data=train_set, # training the model
                   tuneGrid = data.frame(k = seq(1,40,2)),
                   trcontrol = control)


ggplot(knn_model, highlight = TRUE)
knn_model$bestTune

fit_knn <- predict(knn_model, test_set, type = "raw") # Obtaining predictions

# Creating data frame with results
result_knn <- data.frame(Model = "KNN", RMSE = RMSE(test_set$rings, fit_knn),
                         R_Squared = Rsquared(test_set$rings, fit_knn),
                         Tuned_parameter = "k", Best_tune = knn_model$bestTune[1,1])
knitr::kable(result_knn)


### Linear model
modelLookup("glm")
glm_model <- train(rings~., method = "glm", data = train_set) # training the model

fit_glm <- predict(glm_model, test_set, type = "raw") # obtaining predictions

# Creating a data frame with results
result_glm <- data.frame(Model = "GLM", RMSE = RMSE(test_set$rings, fit_glm),
                         R_Squared = Rsquared(test_set$rings, fit_glm),
                         Tuned_parameter = "None", Best_tune = glm_model$bestTune[1,1])
knitr::kable(result_glm)

### Classification and Regression trees (CART) model
modelLookup("rpart")
cart_model <- train(rings~., method = "rpart", data = train_set, # Training the model
                    tuneGrid = data.frame(cp = seq(0,0.1,0.01)))

ggplot(cart_model, highlight = TRUE)
cart_model$bestTune

fit_cart <- predict(cart_model, test_set) # Obtaining predictions

# Creaing a dataframe with results
result_cart <- data.frame(Model = "CART", RMSE = RMSE(test_set$rings, fit_cart),
                          R_Squared = Rsquared(test_set$rings, fit_cart),
                          Tuned_parameter = "Cp", Best_tune = cart_model$bestTune[1,1])
knitr::kable(result_cart)


### Random Forest model
modelLookup("rf")
rf_model <- train(rings~., method = "rf", data = train_set, # Training the model
                  tuneGrid = data.frame(mtry = seq(2, 16, 2)),
                  trcontrol = control)

ggplot(rf_model, highlight = TRUE)

fit_rf <- predict(rf_model, test_set) # Obtaining predictions

# Creating a data frame with results
result_rf <- data.frame(Model = "Random Forest", RMSE = RMSE(test_set$rings, fit_rf),
                     R_Squared = Rsquared(test_set$rings, fit_rf),
                     Tuned_parameter = "mtry", Best_tune = rf_model$bestTune[1,1])
knitr::kable(result_rf)


## Loess smoothing through Generalized additive Model (gamLoess)
modelLookup("gamLoess")

grid_1 <- expand.grid(span = seq(0.1, 0.7, len = 10), degree = 1)
loess1_model <- train(rings~., method = "gamLoess", data = train_set, # training model
                     tuneGrid = grid)

ggplot(loess1_model, highlight = TRUE)

fit_loess1 <- predict(loess1_model, test_set) # obtaining predictions

# Creating a data frame with results
result_loess <- data.frame(Model = "LOESS", 
                           RMSE = RMSE(test_set$rings, fit_loess1),
                           R_Squared = Rsquared(test_set$rings, fit_loess1),
                           Tuned_parameter = "span", Best_tune = loess1_model$bestTune[1,1])
knitr::kable(result_loess)

### Linear support vector Machine
modelLookup("svmLinear")

svmL_model <- train(rings~., method = "svmLinear", data = train_set, # training the model
                    tuneGrid = data.frame(C = seq(0, 2, 0.2)),
                    trcontrol = control)
ggplot(svmL_model, highlight = TRUE)

fit_svml <- predict(svmL_model, test_set$rings) # obtaining predictions

# Creating a data frmae with results
result_svml <- data.frame(Model = "SVM linear",
                          RMSE = RMSE(test_set$rings, fit_svml),
                          R_Squared = Rsquared(test_set$rings, fit_svml),
                          Tuned_parameter = "C", Best_tune = svmL_model$bestTune[1,1])
knitr::kable(result_svml)

### Neural network: Extremely bad results
modelLookup("nnet")
nnet_grid <- expand.grid(decay = seq(0.1, 0.5, 0.1),
                       size = seq(1, 15, 1))
nnet_model <- train(rings~., method = "nnet", data = train_set,
                    tuneGrid = nnet_grid)
ggplot(nnet_model, highlight = TRUE)


### Results together
fits <- rbind(result_knn,result_glm,result_cart, # putting result predictions together
              result_rf,result_loess,result_svml)

knitr::kable(fits)

#### Ensembles

## Listing predictions to create prediction means
ensemble <- data.frame(knn = fit_knn, glm = fit_glm, cart = fit_cart,
                       rf = fit_rf, loess = fit_loess1, svm = fit_svml)
ensem_mx <- list(knn_glm = as.matrix(ensemble[,1:2]), ## creating different matrices to get different ensemble means
                  knn_cart = as.matrix(ensemble[,1:3]),
                  knn_rf = as.matrix(ensemble[,1:4]),
                  knn_loess = as.matrix(ensemble[,1:5]),
                  knn_svm = as.matrix(ensemble))


ensemble_pred <- sapply(ensem_mx, rowMeans) # Means for different ensembles
colnames(ensemble_pred) <- names(ensem_mx)

# Obtaining metric values for ensembles
rmse_ensembles <- apply(X = ensemble_pred, MARGIN = 2, FUN = RMSE, true_rings = test_set$rings)
rsquared_ensembles <- apply(X = ensemble_pred, MARGIN = 2, FUN = Rsquared, true_rings = test_set$rings)

# Putting results in a data frame
result_ensembles <- data.frame(Ensemble = c("KNN - GLM", "KNN-GLM-CART", "KNN-GLM-CART-RF",
                                            "KNN-GLM-CART-RF-loess", "KNN-GLM-CART-RF-loess-SVML"),
                               RMSE = unname(rmse_ensembles), R_Squared = unname(rsquared_ensembles))
knitr::kable(result_ensembles)

## It seems that the best performing are knn, glm, loess and svm, lets see if 
#they alone performs better

ensemble_bests <- data.frame(knn = fit_knn, rf = fit_rf, # Data frame with best predictions
                             loess = fit_loess1, svm = fit_svml)
ensem_best_mx <- as.matrix(ensemble_bests)
pred_best <- rowMeans(ensem_best_mx) ## Getting the mean

# Creating a data frame with result
ensemble_bests <- data.frame(Ensemble = "KNN-RF-loess-SVML",
                             RMSE = RMSE(test_set$rings, pred_best),
                             R_Squared = Rsquared(test_set$rings, pred_best))
knitr::kable(ensemble_bests)
data.frame(Test = test_set$rings, # Exploring the relationship between predictions and real values
           Prediction = pred_best) %>%
  ggplot(aes(Test,Prediction)) +
  geom_point(col = "blue")


#### Validation

## Creating a function to apply the chosen model to the validation test
result_validation <- function(validation_set){
  knn_v <- predict(knn_model, validation_set)
  rf_v <- predict(rf_model, validation_set)
  loess_v <- predict(loess1_model, validation_set)
  svml_v <- predict(svmL_model, validation_set)
  fits_v <- data.frame(knn = knn_v, rf = rf_v, loess = loess_v, svm = svml_v)
  fits_mx <- as.matrix(fits_v)
  pred_v <- rowMeans(fits_mx)
  return(data.frame(Model = "Final Ensemble",
                    RMSE = RMSE(validation$rings, pred_v),
                    R_Squared = Rsquared(validation$rings, pred_v)))
}
validation_result <- result_validation(validation) # Applying the model
knitr::kable(validation_result) # Result

## Creating a function to get raw predicted values
prediction_validation <- function(validation_set){
  knn_v <- predict(knn_model, validation_set)
  rf_v <- predict(rf_model, validation_set)
  loess_v <- predict(loess1_model, validation_set)
  svml_v <- predict(svmL_model, validation_set)
  fits_v <- data.frame(knn = knn_v, rf = rf_v, loess = loess_v, svm = svml_v)
  fits_mx <- as.matrix(fits_v)
  return(prediction <- rowMeans(fits_mx))
}

# Exploring the relationship between predictions and real values
values_pred <- prediction_validation(validation)
data.frame(Rings = validation$rings, Prediction = values_pred) %>% 
  ggplot(aes(Rings, Prediction)) +
  geom_point(alpha = 0.5, col = "blue")
