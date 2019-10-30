#INSTALL AND LOAD PACKAGES-----

#If pacman is missing we install it, then we load libraries
if (!require("pacman")) {
  install.packages("pacman")
  } else{
    library(pacman)
    pacman::p_load(e1071, plotly, purrr, Metrics, randomForestSRC, caTools, Rfast, DMwR, ranger, h2o, lubridate, ggplot2, RMySQL, caret, readr, dplyr, tidyr, rstudioapi)
}

#DIRECTORY? -----

current_path = getActiveDocumentContext()$path
setwd(dirname(current_path))
setwd("..")
getwd()


set.seed(420)

#UPLOAD DATA -----

Train <- read_csv("Data/trainingData.csv")
View(Train)

Test <- read_csv("Data/validationData.csv")
View(Test)


#FEATURE ENGINEERING -----

#Train set
str(Train[520:529])

Train$FLOOR <- as.factor(Train$FLOOR)
Train$BUILDINGID <- as.factor(Train$BUILDINGID)
Train$SPACEID <- as.factor(Train$SPACEID)
Train$RELATIVEPOSITION <- as.factor(Train$RELATIVEPOSITION)
Train$USERID <- as.factor(Train$USERID)
Train$PHONEID <- as.factor(Train$PHONEID)
Train$TIMESTAMP <- as_datetime(Train$TIMESTAMP)

Train[Train == 100] <- -200

#Test set
str(Test[520:530])

Test$FLOOR <- as.factor(Test$FLOOR)
Test$BUILDINGID <- as.factor(Test$BUILDINGID)
Test$SPACEID <- as.factor(Test$SPACEID)
Test$RELATIVEPOSITION <- as.factor(Test$RELATIVEPOSITION)
Test$USERID <- as.factor(Test$USERID)
Test$PHONEID <- as.factor(Test$PHONEID)
Test$TIMESTAMP <- as_datetime(Test$TIMESTAMP)

Test[Test == 100] <- -200




#Create joined dataset -----

plot(Train$LONGITUDE, Train$LATITUDE)
plot(Test$LONGITUDE, Test$LATITUDE)

DF <- rbind(Train, Test)
s_size <- floor(0.75*nrow(DF))
set.seed(420)
inTraining <- sample(seq_len(nrow(DF)), size = s_size)
DF_train <- DF[inTraining,]
DF_test <- DF[-inTraining, ]

#Cross Validation ----

trainX <- Train[,names(Train) != "Direction"]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
preProcValues

# Define train control for k fold cross validation
train_control <- trainControl(method="cv", number=10)


#BUILDING RF -----

df.rg.building <- ranger(BUILDINGID ~ . - LONGITUDE - LATITUDE - FLOOR - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF_train, importance = "permutation")
df.pred.building <- predict(df.rg.building, DF_test)
rg.table.building <- table(DF_test$BUILDINGID, df.pred.building$predictions) #1.0 accuracy
print(confusionMatrix(rg.table.building))
importance(df.rg.building)

#Floor RF per building -----

#B0

DF_train_b0 <- DF_train %>%
  filter(BUILDINGID == 0)
DF_train_b0 <- DF_train_b0[sample(1:nrow(DF_train_b0), nrow(DF_train_b0)*0.25,replace=FALSE),]
DF_test_b0 <- DF_test %>%
  filter(BUILDINGID == 0)
DF_train_b0$FLOOR <- droplevels(DF_train_b0$FLOOR)
DF_test_b0$FLOOR <- droplevels(DF_test_b0$FLOOR)


rg.b0 <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF_train_b0,  importance = "permutation")
pred.rg.b0 <- predict(rg.b0, DF_test_b0)
rg.table.b0 <- table(DF_test_b0$FLOOR, pred.rg.b0$predictions) #0.99 accuracy
print(confusionMatrix(rg.table.b0))


#B1

DF_train_b1 <- DF_train %>%
  filter(BUILDINGID == 1)
DF_train_b1 <- DF_train_b1[sample(1:nrow(DF_train_b1), nrow(DF_train_b1)*0.25,replace=FALSE),]
DF_test_b1 <- DF_test %>%
  filter(BUILDINGID == 1)
DF_train_b1$FLOOR <- droplevels(DF_train_b1$FLOOR)
DF_test_b1$FLOOR <- droplevels(DF_test_b1$FLOOR)

rg.b1x <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF_train,  importance = "permutation")
pred.rg.b1x <- predict(rg.b1x, DF_test_b1)
rg.table.b1x <- table(DF_test_b1$FLOOR, pred.rg.b1x$predictions) #0.99 accuracy
print(confusionMatrix(rg.table.b1x))



#B2

DF_train_b2 <- DF_train %>%
  filter(BUILDINGID == 2)
DF_train_b2 <- DF_train_b2[sample(1:nrow(DF_train_b2), nrow(DF_train_b2)*0.25,replace=FALSE),]
DF_test_b2 <- DF_test %>%
  filter(BUILDINGID == 2)
DF_train_b2$FLOOR <- droplevels(DF_train_b2$FLOOR)


rg.b2 <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF_train_b2,  importance = "permutation")
pred.rg.b2 <- predict(rg.b2, DF_test_b2)
rg.table.b2 <- table(DF_test_b2$FLOOR, pred.rg.b2$predictions) #0.99 accuracy
print(confusionMatrix(rg.table.b2))




#LAT/LON -----


rg.lon <- ranger(LONGITUDE ~ . - FLOOR - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF_train,  importance = "permutation")
pred.lon <- predict(rg.lon, DF_test)

postResample(pred.lon$predictions, DF_test$LONGITUDE)
postResample(rg.lon$predictions, DF_train$LONGITUDE)
mape(DF_test$LONGITUDE, pred.lon$predictions)
mae(DF_test$LONGITUDE, pred.lon$predictions)



rg.lat <- ranger(LATITUDE ~ .  - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF_train,  importance = "permutation")
pred.lat <- predict(rg.lat, DF_test)

postResample(pred.lat$predictions, DF_test$LATITUDE)
postResample(rg.lat$predictions, DF_train$LATITUDE)
mape(DF_test$LATITUDE, pred.lat$predictions)
mae(DF_test$LATITUDE, pred.lat$predictions)


#Culling the data ----

DF_plot <- DF %>%
  dplyr::group_by(PHONEID, BUILDINGID) %>%
  dplyr::summarise_at(.vars = vars(1:520),
                      .funs = c(max="max"))
DF_plot %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()


#Val only -----
DF.val <- Test
s_size <- floor(0.75*nrow(DF.val))
set.seed(420)
inTraining <- sample(seq_len(nrow(DF.val)), size = s_size)
DF.val_train <- DF.val[inTraining,]
DF.val_test <- DF.val[-inTraining, ]


df.val.rg.building <- ranger(BUILDINGID ~ . - LONGITUDE - LATITUDE - FLOOR - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train, importance = "permutation")
df.val.pred.building <- predict(df.val.rg.building, DF.val_test)
rg.val.table.building <- table(DF.val_test$BUILDINGID, df.val.pred.building$predictions) #1.0 accuracy
print(confusionMatrix(rg.val.table.building))

df.val.rg.floor <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train, importance = "permutation")
df.val.pred.floor <- predict(df.val.rg.floor, DF.val_test)
rg.val.table.floor <- table(DF.val_test$FLOOR, df.val.pred.floor$predictions) #0.9 accuracy
print(confusionMatrix(rg.val.table.floor))



#B0

DF.val_train_b0 <- DF.val_train %>%
  filter(BUILDINGID == 0)
DF.val_test_b0 <- DF.val_test %>%
  filter(BUILDINGID == 0)
DF.val_train_b0$FLOOR <- droplevels(DF.val_train_b0$FLOOR)
DF.val_test_b0$FLOOR <- droplevels(DF.val_test_b0$FLOOR)

set.seed(420)
rg.val.b0 <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train_b0,  importance = "permutation")
pred.rg.val.b0 <- predict(rg.val.b0, DF.val_test_b0)
rg.val.table.b0 <- table(DF.val_test_b0$FLOOR, pred.rg.val.b0$predictions) #0.93 accuracy
print(confusionMatrix(rg.val.table.b0))

rf.val.b0 <- train(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, data = DF.val_train_b0, method = "rf", trControl=fitControl, tuneLength = 1)
pred.rf.val.b0 <- predict(rf.val.b0, DF.val_test_b0)

#B1

DF.val_train_b1 <- DF.val_train %>%
  filter(BUILDINGID == 1)
DF.val_test_b1 <- DF.val_test %>%
  filter(BUILDINGID == 1)
DF.val_train_b1$FLOOR <- droplevels(DF.val_train_b1$FLOOR)
DF.val_test_b1$FLOOR <- droplevels(DF.val_test_b1$FLOOR)

set.seed(420)
rg.val.b1 <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train_b1,  importance = "permutation")
pred.rg.val.b1 <- predict(rg.val.b1, DF.val_test_b1)
rg.val.table.b1 <- table(DF.val_test_b1$FLOOR, pred.rg.val.b1$predictions) #0.86 accuracy
print(confusionMatrix(rg.val.table.b1))

DF.bullshit <- DF.val_train_b1 %>%
  select(-c(LONGITUDE, LATITUDE, SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP))
system.time(rf.val.b1 <-train(FLOOR ~ ., data = DF.bullshit, method = "rf", trControl=train_control, tuneLength = 1))
system.time(pred.rf.val.b1 <- predict(rf.val.b1, DF.val_test_b1))
rf.val.table.b1 <- table(DF.val_test_b1$FLOOR, pred.rf.val.b1) #0.88 accuracy
print(confusionMatrix(rf.val.table.b1))

#B2

DF.val_train_b2 <- DF.val_train %>%
  filter(BUILDINGID == 2)
DF.val_test_b2 <- DF.val_test %>%
  filter(BUILDINGID == 2)
DF.val_train_b2$FLOOR <- droplevels(DF.val_train_b2$FLOOR)
DF.val_test_b2$FLOOR <- droplevels(DF.val_test_b2$FLOOR)

set.seed(420)
rg.val.b2 <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train_b2,  importance = "permutation")
pred.rg.val.b2 <- predict(rg.val.b2, DF.val_test_b2)
rg.val.table.b2 <- table(DF.val_test_b2$FLOOR, pred.rg.val.b2$predictions) #0.85 accuracy
print(confusionMatrix(rg.val.table.b2))


#LON

rg.val.lon <- ranger(LONGITUDE ~ . - FLOOR - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train,  importance = "permutation")
pred.val.lon <- predict(rg.val.lon, DF.val_test)

postResample(pred.val.lon$predictions, DF.val_test$LONGITUDE) #8.405 MAE
postResample(rg.val.lon$predictions, DF.val_train$LONGITUDE)  #8.716 MAE

plot(pred.val.lon$predictions,DF.val_test$LONGITUDE, col = DF.val_test$BUILDINGID,
     xlab="predicted",ylab="actual")
abline(a=0,b=1)


rf.val.lon <- train(LONGITUDE ~ . - FLOOR - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train, method = "rf", trControl=fitControl, tuneLength = 1)
pred.rf.val.lon <- predict(rf.val.lon, DF.val_test_lon)

#LAT
rg.val.lat <- ranger(LATITUDE ~ .   - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train,  importance = "permutation")
pred.val.lat <- predict(rg.val.lat, DF.val_test)

postResample(pred.val.lat$predictions, DF.val_test$LATITUDE) #4.087 MAE
postResample(rg.val.lat$predictions, DF.val_train$LATITUDE)  #7.765 MAE

plot(pred.val.lat$predictions, DF.val_test$LATITUDE, col = DF.val_test$BUILDINGID,
     xlab="predicted",ylab="actual")
abline(a=0,b=1)

plot(pred.val.lon$predictions, pred.val.lat$predictions)

#New ds with difference


PredDiff <- data.frame(LAT = abs(DF.val_test$LATITUDE - pred.val.lat$predictions),
                       LON = abs(DF.val_test$LONGITUDE - pred.val.lon$predictions),
                       BUILDING = DF.val_test$BUILDINGID,
                       FLOOR = DF.val_test$FLOOR)

plot(PredDiff$LAT, col = PredDiff$BUILDING,
     xlab="Instance",ylab="Prediction Error")
plot(PredDiff$LON, col = PredDiff$BUILDING,
     xlab="Instance",ylab="Prediction Error")

plot(PredDiff$LAT, col = PredDiff$FLOOR,
     xlab="Instance",ylab="Prediction Error")
plot(PredDiff$LON, col = PredDiff$FLOOR,
     xlab="Instance",ylab="Prediction Error")
                       


plot_ly(type = "scatter3d",
        x =  DF.val$LATITUDE,
        y =  DF.val$LONGITUDE,
        z =  DF.val$FLOOR,
        mode = 'markers')


#Super cool function
BUILDING <- function(Training, Testing){
  
  Training <- Training %>%   select(-c(LONGITUDE, LATITUDE, SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP))
  Training$FLOOR <- droplevels(Training$FLOOR)
  Training$FLOOR <- droplevels(Training$FLOOR)
  
  Predictions <- list()
  Metrics <- list()
  
  #RF
  rg.building <- ranger(BUILDINGID ~ . - LONGITUDE - LATITUDE - FLOOR - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, Training, importance = "permutation")
  rg.pred.building <- predict(rg.building, Testing)
  rg.table.building <- confusionMatrix(table(Testing$BUILDINGID, rg.pred.building$predictions))
  
  Predictions[["RF"]] <- rg.pred.building
  Metrics[["RF"]] <- rg.table.building
  
  #KNN
  
  knn.building <- knn3(BUILDINGID ~ ., Training)
  knn.pred.building <- predict(knn.building, Testing)
  knn.table.building <- confusionMatrix(table(Testing$BUILDINGID, knn.pred.building))
  
  Predictions[["KNN"]] <- knn.pred.building
  Metrics[["KNN"]] <- knn.table.building
  
  #SVM
  svm.building <- svm(BUILDINGID ~ ., Training)
  svm.pred.building <- predict(svm.building, Testing)
  svm.table.building <- confusionMatrix(table(Testing$BUILDINGID, knn.pred.building))
  
  
  Predictions[["SVM"]] <- svm.pred.building
  
  
  Output <- list(Predictions, Metrics)
  Output
}

BUILDING(DF.val_train, DF.val_test)

Training <- DF.val_train %>%   select(-c(LONGITUDE, LATITUDE, SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP))
Testing <- DF.val_test%>%   select(-c(LONGITUDE, LATITUDE, SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP))
knn.building <- knn3(BUILDINGID ~ ., Training)
knn.pred.building <- predict(knn.building, DF.val_test)
knn.table.building <- confusionMatrix(table(DF.val_test$BUILDINGID, knn.pred.building))



#Seed test -----


set.seed(100)
rg.val.lat <- ranger(LATITUDE ~ .   - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train,  importance = "permutation")
pred.val.lat <- predict(rg.val.lat, DF.val_test)

postResample(pred.val.lat$predictions, DF.val_test$LATITUDE) #7.096 MAE

set.seed(200)
rg.val.lat <- ranger(LATITUDE ~ .   - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train,  importance = "permutation")
pred.val.lat <- predict(rg.val.lat, DF.val_test)

postResample(pred.val.lat$predictions, DF.val_test$LATITUDE) #7.164 MAE

set.seed(300)
rg.val.lat <- ranger(LATITUDE ~ .   - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train,  importance = "permutation")
pred.val.lat <- predict(rg.val.lat, DF.val_test)

postResample(pred.val.lat$predictions, DF.val_test$LATITUDE) #7.170 MAE

set.seed(400)
rg.val.lat <- ranger(LATITUDE ~ .   - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train,  importance = "permutation")
pred.val.lat <- predict(rg.val.lat, DF.val_test)

postResample(pred.val.lat$predictions, DF.val_test$LATITUDE) #7.171 MAE

set.seed(500)
rg.val.lat <- ranger(LATITUDE ~ .   - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train,  importance = "permutation")
pred.val.lat <- predict(rg.val.lat, DF.val_test)

postResample(pred.val.lat$predictions, DF.val_test$LATITUDE) #7.175 MAE

set.seed(600)
rg.val.lat <- ranger(LATITUDE ~ .   - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train,  importance = "permutation")
pred.val.lat <- predict(rg.val.lat, DF.val_test)

postResample(pred.val.lat$predictions, DF.val_test$LATITUDE) #7.188 MAE

set.seed(700)
rg.val.lat <- ranger(LATITUDE ~ .   - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train,  importance = "permutation")
pred.val.lat <- predict(rg.val.lat, DF.val_test)

postResample(pred.val.lat$predictions, DF.val_test$LATITUDE) #7.151 MAE

set.seed(800)
rg.val.lat <- ranger(LATITUDE ~ .   - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train,  importance = "permutation")
pred.val.lat <- predict(rg.val.lat, DF.val_test)

postResample(pred.val.lat$predictions, DF.val_test$LATITUDE) #7.191 MAE

set.seed(900)
rg.val.lat <- ranger(LATITUDE ~ .   - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.val_train,  importance = "permutation")
pred.val.lat <- predict(rg.val.lat, DF.val_test)

postResample(pred.val.lat$predictions, DF.val_test$LATITUDE) #7.134 MAE


#Adding Training lines to Validation -----

DF.106 <- Train %>% filter(SPACEID == 106)

plot(DF.106$BUILDINGID)

LATLON <- Train %>%
  mutate(WAPs = sum(Train[1:520])) %>%
  group_by(LATITUDE, LONGITUDE) %>%
  slice(which.max(WAPs))

LATLON$WAPs <- NULL

DF.TV <- bind_rows(DF.val, LATLON)
DF.TV$SPACEID <- as.factor(DF.TV$SPACEID)
DF.TV$RELATIVEPOSITION <- as.factor(DF.TV$RELATIVEPOSITION)
DF.TV$USERID <- as.factor(DF.TV$USERID)
DF.TV$PHONEID <- as.factor(DF.TV$PHONEID)
s_size <- floor(0.75*nrow(DF.TV))
set.seed(420)
inTraining <- sample(seq_len(nrow(DF.TV)), size = s_size)
DF.TV_train <- DF.TV[inTraining,]
DF.TV_test <- DF.TV[-inTraining, ] 

#BUILDING
df.tv.rg.building <- ranger(BUILDINGID ~ . - LONGITUDE - LATITUDE - FLOOR - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV_train, importance = "permutation")
df.tv.pred.building <- predict(df.tv.rg.building, DF.TV_test)
rg.tv.table.building <- table(DF.TV_test$BUILDINGID, df.tv.pred.building$predictions) #0.9933 accuracy
print(confusionMatrix(rg.tv.table.building))

#FLOOR
df.tv.rg.floor <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV_train, importance = "permutation")
df.tv.pred.floor <- predict(df.tv.rg.floor, DF.TV_test)
rg.tv.table.floor <- table(DF.TV_test$FLOOR, df.tv.pred.floor$predictions) #0.942 accuracy
print(confusionMatrix(rg.tv.table.floor))

#B0

DF.TV_train_b0 <- DF.TV_train %>%
  filter(BUILDINGID == 0)
DF.TV_test_b0 <- DF.TV_test %>%
  filter(BUILDINGID == 0)
DF.TV_train_b0$FLOOR <- droplevels(DF.TV_train_b0$FLOOR)
DF.TV_test_b0$FLOOR <- droplevels(DF.TV_test_b0$FLOOR)

set.seed(420)
rg.TV.b0 <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV_train_b0,  importance = "permutation")
pred.rg.TV.b0 <- predict(rg.TV.b0, DF.TV_test_b0)
rg.TV.table.b0 <- table(DF.TV_test_b0$FLOOR, pred.rg.TV.b0$predictions) #0.98 accuracy
print(confusionMatrix(rg.TV.table.b0))

#B1

DF.TV_train_b1 <- DF.TV_train %>%
  filter(BUILDINGID == 1)
DF.TV_test_b1 <- DF.TV_test %>%
  filter(BUILDINGID == 1)
DF.TV_train_b1$FLOOR <- droplevels(DF.TV_train_b1$FLOOR)
DF.TV_test_b1$FLOOR <- droplevels(DF.TV_test_b1$FLOOR)

set.seed(420)
rg.TV.b1 <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV_train_b1,  importance = "permutation")
pred.rg.TV.b1 <- predict(rg.TV.b1, DF.TV_test_b1)
rg.TV.table.b1 <- table(DF.TV_test_b1$FLOOR, pred.rg.TV.b1$predictions) #0.93 accuracy
print(confusionMatrix(rg.TV.table.b1))

#B2

DF.TV_train_b2 <- DF.TV_train %>%
  filter(BUILDINGID == 2)
DF.TV_test_b2 <- DF.TV_test %>%
  filter(BUILDINGID == 2)
DF.TV_train_b2$FLOOR <- droplevels(DF.TV_train_b2$FLOOR)
DF.TV_test_b2$FLOOR <- droplevels(DF.TV_test_b2$FLOOR)

set.seed(420)
rg.TV.b2 <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV_train_b2,  importance = "permutation")
pred.rg.TV.b2 <- predict(rg.TV.b2, DF.TV_test_b2)
rg.TV.table.b2 <- table(DF.TV_test_b2$FLOOR, pred.rg.TV.b2$predictions) #0.92 accuracy
print(confusionMatrix(rg.TV.table.b2))


#LON

set.seed(420)
rg.TV.lon <- ranger(LONGITUDE ~ . - FLOOR - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV_train,  importance = "permutation")
pred.TV.lon <- predict(rg.TV.lon, DF.TV_test)

postResample(pred.TV.lon$predictions, DF.TV_test$LONGITUDE) #8.726 MAE
postResample(rg.TV.lon$predictions, DF.TV_train$LONGITUDE)  #8.403 MAE

plot(pred.TV.lon$predictions,DF.TV_test$LONGITUDE, col = DF.TV_test$BUILDINGID,
     xlab="predicted",ylab="actual")
abline(a=0,b=1)



#LAT
set.seed(420)
rg.TV.lat <- ranger(LATITUDE ~ . - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV_train,  importance = "permutation")
pred.TV.lat <- predict(rg.TV.lat, DF.TV_test)

postResample(pred.TV.lat$predictions, DF.TV_test$LATITUDE) #7.578 MAE
postResample(rg.TV.lat$predictions, DF.TV_train$LATITUDE)  #7.381 MAE

plot(pred.TV.lat$predictions,DF.TV_test$LATITUDE, col = DF.TV_test$BUILDINGID,
     xlab="predicted",ylab="actual")
abline(a=0,b=1)

