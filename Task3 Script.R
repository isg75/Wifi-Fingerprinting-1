#INSTALL AND LOAD PACKAGES-----

#If pacman is missing we install it, then we load libraries
if (!require("pacman")) {
  install.packages("pacman")
  } else{
    library(pacman)
    pacman::p_load(e1071, plotly, purrr, Metrics, randomForestSRC, caTools, Rfast, DMwR, ranger, h2o, lubridate, ggplot2, RMySQL, caret, readr, dplyr, tidyr, rstudioapi)
}

#DIRECTORY -----

current_path = getActiveDocumentContext()$path
setwd(dirname(current_path))
setwd("..")
getwd()



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

Train[Train == 100] <- -105

#Test set
str(Test[520:529])

Test$FLOOR <- as.factor(Test$FLOOR)
Test$BUILDINGID <- as.factor(Test$BUILDINGID)
Test$SPACEID <- as.factor(Test$SPACEID)
Test$RELATIVEPOSITION <- as.factor(Test$RELATIVEPOSITION)
Test$USERID <- as.factor(Test$USERID)
Test$PHONEID <- as.factor(Test$PHONEID)
Test$TIMESTAMP <- as_datetime(Test$TIMESTAMP)

Test[Test == 100] <- -105




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

train_control <- trainControl(method="cv", number=10)


#Training + Validation -----
#BUILDING

df.rg.building <- ranger(BUILDINGID ~ . - LONGITUDE - LATITUDE - FLOOR - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF_train, importance = "permutation")
df.pred.building <- predict(df.rg.building, DF_test)
rg.table.building <- table(DF_test$BUILDINGID, df.pred.building$predictions) #1.0 accuracy
print(confusionMatrix(rg.table.building))
importance(df.rg.building)

#Floor RF per building
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




#LAT/LON


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
set.seed(420)
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
set.seed(420)
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
        x =  Train$LATITUDE,
        y =  Train$LONGITUDE,
        z =  Train$FLOOR,
        mode = 'markers',
        color = ~Train$PHONEID)





#BUILDING
set.seed(420)
df.tv.rg.building <- ranger(BUILDINGID ~ . - LONGITUDE - LATITUDE - FLOOR - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV, importance = "permutation")
df.tv.pred.building <- predict(df.tv.rg.building, DF.val_test)
rg.tv.table.building <- table(DF.val_test$BUILDINGID, df.tv.pred.building$predictions) #0.1 accuracy
print(confusionMatrix(rg.tv.table.building))

#FLOOR
set.seed(420)
df.tv.rg.floor <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV, importance = "permutation")
df.tv.pred.floor <- predict(df.tv.rg.floor, DF.val_test)
rg.tv.table.floor <- table(DF.val_test$FLOOR, df.tv.pred.floor$predictions) #0.9424 accuracy
print(confusionMatrix(rg.tv.table.floor))

#B0

DF.TV_train_b0 <- DF.TV %>%
  filter(BUILDINGID == 0)
DF.TV_test_b0 <- DF.val_test %>%
  filter(BUILDINGID == 0)
DF.TV_train_b0$FLOOR <- droplevels(DF.TV_train_b0$FLOOR)
DF.TV_test_b0$FLOOR <- droplevels(DF.TV_test_b0$FLOOR)

set.seed(420)
rg.TV.b0 <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV_train_b0,  importance = "permutation")
pred.rg.TV.b0 <- predict(rg.TV.b0, DF.TV_test_b0)
rg.TV.table.b0 <- table(DF.TV_test_b0$FLOOR, pred.rg.TV.b0$predictions) #0.94 accuracy
print(confusionMatrix(rg.TV.table.b0))

#B1

DF.TV_train_b1 <- DF.TV %>%
  filter(BUILDINGID == 1)
DF.TV_test_b1 <- DF.val_test %>%
  filter(BUILDINGID == 1)
DF.TV_train_b1$FLOOR <- droplevels(DF.TV_train_b1$FLOOR)
DF.TV_test_b1$FLOOR <- droplevels(DF.TV_test_b1$FLOOR)

set.seed(420)
rg.TV.b1 <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV_train_b1,  importance = "permutation")
pred.rg.TV.b1 <- predict(rg.TV.b1, DF.TV_test_b1)
rg.TV.table.b1 <- table(DF.TV_test_b1$FLOOR, pred.rg.TV.b1$predictions) #0.90 accuracy
print(confusionMatrix(rg.TV.table.b1))

#B2

DF.TV_train_b2 <- DF.TV %>%
  filter(BUILDINGID == 2)
DF.TV_test_b2 <- DF.val_test %>%
  filter(BUILDINGID == 2)
DF.TV_train_b2$FLOOR <- droplevels(DF.TV_train_b2$FLOOR)
DF.TV_test_b2$FLOOR <- droplevels(DF.TV_test_b2$FLOOR)

set.seed(420)
rg.TV.b2 <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV_train_b2,  importance = "permutation")
pred.rg.TV.b2 <- predict(rg.TV.b2, DF.TV_test_b2)
rg.TV.table.b2 <- table(DF.TV_test_b2$FLOOR, pred.rg.TV.b2$predictions) #0.98 accuracy
print(confusionMatrix(rg.TV.table.b2))


#LON

set.seed(420)
rg.TV.lon <- ranger(LONGITUDE ~ . - FLOOR - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV,  importance = "permutation")
pred.TV.lon <- predict(rg.TV.lon, DF.val_test)

postResample(pred.TV.lon$predictions, DF.val_test$LONGITUDE) #7.268 MAE
postResample(rg.TV.lon$predictions, DF.TV$LONGITUDE)  #7.904 MAE

plot(pred.TV.lon$predictions,DF.val_test$LONGITUDE, col = DF.val_test$BUILDINGID,
     xlab="predicted",ylab="actual")
abline(a=0,b=1)



#LAT
set.seed(420)
rg.TV.lat <- ranger(LATITUDE ~ . - FLOOR - LONGITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, DF.TV,  importance = "permutation")
pred.TV.lat <- predict(rg.TV.lat, DF.val_test)

postResample(pred.TV.lat$predictions, DF.val_test$LATITUDE) #6.247 MAE
postResample(rg.TV.lat$predictions, DF.TV$LATITUDE)  #7.025 MAE

plot(pred.TV.lat$predictions,DF.val_test$LATITUDE, col = DF.val_test$BUILDINGID,
     xlab="predicted",ylab="actual")
abline(a=0,b=1)



PredDiff <- data.frame(LAT = abs(DF.val_test$LATITUDE - pred.TV.lat$predictions),
                       LON = abs(DF.val_test$LONGITUDE - pred.TV.lon$predictions),
                       BUILDING = DF.val_test$BUILDINGID,
                       FLOOR = DF.val_test$FLOOR,
                       PHONEID = DF.val_test$PHONEID,
                       USERID = DF.val_test$USERID)

plot(PredDiff$LAT, col = PredDiff$BUILDING,
     xlab="Instance",ylab="Prediction Error")
plot(PredDiff$LON, col = PredDiff$BUILDING,
     xlab="Instance",ylab="Prediction Error")

plot(PredDiff$LAT, col = PredDiff$FLOOR,
     xlab="Instance",ylab="Prediction Error")
plot(PredDiff$LON, col = PredDiff$FLOOR,
     xlab="Instance",ylab="Prediction Error")


table(Train$USERID)
table(Train$PHONEID)


plot_ly(type = "scatter3d",
        x =  Train$LATITUDE,
        y =  Train$LONGITUDE,
        z =  Train$FLOOR,
        mode = 'markers',
        color = ~Train$PHONEID)





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

            
LATLONf0 <- Train %>%
  filter(FLOOR == 0) %>%
  mutate(WAPs = sum(Train[1:520])) %>%
  group_by(LATITUDE, LONGITUDE) %>%
  slice(which.max(WAPs))

LATLONf1 <- Train %>%
  filter(FLOOR == 1) %>%
  mutate(WAPs = sum(Train[1:520])) %>%
  group_by(LATITUDE, LONGITUDE) %>%
  slice(which.max(WAPs))

LATLONf2 <- Train %>%
  filter(FLOOR == 2) %>%
  mutate(WAPs = sum(Train[1:520])) %>%
  group_by(LATITUDE, LONGITUDE) %>%
  slice(which.max(WAPs))

LATLONf3 <- Train %>% 
  filter(FLOOR == 3) %>%
  mutate(WAPs = sum(Train[1:520])) %>%
  group_by(LATITUDE, LONGITUDE) %>%
  slice(which.max(WAPs))

LATLONf4 <- Train %>%
  filter(FLOOR == 4) %>%
  mutate(WAPs = sum(Train[1:520])) %>%
  group_by(LATITUDE, LONGITUDE) %>%
  slice(which.max(WAPs))

LATLON <- bind_rows(LATLONf0, LATLONf1, LATLONf2, LATLONf3, LATLONf4)
LATLON$WAPs <- NULL

DF.TV <- bind_rows(DF.val_train, LATLON)
DF.TV$SPACEID <- as.factor(DF.TV$SPACEID)
DF.TV$RELATIVEPOSITION <- as.factor(DF.TV$RELATIVEPOSITION)
DF.TV$USERID <- as.factor(DF.TV$USERID)
DF.TV$PHONEID <- as.factor(DF.TV$PHONEID)

#Super cool function -----
BUILDING <- function(Training, Testing){
  
  Training <- Training %>%   select(-c(LONGITUDE, LATITUDE, FLOOR, SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP))
  
  Model <- list()
  Predictions <- list()
  Metrics <- list()
  
  #RF
  set.seed(420)
  rg.building <- ranger(BUILDINGID ~ . - LONGITUDE - LATITUDE - FLOOR - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, Training, importance = "permutation")
  rg.pred.building <- predict(rg.building, Testing)
  rg.table.building <- confusionMatrix(table(Testing$BUILDINGID, rg.pred.building$predictions))
  
  Model[["RF"]] <- rg.building
  Predictions[["RF"]] <- rg.pred.building
  Metrics[["RF"]] <- rg.table.building
  
  #KNN
  set.seed(420)
  knn.building <- train(BUILDINGID ~ ., Training, method = "knn")
  knn.pred.building <- predict(knn.building, Testing)
  knn.table.building <- confusionMatrix(table(Testing$BUILDINGID, knn.pred.building))
  
  Model[["KNN"]] <- knn.building
  Predictions[["KNN"]] <- knn.pred.building
  Metrics[["KNN"]] <- knn.table.building
  
  #SVM
  set.seed(420)
  svm.building <- svm(BUILDINGID ~ ., Training)
  svm.pred.building <- predict(svm.building, Testing)
  svm.table.building <- confusionMatrix(table(Testing$BUILDINGID, svm.pred.building))
  
  Model[["SVM"]] <- svm.building
  Predictions[["SVM"]] <- svm.pred.building
  Metrics[["SVM"]] <- svm.table.building
  
  
  Output <- list(Model, Predictions, Metrics)
  Output
}

FLOOR <- function(Training, Testing){
  
  Training <- Training %>%   select(-c(LONGITUDE, LATITUDE, SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP))
  Training$FLOOR <- droplevels(Training$FLOOR)
  Training$FLOOR <- droplevels(Training$FLOOR)
  
  Model <- list()
  Predictions <- list()
  Metrics <- list()
  
  #RF
  set.seed(420)
  rg.floor <- ranger(FLOOR ~ . - LONGITUDE - LATITUDE - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, Training, importance = "permutation")
  rg.pred.floor <- predict(rg.floor, Testing)
  rg.table.floor <- confusionMatrix(table(Testing$FLOOR, rg.pred.floor$predictions))
  
  Model[["RF"]] <- rg.floor
  Predictions[["RF"]] <- rg.pred.floor
  Metrics[["RF"]] <- rg.table.floor
  
  #KNN
  set.seed(420)
  knn.floor <- train(FLOOR ~ ., Training, method = "knn")
  knn.pred.floor <- predict(knn.floor, Testing)
  knn.table.floor <- confusionMatrix(table(Testing$FLOOR, knn.pred.floor))
  
  Model[["KNN"]] <- knn.floor
  Predictions[["KNN"]] <- knn.pred.floor
  Metrics[["KNN"]] <- knn.table.floor
  
  #SVM
  set.seed(420)
  svm.floor <- svm(FLOOR ~ ., Training)
  svm.pred.floor <- predict(svm.floor, Testing)
  svm.table.floor <- confusionMatrix(table(Testing$FLOOR, svm.pred.floor))
  
  Model[["SVM"]] <- svm.floor
  Predictions[["SVM"]] <- svm.pred.floor
  Metrics[["SVM"]] <- svm.table.floor
  
  
  Output <- list(Model, Predictions, Metrics)
  Output
}

LATITUDE <- function(Training, Testing){
  
  Training <- Training %>%   select(-c(LONGITUDE, FLOOR, SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP))
  
  Model <- list()
  Predictions <- list()
  Metrics <- list()
  
  #RF
  set.seed(420)
  rg.lat <- ranger(LATITUDE ~ . - LONGITUDE - FLOOR - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, Training, importance = "permutation")
  rg.pred.lat <- predict(rg.lat, Testing, interval = "predict")
  rg.table.lat <- postResample(rg.pred.lat$predictions, Testing$LATITUDE)
  
  Model[["RF"]] <- rg.lat
  Predictions[["RF"]] <- rg.pred.lat
  Metrics[["RF"]] <- rg.table.lat
  
  #KNN
  set.seed(420)
  knn.lat <- train(LATITUDE ~ ., Training, method = "knn")
  knn.pred.lat <- predict(knn.lat, Testing)
  knn.table.lat <- postResample(knn.pred.lat, Testing$LATITUDE)
  
  Model[["KNN"]] <- knn.lat
  Predictions[["KNN"]] <- knn.pred.lat
  Metrics[["KNN"]] <- knn.table.lat
  
  #SVM
  set.seed(420)
  svm.lat <- svm(LATITUDE ~ ., Training)
  svm.pred.lat <- predict(svm.lat, Testing, interval = "predict")
  svm.table.lat <- postResample(svm.pred.lat, Testing$LATITUDE)
  
  Model[["SVM"]] <- svm.lat
  Predictions[["SVM"]] <- svm.pred.lat
  Metrics[["SVM"]] <- svm.table.lat
  
  
  Output <- list(Model, Predictions, Metrics)
  Output
}

LONGITUDE <- function(Training, Testing){
  
  Training <- Training %>%   select(-c(LATITUDE, FLOOR, SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP))
  
  Model <- list()
  Predictions <- list()
  Metrics <- list()
  
  #RF
  set.seed(420)
  rg.lon <- ranger(LONGITUDE ~ . - LATITUDE - FLOOR - SPACEID - RELATIVEPOSITION - USERID - PHONEID - TIMESTAMP, Training, importance = "permutation")
  rg.pred.lon <- predict(rg.lon, Testing)
  rg.table.lon <- postResample(rg.pred.lon$predictions, Testing$LONGITUDE)
  
  Model[["RF"]] <- rg.lon
  Predictions[["RF"]] <- rg.pred.lon
  Metrics[["RF"]] <- rg.table.lon
  
  #KNN
  set.seed(420)
  knn.lon <- train(LONGITUDE ~ ., Training, method = "knn")
  knn.pred.lon <- predict(knn.lon, Testing)
  knn.table.lon <- postResample(knn.pred.lon, Testing$LONGITUDE)
  
  Model[["KNN"]] <- knn.lon
  Predictions[["KNN"]] <- knn.pred.lon
  Metrics[["KNN"]] <- knn.table.lon
  
  #SVM
  set.seed(420)
  svm.lon <- svm(LONGITUDE ~ ., Training)
  svm.pred.lon <- predict(svm.lon, Testing)
  svm.table.lon <- postResample(svm.pred.lon, Testing$LONGITUDE)
  
  Model[["SVM"]] <- svm.lon
  Predictions[["SVM"]] <- svm.pred.lon
  Metrics[["SVM"]] <- svm.table.lon
  
  
  Output <- list(Model, Predictions, Metrics)
  Output
}

Results <- list()
Results[["BUILDING"]] <- BUILDING(DF.TV, DF.val_test)
Results[["FLOOR"]] <- FLOOR(DF.TV, DF.val_test)
Results[["LATITUDE"]] <- LATITUDE(DF.TV, DF.val_test)
Results[["LONGITUDE"]] <- LONGITUDE(DF.TV, DF.val_test)


#PLOTS ----
#LAT errors
plot(Results[[3]][[2]]$KNN - DF.val_test$LATITUDE, type = "l", col =  "blue")
lines(Results[[3]][[2]]$RF$predictions - DF.val_test$LATITUDE, type = "l", col = "red")

#LON errors
plot(Results[[4]][[2]]$KNN - DF.val_test$LONGITUDE, type = "l", col =  "blue")
lines(Results[[4]][[2]]$RF$predictions - DF.val_test$LONGITUDE, type = "l", col = "red")

#ESQUISSE
library(esquisse)

PredDiff <- data.frame(LAT.KNN <- Results[[3]][[2]]$KNN - DF.val_test$LATITUDE,
                       LAT.RF <- Results[[3]][[2]]$RF$predictions - DF.val_test$LATITUDE,
                       LON.KNN <- Results[[4]][[2]]$KNN - DF.val_test$LONGITUDE,
                       LON.RF <- Results[[4]][[2]]$RF$predictions - DF.val_test$LONGITUDE,
                       B.KNN <- Results[[1]][[2]]$KNN,
                       B.RF <- Results[[1]][[2]]$RF$predictions,
                       B.Test <- DF.val_test$BUILDINGID,
                       F.KNN <- Results[[2]][[2]]$KNN,
                       F.RF <- Results[[2]][[2]]$RF$predictions,
                       F.Test <- DF.val_test$FLOOR,
                       BUILDING = DF.val_test$BUILDINGID,
                       FLOOR = DF.val_test$FLOOR,
                       PHONEID = DF.val_test$PHONEID,
                       USERID = DF.val_test$USERID)


#LAT ERRORS
ggplot(DF.val_test) + aes(x = as.numeric(row.names(PredDiff))) + 
  geom_line(aes(y = PredDiff$LAT.KNN), stat = "identity", col = "#00bfa5", size = 0.75) + 
  geom_line(aes(y = PredDiff$LAT.RF), stat = "identity", col = "#ef6c00", size = 0.75) +
  labs(x = "", y = "MAE", title = "Latitude Errors", subtitle = "KNN vs RF") +
  theme_minimal()

#LON ERRORS
ggplot(DF.val_test) + aes(x = as.numeric(row.names(PredDiff))) + 
  geom_line(aes(y = PredDiff$LON.KNN), stat = "identity", col = "#00bfa5", size = 0.75) + 
  geom_line(aes(y = PredDiff$LON.RF), stat = "identity", col = "#ef6c00", size = 0.75) +
  labs(x = "", y = "MAE", title = "Longitude Errors", subtitle = "KNN vs RF") +
  theme_minimal()



ggplot(Train) +
  aes(x = LATITUDE, y = LONGITUDE) +
  geom_point(size = 1L, colour = "#00bfa5") +
  labs(title = "Train Latitude Longitude") +
  theme_minimal()

ggplot(Test) +
  aes(x = LATITUDE, y = LONGITUDE) +
  geom_point(size = 1L, colour = "#00bfa5") +
  labs(title = "Validation Latitude Longitude") +
  theme_minimal()



