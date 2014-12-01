#CODE FOR USING SMOTE

# required packages
install.packages("DMwR")
install.packages("caret", dependencies = c("Depends", "Suggests"))
install.packages("pROC")
install.packages("nloptr")
install.packages("minqa")
require(DMwR)
require(pROC)
require(caret)

#import csv function
Churndata=read.csv('data.csv',stringsAsFactors=FALSE)
str(Churndata)
#Removing names to prevent overfitting
#Regardless of the machine learning method, 
#Unique variables like ID, Name or in some cases, date  should always be excluded. 
#Neglecting to do so can lead to erroneous findings because the ID can be used to 
#uniquely "predict" each example. 
#Therefore, a model that includes an identifier will most likely suffer from overfitting, 
#and is not likely to generalize well to other data.

myvars = names(Churndata) %in% c("column1", "column2", "column3.On") 
Churndata_final = Churndata[!myvars]

#Churndata_final$Cancelled = as.factor(Churndata_final$Cancelled)
str(Churndata_final)
Churndata_final$Canceled = ifelse(Churndata_final$Canceled=='Not Canceled',0,1)
table(Churndata_final$Canceled)

#Confirm Rare Event
prop.table(table(Churndata_final$Canceled))


# split data set in two for training and testing portions
library(caret)
set.seed(1234)
splitIndex = createDataPartition(Churndata_final$Canceled, p = .80, list = FALSE, times = 1)
trainSplit = Churndata_final[ splitIndex,]
testSplit = Churndata_final[-splitIndex,]

# model using treebag
ctrl <- trainControl(method = "cv", number = 5)
tbmodel <- train(Canceled ~ ., data = trainSplit, method = "treebag", trControl = ctrl)
tbmodel
# predict
predictors <- names(trainSplit)[names(trainSplit) != 'Canceled']
pred <- predict(tbmodel$finalModel, testSplit[,predictors])

# score prediction using AUC
library(pROC)
auc <- roc(testSplit$Canceled, pred)
print(auc)
auc1<-plot(auc)

# SMOTE the data
library(DMwR)
trainSplit$target <- as.factor(trainSplit$Canceled)
trainSplit <- SMOTE(target ~ ., trainSplit, perc.over = 100, perc.under=200)
trainSplit$target <- as.numeric(trainSplit$Canceled)
print(prop.table(table(trainSplit$Canceled)))

# re-train and predict
tbmodel <- train(Canceled ~ ., data = trainSplit, method = "treebag", trControl = ctrl)
pred <- predict(tbmodel$finalModel, testSplit[,predictors])

# score SMOTE prediction
auc <- roc(testSplit$Canceled, pred)
print(auc)
auc2<-plot(auc)

