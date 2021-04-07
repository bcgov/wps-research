# code sample provided May 11, 2020 by Dr. David Hill, professor of Geography at Thompson Rivers University
library(randomForest)
library(caret)
library(tuneRanger)
library(mlr)
library(OpenML)
iris
names(iris)

class=iris$Species
features = iris[,-5]
names(features)

#No hyperparameter optimization

randomForest(x=features, y=class,
				mtry=2, ntree=1000, 
				keep.forest = TRUE)

ranger(Species~.,iris, 
		mtry=2, num.trees=1000,
		write.forest=TRUE) 

# With hyperparameter optimization by caret
parGrid <-  expand.grid(mtry = 1:4)
trCntrl = trainControl(method='repeatedcv', number=10, repeats=5)
library(caret)
fit.rf<-caret::train(Species~.,data=iris,
				method='rf', 
				metric='Accuracy', 
				trControl=trCntrl,
				tuneGrid=parGrid,
				trace=FALSE,
				ntree=1000
			)
			#trace=FALSE quiets screen ouput during iterations of training
	
fit.rf


# with hyperparameter tuning with tuneRanger
iris.task=makeClassifTask(data=iris, target="Species")

estimateTimeTuneRanger(iris.task, num.tree=1000)

res=tuneRanger(iris.task, 
	measure=list(multiclass.brier), 
	num.trees=1000, num.threads=2,
	iters=70, iters.warmup=30,
	parameters = list(replace = FALSE, 
		respect.unordered.factors = "order",
		importance="impurity")
	)
	
res$model
