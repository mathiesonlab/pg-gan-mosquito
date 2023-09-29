# Script for ABC cross-validation and empirical data prediction for pg_gan_mosquito model selection
# ABC rejection analysis 

# Adopted from From the manuscript Kirschner & Perez et al. “Congruent evolutionary
# responses of European steppe biota to late Quaternary climate change:
# insights from convolutional neural network-based demographic modeling.”


if (!require("abc")) install.packages("abc",repos = "http://cran.us.r-project.org")
# Load the ABC library.
library(abc)

args = commandArgs(trailingOnly=TRUE)

if (length(args)!=4) {
  stop("Not 4 arguments were supplied (input file).n", call.=FALSE)
} 

print(args)

testSet_labels = args[1]
testSet_Predictions = args[2]
Emp_Predictions = args[3]
output = args[4]

# The list of the generating model for each simulation in the test set.
models<-as.numeric(unlist(read.table(testSet_labels)))

# The predictions made by the trained CNN for each simulation, which will be used as SuSt.
sust<-read.table(testSet_Predictions)

# The median value of CNN predictions for the empirical data.
emp<-read.table(Emp_Predictions)
emp<-apply(emp, 2, FUN = median)

#Perform cross validation using the test set and different thresholds to select the tolerance value with the highest accuracy.

cv.modsel <- cv4postpr(models, sust, nval=10, tol =c(.05,.01,.005,.002,.001), method="rejection")
summary(cv.modsel)
#plotting for multi-class classification and less deterministic confusion matrixs
#plot(cv.modsel, names.arg=c("dadi_joint", "dadi_joint_mig"))

#Perform rejection with the empirical data and the selected threshold
Rej.05<-postpr(emp, models, sust, tol = 0.05, method = "rejection")
summary(Rej.05)

sink(output)


