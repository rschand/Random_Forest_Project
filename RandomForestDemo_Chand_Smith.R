library(randomForest)
library(caret)
library(tictoc)

set.seed(1012)

##Q0
#load data
font1 = read.table("BAUHAUS.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
font2 = read.table("JOKERMAN.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
font3 = read.table("MAGNETO.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
font4 = read.table("AGENCY.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
font5 = read.table("FRENCH.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
font6 = read.table("PALACE.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

##Q1
#filter out unnecessary columns from font data, clean data
font1 = subset(font1, select = -c(fontVariant:m_label, orientation:w))
font1[font1==""]<-NA
font1[font1==" "]<-NA
font1<-font1[complete.cases(font1),]
font2 = subset(font2, select = -c(fontVariant:m_label, orientation:w))
font2[font2==""]<-NA
font2[font2==" "]<-NA
font2<-font2[complete.cases(font2),]
font3 = subset(font3, select = -c(fontVariant:m_label, orientation:w))
font3[font3==""]<-NA
font3[font3==" "]<-NA
font3<-font3[complete.cases(font3),]
font4 = subset(font4, select = -c(fontVariant:m_label, orientation:w))
font4[font4==""]<-NA
font4[font4==" "]<-NA
font4<-font4[complete.cases(font4),]
font5 = subset(font5, select = -c(fontVariant:m_label, orientation:w))
font5[font5==""]<-NA
font5[font5==" "]<-NA
font5<-font5[complete.cases(font5),]
font6 = subset(font6, select = -c(fontVariant:m_label, orientation:w))
font6[font6==""]<-NA
font6[font6==" "]<-NA
font6<-font6[complete.cases(font6),]

#create classes with only strength = 0.7 and italic = 0 values
cl1 = subset(font1, font1$strength == 0.7)
cl1 = subset(cl1, cl1$italic == 0)
cl2 = subset(font2, font2$strength == 0.7)
cl2 = subset(cl2, cl2$italic == 0)
cl3 = subset(font3, font3$strength == 0.7)
cl3 = subset(cl3, cl3$italic == 0)
cl4 = subset(font4, font4$strength == 0.7)
cl4 = subset(cl4, cl4$italic == 0)
cl5 = subset(font5, font5$strength == 0.7)
cl5 = subset(cl5, cl5$italic == 0)
cl6 = subset(font6, font6$strength == 0.7)
cl6 = subset(cl6, cl6$italic == 0)

#calculate and display class size relative to total number of data points
total.size = nrow(cl1) + nrow(cl2) + nrow(cl3) + nrow(cl4) + nrow(cl5) + nrow(cl5)
cl1.perc = round((nrow(cl1)/total.size), 3)*100
cl2.perc = round((nrow(cl2)/total.size), 3)*100
cl3.perc = round((nrow(cl3)/total.size), 3)*100
cl4.perc = round((nrow(cl4)/total.size), 3)*100
cl5.perc = round((nrow(cl5)/total.size), 3)*100
cl6.perc = round((nrow(cl6)/total.size), 3)*100
cat("Class 1 Size: ", nrow(cl1),"\t\t"); cat("Class 2 Size: ", nrow(cl2),"\t\t"); cat("Class 3 Size: ", nrow(cl3))
cat("Class 4 Size: ", nrow(cl4),"\t\t"); cat("Class 5 Size: ", nrow(cl5),"\t\t"); cat("Class 6 Size: ", nrow(cl6))
cat("Class 1 Percentage: ", cl1.perc,"\t"); cat("Class 2 Percentage: ", cl2.perc,"\t"); cat("Class 3 Percentage: ", cl3.perc,"\t")
cat("Class 4 Percentage: ", cl4.perc,"\t"); cat("Class 5 Percentage: ", cl5.perc,"\t"); cat("Class 6 Percentage: ", cl6.perc,"\t")

#removes unnecessary columns from class data
cl1 = subset(cl1, select = -c(strength:italic))
cl2 = subset(cl2, select = -c(strength:italic))
cl3 = subset(cl3, select = -c(strength:italic))
cl4 = subset(cl4, select = -c(strength:italic))
cl5 = subset(cl5, select = -c(strength:italic))
cl6 = subset(cl6, select = -c(strength:italic))

#combine filtered font classes, create empty columns for mean and sd data
font.data = rbind(cl1, cl2, cl3, cl4, cl5, cl6)
mean.x = c()
sd.x = c()

##Q2
#take mean and std. dev. of each feature column
counter = 1:(length(font.data))
for (i in counter) {
  mean.x[i] = mean(font.data[,i])
  sd.x[i] = sd(font.data[,i])
}

#remove first NA value from font column
mean.x = mean.x[-1]
sd.x = sd.x[-1]

#scale and center data
RESF = scale(font.data[,2:401])

##Q3
#create correlation matrix
RESF.cor = cor(RESF)

#compute eigenvalues
RESF.ev = eigen(RESF.cor)
L = RESF.ev$values
W = RESF.ev$vectors

#plot eigenvalues v. mean
v=c(1:400)
plot(v, L, xlab="m", ylab="L(m)", main="Eigenvalues v. m")

##Q4
#calculate number of eigenvalues that capture 97.5% of data variance
cumulative_variance_percent=cumsum(RESF.ev$values)/400
threshold = 0.975
number_components=min(v[(cumulative_variance_percent>threshold)])
plot(v, cumulative_variance_percent, xlab="m", ylab="PEV(m)",
     main="Percentage of Explained Variance v. m")
cat(number_components, "components capture 97.5% of variance.")

redW = W[,1:number_components]

##Q5
#calculate matrix Z
Z = RESF%*%redW

##Q8
#separate data and create label column
Zlabel = font.data[,1]
Zlabel = as.factor(Zlabel)

tic()
model1 = randomForest(Zlabel~., Z, ntree = 100,importance=T,do.trace=T)
toc()
tic.clear

#print mean OOB accuracy
font.names = colnames(model1$err.rate)
n100err = (1-mean(model1$err.rate[100,]))*100
cat("OOB Accuracy for n=100 trees:",n100err)

#each 100 trees takes around 2-3 seconds of processing time. lets say t ~= .025n

##Q9
#repeat randomForest for n=200,300,400,500, and compute mean OOB accuracy
tic()
model2 = randomForest(Zlabel~., Z, ntree = 200,importance=T,do.trace=T)
toc()
tic.clear
n200err = (1-mean(model2$err.rate[200,]))*100

tic()
model3 = randomForest(Zlabel~., Z, ntree = 300,importance=T,do.trace=T)
toc()
tic.clear
n300err = (1-mean(model3$err.rate[300,]))*100

tic()
model4 = randomForest(Zlabel~., Z, ntree = 400,importance=T,do.trace=T)
toc()
tic.clear
n400err = (1-mean(model4$err.rate[400,]))*100

tic()
model5 = randomForest(Zlabel~., Z, ntree = 500,importance=T,do.trace=T)
toc()
tic.clear
n500err = (1-mean(model5$err.rate[500,]))*100

#create table comparing tree number, accuracy, and computation time
tree.qty = c("100 trees","200 trees","300 trees","400 trees","500 trees")
tree.acc = c(n100err, n200err, n300err, n400err, n500err)
comp.time = c("2.5s", "5s", "7.5s", "10s", "12.5s")
tree.comp = data.frame(tree.qty,tree.acc,comp.time)
colnames(tree.comp) = c("Tree Quantity", "% OOB Accuracy", "Comp. Time")
tree.comp

#plot number of trees vs. tree error
n.trees = c(100,200,300,400,500)
mean.error = c((100-n100err),(100-n200err),(100-n300err),(100-n400err),(100-n500err))

plot(n.trees, mean.error, xlab = "Number of Trees", ylab = "% Mean Error", 
     main = "Number of Trees v. Mean Error", type = "l")
points(n.trees,mean.error, col = "red")

##Q10
#set number of trees at 400, split data
n = dim(Z)[1]
set.seed(123)
idx = sample(1:n,floor(n*0.9))

x_train = Z[idx,]
x_test = Z[-idx,]
y_train = Zlabel[idx]
y_test = Zlabel[-idx]

#run n=400 trees on training data
model.opt = randomForest(y_train~.,x_train, ntree = 400,importance=T,do.trace=T)

#calculate OOB accuracy and standard accuracy
train.acc = (sum(model.opt$predicted == y_train)/length(y_train))*100
train.oob = (1-mean(model.opt$err.rate[400,]))*100
cat("Training standard accuracy:",train.acc,"%")
cat("Training OOB accuracy:",train.oob,"%")

#use training model on test data, compare accuracies
model_pred = predict(model.opt,x_test)
test.acc = (sum(model_pred == y_test)/length(y_test))*100
cat("Test accuracy:",test.acc,"%")

##Q11
#confusion matrix for
print("Training Data Confusion Matrix (%)")
train.conf = model.opt$confusion
train.conf = train.conf[,-7]
train.conf = round((train.conf/length(y_train))*100,2)
train.conf

print("Test Data Confusion Matrix (%)")
test.conf = confusionMatrix(model_pred,y_test)
test.conf = test.conf$table
test.conf = round((test.conf/length(y_test))*100,2)
test.conf

##Q12
#calculate mean change in accuracy
fZ = c(1:number_components)
redL = L[1:number_components]
model4.imp = model4$importance[,7]
model4.imp = cbind(fZ, model4.imp)
row.names(model4.imp) = NULL
redL = cbind(fZ, redL)
model4.imp.sorted = model4.imp[order(model4.imp[,2], decreasing = T),]

#plot effect of features on accuracy
plot(fZ, model4.imp[,2], xlab = "Features Z(j)", ylab = "Effect on Accuracy (%)", main = "Importance of Features")

#Q13
#percent difference between eigenvalues and random forest importance
PCA.rF.perc = t(t((model4.imp.sorted[,1]-redL[,1])/(number_components/100)))
abs.PCA.rF.perc = abs(PCA.rF.perc)
summary(abs.PCA.rF.perc)

#plot percent difference
plot(fZ, PCA.rF.perc, xlab = "Features Z(j)", ylab = "Difference (%)", main = "Feature Importance v Eigenvalue")