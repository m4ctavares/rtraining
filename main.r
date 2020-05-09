install.packages("dplyr")


rm(list=ls())
library(dplyr)
#######################################################################################
#######################################################################################
################################ Abrindo Base de Dados ################################
#######################################################################################
#######################################################################################

setwd("https://raw.githubusercontent.com/m4ctavares/rtraining/branch/ds_sample_200508.rds")
db <- readRDS("Dataset_aula_6.rds")

######################################################################
###################### Análise da Base de Dados ######################
######################################################################

names(db)

### Proporção de default (Não Pagou) na base de dados
table(db$default)
prop.table(table(db$default))

library(gmodels)
CrossTable(db$default)

# Magnitude da Dívida
CrossTable(x = db$loan_amount, y = db$default, prop.r = TRUE, prop.c = FALSE, prop.t = FALSE, prop.chisq = FALSE)

# Score de Crédito
CrossTable(x = db$credit_score, y = db$default, prop.r = TRUE, prop.c = FALSE, prop.t = FALSE, prop.chisq = FALSE)

######################################################################
############ Gerando a Base de Dados de Treino e de Teste ############
######################################################################

### Gerando db de treino:
set.seed(567)
library(rsample)
split <- initial_split(db, prop = 0.6)
db_Treino <- training(split)    
db_Teste <- testing(split)
rm(split)

#############################################################
#################### Regressão Logística ####################
#############################################################

CrossTable(db_Treino$default)

################
### Modelo 1 ###
################
Modelo_1 <- glm(default ~ loan_amount + credit_score,
                family = binomial(link = logit), data = db_Treino)

summary(Modelo_1)
exp(cbind(OR = coef(Modelo_1), confint(Modelo_1)))

# Aplicando o modelo na base de teste.
db_Teste$Modelo_1 <- predict(Modelo_1, newdata = db_Teste, type = "response")
hist(db_Teste$Modelo_1)
range(db_Teste$Modelo_1, na.rm = TRUE)

# Criando a variável predita com base no cutoff 0.15
db_Teste$Modelo_1_cutoff <- ifelse(db_Teste$Modelo_1 > 0.2, 1, 0)
table(db_Teste$default, db_Teste$Modelo_1_cutoff)

## Acurácia (?????)

## Sensibilidade (?????)

## Especificidade (?????)

################
### Modelo 2 ###
################
Modelo_2 <- glm(default ~ loan_amount + income + credit_score +
                bad_public_record + credit_utilization + past_bankrupt,
                family = binomial(link = logit), data = db_Treino)

summary(Modelo_2)
exp(cbind(OR = coef(Modelo_2), confint(Modelo_2)))

# Aplicando o modelo na base de teste.
db_Teste$Modelo_2 <- predict(Modelo_2, newdata = db_Teste, type = "response")
hist(db_Teste$Modelo_2)
range(db_Teste$Modelo_2, na.rm = TRUE)

# Criando a variável predita com base no cutoff 0.15
db_Teste$Modelo_2_cutoff <- ifelse(db_Teste$Modelo_2 > 0.2, 1, 0)
table(db_Teste$default, db_Teste$Modelo_2_cutoff)

## Acurácia

## Sensibilidade

## Especificidade


################
### Modelo 3 ###
################
Modelo_3 <- glm(default ~ loan_amount + emp_length + income + debt_to_income + credit_score + recent_inquiry +
                  delinquent + credit_accounts + bad_public_record + credit_utilization + past_bankrupt,
                    family = binomial(link = logit), data = db_Treino)

summary(Modelo_3)
exp(cbind(OR = coef(Modelo_3), confint(Modelo_3)))

# Aplicando o modelo na base de teste.
db_Teste$Modelo_3 <- predict(Modelo_3, newdata = db_Teste, type = "response")
hist(db_Teste$Modelo_3)
range(db_Teste$Modelo_3, na.rm = TRUE)

# Criando a variável predita com base no cutoff 0.15
db_Teste$Modelo_3_cutoff <- ifelse(db_Teste$Modelo_3 > 0.2, 1, 0)
table(db_Teste$default, db_Teste$Modelo_3_cutoff)

## Acurácia
mean(db_Teste$default == db_Teste$Modelo_3_cutoff)

## Sensibilidade
print(649/(649 + 1545))*100

## Especificidade
print(11496/(11496+2202))



#########################
####### Curva ROC #######
library(pROC)
ROC_Modelo_1 <- roc(db_Teste$default, db_Teste$Modelo_1)
ROC_Modelo_2 <- roc(db_Teste$default, db_Teste$Modelo_2)
ROC_Modelo_3 <- roc(db_Teste$default, db_Teste$Modelo_3)
plot(ROC_Modelo_1)
lines(ROC_Modelo_2, col="blue")
lines(ROC_Modelo_3, col="red")
auc(ROC_Modelo_1)
auc(ROC_Modelo_2)
auc(ROC_Modelo_3)
#############################################################
##################### Árvore de Decisão #####################
#############################################################
library(rpart)

Modelo_4 <- rpart(default ~ loan_amount + emp_length + income + debt_to_income + credit_score + recent_inquiry +
                      delinquent + credit_accounts + bad_public_record + credit_utilization + 
                      past_bankrupt, data = db_Treino, method = "class", control = rpart.control(cp = 0))

# Gerando o resultado previsto do modelo de Árvore de Decisão
db_Teste$Modelo_4 <- predict(Modelo_4, newdata = db_Teste, type = "class")

dim(db_Teste)
table(db_Teste$default, db_Teste$Modelo_4)

# Acurácia
mean(db_Teste$default == db_Teste$Modelo_4)
print((13398 + 94)/15892)*100

## Sensibilidade
print(94/(94+2100))*100

## Especificidade
print(13398/(13398+300))

## Vizualizando a Árvore
library(rpart.plot)
rpart.plot(Modelo_2)
rpart.plot(Modelo_2, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE)


#########################
####### Curva ROC #######
library(pROC)
ROC_Modelo_1 <- roc(db_Teste$default, db_Teste$Modelo_1)
ROC_Modelo_2 <- roc(db_Teste$default, db_Teste$Modelo_2)
ROC_Modelo_3 <- roc(db_Teste$default, db_Teste$Modelo_3)
ROC_Modelo_4 <- roc(db_Teste$default, as.numeric(db_Teste$Modelo_4))
plot(ROC_Modelo_1)
lines(ROC_Modelo_2, col="blue")
lines(ROC_Modelo_3, col="red")
lines(ROC_Modelo_4, col="green")
auc(ROC_Modelo_1)
auc(ROC_Modelo_2)
auc(ROC_Modelo_3)
auc(ROC_Modelo_4)

