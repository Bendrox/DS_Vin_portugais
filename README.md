# DS_Vin_portugais

Proposition de correction d'un exercice en machine learning donné lors de l'examen final 2023 au centre de formation continue ENSAE-ENSAE.

La correction proposée est une extraction sans modification de la copie rendue.

Le langage utilisé : R

![image](https://github.com/Bendrox/DS_Vin_portugais/assets/145064474/c0ccaf34-3b35-464f-bd47-c320a216f385)


library(readr)
library(dplyr)
library(foreign) 
library(ranger)
library(rpart)
library(inspectdf)
library(kknn)
library(data.table)
library(corrplot)
library(factoextra)
library(FactoMineR)
library(tidymodels)
library(tidyr)
library(tidyverse)
library(ROSE)
library(themis)
library(FactoMineR)

### Import data  
setwd("C:/Users/OussA/Downloads")
df <- read_rds("vin.rds")

# Inspect df
sum(is.na(df)) # ok pas de NA 
df%>% inspect_num() %>% show_plot() # ok idée global 
glimpse(df) # ok pas de transfo a faire 

# Check équilibre Y
prop.table(table(df$quality)) # ok équilibre 
levels(df$quality) <- c("No", "Yes") # je prefere No / Yes 

### Analyse descriptive 
df_x <- df %>% select(-quality) #df sans Y pour la suite

### Correlation des variables 
corrplot(cor(df_x),method="circle",type="upper") 

# Classement des variables en fonction de la corr
df_cor <- df_x %>% cor %>% melt %>% arrange(desc(value)) 
df_cor_unique <- df_cor[!duplicated(df_cor$value), ]
df_cor_unique %>% filter(abs(value)> 0.6 & abs(value) <1) # les variables les + correlées 

# ACP 
res.acp <- PCA(df_x)
fviz_eig(res.acp, addlabels = TRUE, ylim = c(0, 50)) #pr choix nbr de dim
var <- get_pca_var(res.acp)
corrplot(var$cos2, is.corr=FALSE) #qualité de projection des variables 
corrplot(var$contrib, is.corr=FALSE) # contribution aux dim

# Classif non supervisée 1 : CAH
res.hcpc <- HCPC(res.acp, nb.clust = 3) # 3 fixé après visualisation 
res.hcpc$data.clust # données + classes CAH.
res.hcpc$desc.var   # description des classes avec les variables 
res.hcpc$desc.ind   # description des classes avec les individus

# Classif non supervisée 2 : kmeans 
#trouver nombre de classes optimal 
n_classe_max <- 15
part_inter <- rep(0,time=n_classe_max) # var vide : 15 x 0

for(i in 1:n_classe_max){
  km <- kmeans(df_x,centers=i,nstart=20)
  part_inter[i] <- km$betweenss/km$totss*100  #inertie inter =
  #The between-cluster sum of squares /
  #totss	= The total sum of squares.
}

df_kmean <- data.frame(classe=1:n_classe_max, part_inter=part_inter)

# Plot kmeans en fonction des inerties inter classes 
ggplot(data=df_kmean,aes(x=classe,y=part_inter))+
  geom_bar(stat="identity",fill="steelblue")+
  scale_x_continuous(breaks=1:n_classe_max)+
  labs(x="Classe",y="Part d'inertie inter-classes",title="K-means")
#Plus grand saut entre 2 et 3 : on retient 3 classes et reboucle avec CAH.
head(df_x)

# Exploitation du kmeans 
nclust <- 3 # nombre de classes retenu
km <- kmeans(df_x,centers=nclust) #kmean avec le nmbr de classes
names(km) 
clusters <- as.factor(km$cluster) # la classe attribuée
table(clusters) # classes + effectif 
centre <- as.data.frame(km$centers) # centre des classes 
head(df_x)
ggplot()+
  geom_point(data=df_x,aes(x=fixed.acidity,y=total.sulfur.dioxide,colour=clusters))+
  geom_point(data=centre,aes(x=fixed.acidity,y=total.sulfur.dioxide),shape=15,size=3)+
  labs(color="Classe",title="K-means")
  
# On arrive a visionner les 3 classes qui se dégagent en fonction des deux variables les plus contributrices aux 2 1eres dim de l'ACP

## Modélisation avc tidymodels
# Phase 1: Decoupage   
#Train/test
df<- rename(df, target = quality) #renom quality en target 
colnames(df) # ok transfo
set.seed(1234)
ini_split <- initial_split(df, 
                           prop = 0.9,
                           strata= target)
df_train <- ini_split %>% training()
df_test <- ini_split %>% testing()
ini_split

# CV param 
set.seed(1234)
df_folds <- vfold_cv(df_train, v = 10,
                     strata = target)
# metrics eval  
metrics <-metric_set(accuracy, roc_auc, f_meas)

# Phase 2: Recipe (partie FE)
# 4 reccettes FE 
rec_basic <- recipe(data = df_train, target~.) %>%
  step_normalize(all_numeric_predictors())

rec_interac  <- rec_basic %>% # rajouter les interractions 
  step_interact(~fixed.acidity : citric.acid) %>%
  step_interact(~ total.sulfur.dioxide : free.sulfur.dioxide)

rec_inter_spline <- rec_interac %>% #rajouter les splines aux interract
  step_ns( fixed.acidity : citric.acid, deg_free = tune())

rec_spline_pur <- rec_basic %>% #faire des splines sans interractions 
  step_ns( fixed.acidity : citric.acid, deg_free = tune())

# Phase 3: Algos (v.tunable)
# Algo 1: lasso 
dt_tune_model_lasso <- logistic_reg( penalty = tune(),mixture = 1 ) %>% 
  set_engine('glmnet') %>%
  set_mode('classification')
# Algo 2:ridge 
dt_tune_model_ridge <- logistic_reg( penalty = tune(),mixture = 0 ) %>% 
  set_engine('glmnet') %>%
  set_mode('classification')
# Algo 3:elastic net
dt_tune_model_elastic <- logistic_reg( penalty = tune(),mixture = 0.5 ) %>% 
  set_engine('glmnet') %>%
  set_mode('classification')
# Algo 4: Arbres 
tunable_tree <- decision_tree(cost_complexity = tune(),tree_depth = tune(),
                              min_n = tune()) %>% 
  set_engine('rpart') %>% 
  set_mode('classification')
# Algo 5: forets  
tunable_randomfor <- rand_forest(trees = tune(), min_n = tune())%>%
  set_engine('ranger')%>%
  set_mode('classification')

# Algo 6: XG boost 
xgboost_model_tune <- 
  boost_tree(trees = tune(), min_n = tune(), tree_depth = tune(), 
             learn_rate = tune(), loss_reduction = tune()) %>% 
  set_engine('xgboost') %>% 
  set_mode('classification')

# Algo 7: KNN  
knn_tune <- nearest_neighbor(neighbors = tune(),
                             dist_power = tune(),
                             weight_func = tune())%>%
  set_engine('kknn')%>%
  set_mode('classification')

# Algo 8: SVM Polynomial 
svm_p_spec_tune <- svm_poly(cost = tune(), degree = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

# Algo 9: SVM Radial
svm_r_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# Algo 10: Single layer neural network
nnet_tune <-mlp(penalty = tune(), epochs = tune())%>%
  set_engine('nnet')%>%
  set_mode('classification')

# Phase 4: regroupe algo (10) x recettes FE (4) -> 40 combinaisons !
wf_set <-workflow_set(
  preproc = list(     # on liste les recettes (partie FE)
    basic = rec_basic, 
    inter = rec_interac,
    splines_pur = rec_spline_pur, 
    InterSpline = rec_inter_spline),
  models = list( 
    # on liste les modèles dans leur v. tunable
    laso =dt_tune_model_lasso ,
    ridge= dt_tune_model_ridge,
    lastic = dt_tune_model_elastic,
    XGboost =xgboost_model_tune ,
    RF= tunable_randomfor,
    Tree= tunable_tree,
    Knn =knn_tune,
    #Neuron = nnet_tune, # ne gere pas trop de variables
    SVM_p = svm_p_spec_tune,
    SVM_R = svm_r_spec))

# enregristrer les recherches  
keep_pred <- control_resamples( # permet de rajouter des param
  save_pred = T, # enreg des prévisions pour chaque modele x recipe
  save_workflow = T) # enreg wflr pour chaque modele x recette

## Phase 5: tunage

# paramètrage du workflow_set (10 algo x 4recipes x 20 valos)
set.seed(123)
res_wf_set <- wf_set%>%
  workflow_map(resamples = df_folds, #Input découpage pour VC
               fn = "tune_grid",    
               grid = 5, #Ici parametre pour grille de tunage !
               # -> pour chaque modèle X workflow il essaye n param 
               metrics = metric_set(roc_auc),  # On met deux criteres
               control = keep_pred, #Input fonction enregis des résultats
               verbose = T)

res_rank_auc <- rank_results(res_wf_set, rank_metric = 'roc_auc') %>%
  filter(.metric =="roc_auc")
head(res_rank_auc, n=5)
#view(res_rank_auc) # Idem avant en fonction RSQ

## Phase 6: Finalisation
#choix meilleur classe d'algo
best_result <- 
  res_wf_set %>%
  extract_workflow_set_result('inter_RF')%>% #Input ici meilleur MODELE trouvé
  select_best(metric = 'roc_auc') # en fonction de roc_auc

best_result

##Paramètrage final sur tout le train set  et pred. sur test
best_res_fit <-res_wf_set %>%
  extract_workflow('inter_RF')%>% # extract modele et la recette 
  finalize_workflow(best_result)%>% # ajuste  param
  last_fit(split = ini_split) # adaptation au bloc test

#voir pref sur bloc test 
best_res_fit%>%collect_metrics() 
#

# Nous pourrions augmenter la grille de recherche étant donné que les perfs 
# sur test sont suppérieurs a celles sur le test mais attention au surapprentissage.
