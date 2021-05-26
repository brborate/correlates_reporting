#-----------------------------------------------
# obligatory to append to the top of each script
renv::activate(project = here::here(".."))

# There is a bug on Windows that prevents renv from working properly. The following code provides a workaround:
if (.Platform$OS.type == "windows") .libPaths(c(paste0(Sys.getenv ("R_HOME"), "/library"), .libPaths()))

source(here::here("..", "_common.R"))
#-----------------------------------------------

## load required libraries and functions
library(tidyverse)
library(here)
library(methods)
library(SuperLearner)
library(e1071)
library(glmnet)
library(kyotil)
library(argparse)
library(vimp)
library(nloptr)
library(RhpcBLASctl)
library(conflicted)
conflicted::conflict_prefer("filter", "dplyr")
conflict_prefer("summarise", "dplyr")

# Define code version to run
# the demo version is simpler and runs faster!
# the production version runs SL with a diverse set of learners
run_demo <- FALSE
run_prod <- TRUE

# get utility files
source(here("code", "sl_screens.R")) # set up the screen/algorithm combinations
source(here("code", "utils.R")) # get CV-AUC for all algs

# SL optimal surrogate analysis: Superlearner code requires computing environment with more than 10 cores!
num_cores <- parallel::detectCores()
if(num_cores < 10) stop("Number of cores on this computing environment are less than 10! Superlearner code needs atleast 11 cores to run smoothly.")

############ SETUP INPUT #######################
# Read in data file
inputFile <- read.csv(here::here("..", "data_clean", "practice_data_with_riskscore.csv"))
briskfactors <- c("risk_score", "HighRiskInd", "MinorityInd")

# Identify the endpoint variable
endpoint <- "EventIndPrimaryD57"
################################################    

# Create combined new dataset which has imputed values of demographics (for phase 1 data) from dat.covar.imp AND 
# imputed values for markers (for phase 2 data) from dat.wide.v
dat.ph1 <- inputFile %>%
  filter(Perprotocol == 1) %>%
  filter(Trt == 1) %>% # consider only vaccine group
  # Take in the risk scores
  # left_join(read.csv("../../SL_risk_score/output/vaccine_ptids_with_riskscores.csv") %>% select(Ptid, risk_score), by = "Ptid") %>%
  mutate(Delta57overBbindSpike_2fold = ifelse(Day57bindSpike > (BbindSpike + log10(2)), 1, 0),
         Delta57overBbindSpike_4fold = ifelse(Day57bindSpike > (BbindSpike + log10(4)), 1, 0),
         Delta57overBbindRBD_2fold = ifelse(Day57bindRBD > (BbindRBD  + log10(2)), 1, 0),
         Delta57overBbindRBD_4fold = ifelse(Day57bindRBD > (BbindRBD  + log10(4)), 1, 0),
         Delta57overBpseudoneutid50_2fold = ifelse(Day57pseudoneutid50 > (Bpseudoneutid50  + log10(2)), 1, 0), 
         Delta57overBpseudoneutid50_4fold = ifelse(Day57pseudoneutid50 > (Bpseudoneutid50  + log10(4)), 1, 0), 
         Delta57overBpseudoneutid80_2fold = ifelse(Day57pseudoneutid80 > (Bpseudoneutid80  + log10(2)), 1, 0), 
         Delta57overBpseudoneutid80_4fold = ifelse(Day57pseudoneutid80 > (Bpseudoneutid80  + log10(4)), 1, 0), 
         Delta57overBliveneutmn50_2fold = ifelse(Day57liveneutmn50 > (Bliveneutmn50  + log10(2)), 1, 0), 
         Delta57overBliveneutmn50_4fold = ifelse(Day57liveneutmn50 > (Bliveneutmn50  + log10(4)), 1, 0),
         Delta57overBliveneutmn50 = Day57liveneutmn50 / Bliveneutmn50) %>%
  # Drop any observation with NA values in Ptid, Trt, or endpoint!
  drop_na(Ptid, Trt, all_of(briskfactors), all_of(endpoint), wt.D57) %>%
  #filter(!is.na(Ptid), !is.na(Trt), !is.na(risk_score), !is.na(Age), !is.na(EventIndPrimaryD57), !is.na(wt)) %>%
  arrange(desc(get(endpoint)))


dat.ph2 = dat.ph1 %>%
  filter(TwophasesampIndD57==1) %>%
  # Baseline Risk Factor includes only BRiskScore and Age (UPDATE THIS as mentioned in SAP)
  select(Ptid, Trt, all_of(briskfactors), all_of(endpoint), wt.D57,
         Day57bindSpike, Delta57overBbindSpike, Delta57overBbindSpike_2fold, Delta57overBbindSpike_4fold,
         Day57bindRBD, Delta57overBbindRBD, Delta57overBbindRBD_2fold, Delta57overBbindRBD_4fold,
         Day57pseudoneutid50, Delta57overBpseudoneutid50, Delta57overBpseudoneutid50_2fold, Delta57overBpseudoneutid50_4fold,
         Day57pseudoneutid80, Delta57overBpseudoneutid80, Delta57overBpseudoneutid80_2fold, Delta57overBpseudoneutid80_4fold,
         Day57liveneutmn50, Delta57overBliveneutmn50, Delta57overBliveneutmn50_2fold, Delta57overBliveneutmn50_4fold) %>%
  filter(!is.na(Day57bindSpike), !is.na(Day57bindRBD), !is.na(Day57pseudoneutid50), !is.na(Day57pseudoneutid80), !is.na(Day57liveneutmn50)) %>%
  arrange(desc(get(endpoint)))

# Limit total variables that will be included in models 
nv <- sum(dat.ph2 %>% select(matches(endpoint)))
# maxVar <- max(20, floor(nv / 20))

# Save ptids to merge with predictions later
ph1_vacc_ptids <- dat.ph1 %>% select(Ptid, all_of(endpoint))

Z_plus_weights <- dat.ph1 %>% 
  select(Ptid, all_of(endpoint), wt.D57, Trt, all_of(briskfactors)) %>%
  #mutate(ptid = as.numeric(ptid)) %>% 
  # Drop any observation with NA values in Ptid, Trt, briskfactors, endpoint or wt.D57!
  drop_na(Ptid, Trt, all_of(briskfactors), all_of(endpoint), wt.D57) 
  
###########################################################################
# Create combination scores across the 6 markers
dat.ph2 <- dat.ph2 %>% 
  left_join(get.pca.scores(dat.ph2 %>%
                             select(Ptid, Day57bindSpike, Day57bindRBD, Day57pseudoneutid50, Day57pseudoneutid80, Day57liveneutmn50)), 
            by = "Ptid") %>%
  # left_join(get.nonlinearPCA.scores(dat.ph2 %>%
  #                                     select(Ptid, Day57bindSpike, Day57bindRBD, Day57pseudoneutid50, Day57pseudoneutid80, Day57liveneutmn50)),
  #          by = "Ptid") %>%
  mutate(nlPCA1 = PC1, 
         nlPCA2 = PC2,
         max.signal.div.score = get.maxSignalDivScore(dat.ph2 %>%
                                                        select(Day57bindSpike, Day57bindRBD, Day57pseudoneutid50, Day57pseudoneutid80, Day57liveneutmn50)))

markers <- dat.ph2 %>%
  select(Day57bindSpike:max.signal.div.score) %>%
  colnames()

#####################################################################################################################
## Create variable sets and set up X, Y for super learning
# Maternal enrollment variables are default in all sets

# 1. None (No markers; only maternal enrollment variables), phase 1 data
varset_baselineRiskFactors <- rep(FALSE, length(markers))

# 2-12
varset_bAbSpike <- create_varsets(markers, c("Day57bindSpike", "Delta57overBbindSpike", "Delta57overBbindSpike_2fold", "Delta57overBbindSpike_4fold"))
varset_bAbRBD <- create_varsets(markers, c("Day57bindRBD", "Delta57overBbindRBD", "Delta57overBbindRBD_2fold", "Delta57overBbindRBD_4fold"))
varset_pnabID50 <- create_varsets(markers, c("Day57pseudoneutid50", "Delta57overBpseudoneutid50", "Delta57overBpseudoneutid50_2fold", "Delta57overBpseudoneutid50_4fold"))
varset_pnabID80 <- create_varsets(markers, c("Day57pseudoneutid80", "Delta57overBpseudoneutid80", "Delta57overBpseudoneutid80_2fold", "Delta57overBpseudoneutid80_4fold"))
varset_lnabMN50 <- create_varsets(markers, c("Day57liveneutmn50", "Delta57overBliveneutmn50", "Delta57overBliveneutmn50_2fold", "Delta57overBliveneutmn50_4fold"))
varset_bAb_pnabID50 <- create_varsets(markers, c("Day57bindSpike", "Delta57overBbindSpike", "Delta57overBbindSpike_2fold", "Delta57overBbindSpike_4fold",
                                                 "Day57bindRBD", "Delta57overBbindRBD", "Delta57overBbindRBD_2fold", "Delta57overBbindRBD_4fold",
                                                 "Day57pseudoneutid50", "Delta57overBpseudoneutid50", "Delta57overBpseudoneutid50_2fold", "Delta57overBpseudoneutid50_4fold"))
varset_bAb_pnabID80 <- create_varsets(markers, c("Day57bindSpike", "Delta57overBbindSpike", "Delta57overBbindSpike_2fold", "Delta57overBbindSpike_4fold",
                                                 "Day57bindRBD", "Delta57overBbindRBD", "Delta57overBbindRBD_2fold", "Delta57overBbindRBD_4fold",
                                                 "Day57pseudoneutid80", "Delta57overBpseudoneutid80", "Delta57overBpseudoneutid80_2fold", "Delta57overBpseudoneutid80_4fold"))
varset_bAb_lnabMN50 <- create_varsets(markers, c("Day57bindSpike", "Delta57overBbindSpike", "Delta57overBbindSpike_2fold", "Delta57overBbindSpike_4fold",
                                                 "Day57bindRBD", "Delta57overBbindRBD", "Delta57overBbindRBD_2fold", "Delta57overBbindRBD_4fold",
                                                 "Day57liveneutmn50", "Delta57overBliveneutmn50", "Delta57overBliveneutmn50_2fold", "Delta57overBliveneutmn50_4fold"))
varset_bAb_combScores <- create_varsets(markers, c("Day57bindSpike", "Delta57overBbindSpike", "Delta57overBbindSpike_2fold", "Delta57overBbindSpike_4fold",
                                                   "Day57bindRBD", "Delta57overBbindRBD", "Delta57overBbindRBD_2fold", "Delta57overBbindRBD_4fold",
                                                   "PC1", "PC2", "nlPCA1", "nlPCA2", "max.signal.div.score"))
varset_allMarkers <- create_varsets(markers, markers[1:20])
varset_allMarkers_combScores <- create_varsets(markers, markers[1:25])


varset_names <- c("1_baselineRiskFactors", 
                  "2_varset_bAbSpike", "3_varset_bAbRBD", "4_varset_pnabID50", "5_varset_pnabID80", "6_varset_lnabMN50", 
                  "7_varset_bAb_pnabID50", "8_varset_bAb_pnabID80", "9_varset_bAb_lnabMN50", 
                  "10_varset_bAb_combScores", "11_varset_allMarkers", "12_varset_allMarkers_combScores")

## set up a matrix of all 
varset_matrix <- rbind(varset_baselineRiskFactors, 
                       varset_bAbSpike, varset_bAbRBD, varset_pnabID50, varset_pnabID80, varset_lnabMN50, 
                       varset_bAb_pnabID50, varset_bAb_pnabID80, varset_bAb_lnabMN50, 
                       varset_bAb_combScores, varset_allMarkers, varset_allMarkers_combScores)

job_id <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
job_id <- 2
this_var_set <- varset_matrix[job_id, ]
cat("\n Running ", varset_names[job_id], "\n")


X_covars2adjust_ph2 <- dat.ph2 %>% select(all_of(c(briskfactors, markers)))   

## scale covars2adjust to have mean 0, sd 1 for all vars
for (a in colnames(X_covars2adjust_ph2)) {
  X_covars2adjust_ph2[[a]] <- scale(X_covars2adjust_ph2[[a]],
                                    center = mean(X_covars2adjust_ph2[[a]], na.rm = T), 
                                    scale = sd(X_covars2adjust_ph2[[a]], na.rm = T))    
}


markers_start <- length(briskfactors) + 1
X_markers_varset <- bind_cols(X_covars2adjust_ph2[1:length(briskfactors)], 
                              X_covars2adjust_ph2[markers_start:length(X_covars2adjust_ph2)][this_var_set]) %>%
  select_if(function(x) any(!is.na(x))) # Drop column if it has 0 variance, and returned all NAN's from scale function. 
Y = dat.ph2 %>% pull(endpoint)
weights = dat.ph2$wt.D57
sl_lib <- SL_library

treatmentDAT <- dat.ph2 %>% select(Ptid, Trt, wt.D57, EventIndPrimaryD57, all_of(c(briskfactors, markers))) %>%
  filter(Trt == 1) %>%
  select(-Trt)

# match the rows in treatmentDAT to get Z, C
all_cc_treatment <- Z_plus_weights %>%
  filter(Ptid %in% treatmentDAT$Ptid)
# pull out the participants who are NOT in the cc cohort and received the vaccine
all_non_cc_treatment <- Z_plus_weights %>%
  filter(!(Ptid %in% treatmentDAT$Ptid))
# put them back together
phase_1_data_treatmentDAT <- dplyr::bind_rows(all_cc_treatment, all_non_cc_treatment) %>%
  select(-Trt)
Z_treatmentDAT <- phase_1_data_treatmentDAT %>%
  select(-Ptid, -wt.D57)
all_ipw_weights_treatment <- phase_1_data_treatmentDAT %>%
  pull(wt.D57)
C <- (phase_1_data_treatmentDAT$Ptid %in% treatmentDAT$Ptid)

## set up outer folds for cv variable importance; do stratified sampling
V_outer <- 5
if (sum(dat.ph2$EventIndPrimaryD57) <= 25){
  V_inner <- length(Y) - 1
  maxVar <- 5
  
} else if(sum(dat.ph2$EventIndPrimaryD57) > 25){
  V_inner <- 5
  maxVar <- floor(nv/6)
  #V_inner <- length(Y) - 1
}
  

## ---------------------------------------------------------------------------------
## run super learner, with leave-one-out cross-validation and all screens
## do 10 random starts, average over these
## use assay groups as screens
## ---------------------------------------------------------------------------------
## ensure reproducibility
set.seed(20201202)
seeds <- round(runif(10, 1000, 10000)) # average over 10 random starts
#seeds <- round(runif(1, 1000, 10000))

##solve cores issue
library(RhpcBLASctl)
blas_get_num_procs()
blas_set_num_threads(1)
print(blas_get_num_procs())
stopifnot(blas_get_num_procs()==1)

fits <- parallel::mclapply(seeds, FUN = run_cv_sl_once, 
                           Y = Y, 
                           X_mat = X_markers_varset, 
                           family = "binomial",
                           obsWeights = weights,
                           all_weights = all_ipw_weights_treatment,
                           ipc_est_type = "ipw",
                           sl_lib = sl_lib,
                           method = "method.CC_nloglik",
                           cvControl = list(V = V_outer, stratifyCV = TRUE),
                           innerCvControl = list(list(V = V_inner)),
                           Z = Z_treatmentDAT, 
                           C = C, 
                           # z_lib = c("SL.glm", "SL.bayesglm", "SL.step", "SL.gam","SL.cforest"), # new arguments
                           z_lib = "SL.glm",
                           scale = "identity", # new argument
                           vimp = FALSE,
                           mc.cores = num_cores
)

cvaucs <- list()
cvfits <- list()

for(i in 1:length(seeds)) {
  cvaucs[[i]] = fits[[i]]$cvaucs$aucs
  cvfits[[i]] = fits[[i]]$cvfits
}

saveRDS(cvaucs, file = here("output", paste0("CVSLaucs_vacc_", endpoint, "_", varset_names[job_id], ".rds")))
save(cvfits, file = here("output", paste0("CVSLfits_vacc_", endpoint, "_", varset_names[job_id], ".rda")))
save(ph1_vacc_ptids, file = here("output", "ph1_vacc_ptids.rda"))
save(run_prod, Y, dat.ph1, dat.ph2, weights, inputFile, briskfactors, endpoint, maxVar,
     V_outer, file = here("output", "objects_for_running_SL.rda"))

# save(risk_placebo_ptids, file = here("output", "risk_placebo_ptids.rda"))
# save(run_prod, Y, X_riskVars, weights, inputFile, risk_vars, endpoint, maxVar,
#      V_outer, file = here("output", "objects_for_running_SL.rda"))


# X_mat = X_markers_varset
# family = "binomial"
# Z = Z_treatmentDAT
# z_lib = c("SL.glm", "SL.bayesglm", "SL.step", "SL.gam","SL.cforest")
# z_lib = "SL.glm"
# obsWeights = weights
# all_weights = all_ipw_weights_treatment
# scale = "identity"
# ipc_est_type = "ipw"
# method = "method.CC_nloglik"
# cvControl = list(V = V_outer, stratifyCV = TRUE)
# innerCvControl = list(list(V = V_inner))
# vimp = FALSE
# mc.cores = num_cores
