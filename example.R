
library(pCMF)

set.seed(250)

## generate data
n <- 100
p <- 500
K <- 20
factorU <- generate_factor_matrix(n, K, ngroup=2,
                                  average_signal=120,
                                  group_separation=0.8,
                                  distribution="exponential",
                                  shuffle_feature=TRUE)
factorV <- generate_factor_matrix(p, K, ngroup=2, average_signal=120,
                                  group_separation=0.8,
                                  distribution="exponential",
                                  shuffle_feature=TRUE,
                                  prop_noise_feature=0.6)
U <- factorU$factor_matrix
V <- factorV$factor_matrix
count_data <- generate_count_matrix(n, p, K, U, V,
                                    ZI=TRUE, prob1=rep(0.4,p))
X <- count_data$X
## or use your own data as a count matrix
## of dimension cells x genes (individuals x features)

## Heatmap of the count data matrix:
matrix_heatmap(X)

## Pre-filtering of the variables (genes)
kept_cols <- prefilter(X, prop = 0.05, quant_max = 0.9,
                       presel = TRUE, threshold = 0.2)
# or use the variance-based heuristic for  variable/gene pre-selection
kept_cols <- prefilter(X, prop = 0.05, quant_max = 0.9,
                       presel = TRUE, threshold = 0.2)

## Remove variables that were filtered out:
X <- X[,kept_cols]

## Apply the pCMF approach on the data with $K=2$
# run pCMF algorithm
res1 <- pCMF(X, K=2, verbose=FALSE, zero_inflation = TRUE,
             sparsity = TRUE, ncores=8) # edit nb of cores

## Check which variables (genes) contribute to the low-dimensional
## representation (non-null entries in $V$), and re-apply the method with this
## genes

# estimated probabilities
matrix_heatmap(res1$sparse_param$prob_S)
# corresponding indicator (prob_S > threshold, where threshold = 0.5)
matrix_heatmap(res1$sparse_param$S)
# rerun with genes that contributes
res2 <- pCMF(X[,res1$sparse_param$prior_prob_S>0],
             K=2, verbose=FALSE, zero_inflation = TRUE,
             sparsity = FALSE, ncores=8)

## Get estimated factor matrices: $\hat{U}$ and $\hat{V}$ can be used for
## clustering of individuals (cells) or variables (genes):
hatU <- getU(res2)
hatV <- getV(res2)

## Data visualization**:
# individual (cell) representation
graphU(res2, axes=c(1,2), labels=factorU$feature_label)
# variable (gene) representation (0 are noise variables), sparse representation
graphV(res1, axes=c(1,2),
       labels=factorV$feature_label[kept_cols])
# variable representation (0 are noise variables) after gene selection
graphV(res2, axes=c(1,2),
       labels=factorV$feature_label[kept_cols][res1$sparse_param$prior_prob_S>0])

