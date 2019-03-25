### Copyright 2018 Ghislain DURIF
###
### This file is part of the `pCMF' library for R and related languages.
### It is made available under the terms of the GNU General Public
### License, version 2, or at your option, any later version,
### incorporated herein by reference.
###
### This program is distributed in the hope that it will be
### useful, but WITHOUT ANY WARRANTY; without even the implied
### warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
### PURPOSE.  See the GNU General Public License for more
### details.
###
### You should have received a copy of the GNU General Public
### License along with this program; if not, write to the Free
### Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
### MA 02111-1307, USA

#' R package pCMF
#'
#' See \url{https://gitlab.inria.fr/gdurif/pCMF}
#'
#' @description
#' The pCMF package implements algorithms for factorization of count matrices
#' based on probabilistic models, especially Gamma-Poisson factor models (with
#' variants including zero-inflation and sparsity). It was specifically
#' designed to analyse single-cell gene expression data (scRNA-seq).
#'
#' @details
#' More details can be found in Durif et al. (2017). Matrix factorization is an
#' unsupervised dimension reduction approach, suitable for data visualization
#' or as a preliminary step for clustering.
#'
#' @author
#' Ghislain Durif \email{gd.dev@libertymail.net}
#'
#' Maintainer: Ghislain Durif \email{gd.dev@libertymail.net}
#'
#' @references
#' Durif, G., Modolo, L., Mold, J.E., Lambert-Lacroix, S., Picard, F., 2017.
#' Probabilistic Count Matrix Factorization for Single Cell Expression Data
#' Analysis. arXiv:1710.11028 [stat].
#'
#' @name pCMF.package
#' @title pCMF.package
#' @docType package
NULL

#' @title pCMF
#' @name pCMF
#'
#' @description
#' Implementation of the probabilistic Count Matrix Factorization (pCMF)
#' method based on the Gamma-Poisson hirerarchical factor model with
#' potential sparisty-inducing priors on factor V. This method
#' is specifically designed to analyze large count matrices with numerous
#' potential drop-out events (also called zero-inflation) such as gene
#' expression profiles from single cells data (scRNA-seq) obtained by
#' high throughput sequencing.
#'
#' This function is a wrapper to call the different versions of pCMF detailed
#' in Durif et al. (2017): \code{\link{run_gap_factor}},
#' \code{\link{run_zi_gap_factor}}, \code{\link{run_sparse_gap_factor}},
#' \code{\link{run_zi_sparse_gap_factor}}.
#'
#' @details
#' In the probabilistic Count Matrix Factorization framework (pCMF), the
#' count data matrix \eqn{X} (dim \eqn{n \times p}) is approximated
#' by a matrix product \eqn{U V^t} where \eqn{U_{n\times K}} and
#' \eqn{V_{p\times K}} respectively represent the individuals
#' (rows of \eqn{X}) and variables (cols of \eqn{X})
#' in a sub-space of dimension \eqn{K}.
#'
#' In the pCMF framework, the approximation between \eqn{X} and
#' \eqn{U V^t} is made regarding the Kullback-Leibler divergence, which
#' corresponds to the Bregman divergence derived from the model
#' \eqn{X \sim P(U V^t)}, i.e. each entry \eqn{X_{ij}} is assumed to follow
#' a Poisson distribution of parameter \eqn{\sum_k U_{ik} V_{jk}}.
#' In addition, we consider a hierarchical model with prior distributions on
#' factors \eqn{U} and \eqn{V}. Our probabilistic model is able to account
#' for zero-inflation (potential drop-out events) in the data matrix
#' \eqn{X} and/or for sparsity in the factor matrix \eqn{V}.
#'
#' More details regarding pCMF can be found in Durif et al. (2017).
#'
#' Details about output, depending on input parameter values:
#'
#' \tabular{lcc}{
#' \tab \code{zero_inflation=TRUE} \tab \code{zero_inflation=FALSE} \cr
#' \code{sparsity=TRUE} \tab see (1) \tab see (2) \cr
#' \code{sparsity=FALSE} \tab see (3) \tab see (4)}
#'
#' (1) \code{\link{run_zi_sparse_gap_factor}}
#' (2) \code{\link{run_sparse_gap_factor}}
#' (3) \code{\link{run_zi_gap_factor}}
#' (4) \code{\link{run_gap_factor}}
#'
#' After running our method with \code{sparsity=TRUE}, it is recommended to
#' check which variables/columns/genes contribute to the representation (by
#' checking \code{$sparse_param$prob_S} and \code{$sparse_param$prior_prob_S}
#' in the output), and apply the same pCMF with \code{sparsity=FALSE} on the
#' data where you removed the variables/columns/genes that did not contribute
#' to the representation (e.g. variables/columns/genes corresponding to
#' null entries in \code{$sparse_param$prior_prob_S}).
#'
#' @references
#' Durif, G., Modolo, L., Mold, J.E., Lambert-Lacroix, S., Picard, F., 2017.
#' Probabilistic Count Matrix Factorization for Single Cell Expression Data
#' Analysis. arXiv:1710.11028 [stat].
#'
#' @author
#' Ghislain Durif, \email{gd.dev@libertymail.net}
#'
#' @param X a count data matrix of dimension \code{n x p}.
#' @param K integer, required dimension of the subspace for the latent
#' representation.
#' @param zero_inflation boolean indicating if the considered model accounts
#' for zero-inflation. If TRUE (default), \code{pCMF} algorithm is based on a
#' Gamma-Poisson factor model with zero-inflation (see Durif et al. 2017) and
#' if FALSE on a standard Gamma-Poisson factor model.
#' @param sparsity boolean indicating if the considered model accounts
#' for sparsity in factor \code{V}. If TRUE (default), \code{pCMF} algorithm is
#' based on a Gamma-Poisson factor model with sparsity on \code{V}
#' (see Durif et al. 2017) and if FALSE on a standard Gamma-Poisson
#' factor model.
#' @param sel_bound real value in [0,1] used to threshold sparsity
#' probabilities for factor V, used only if \code{sparsity=TRUE}. Default
#' value is 0.5.
#' @param verbose boolean indicating verbosity. Default is TRUE.
#' @param monitor boolean indicating if model related measures
#' (log-likelihood, deviance, Bregman divergence between \eqn{X}
#' and \eqn{UV^t}) should be computed. Default is TRUE.
#' @param iter_max integer, maximum number of iterations after which the
#' optimization is stopped even when the algorithm did not converge.
#' Default is 1000.
#' @param iter_min integer, minimum number of iterations enforced even if the
#' the algorithm converge. Default is NULL, and \code{iter_max/2} is used.
#' @param ninit integer, number of initialization to consider. In multiple
#' initialization mode (>1), the algorithm is run for \code{iter_init}
#' iterations with mutliple seeds and the best one (regarding the optimization
#' criterion) is kept. Default value is 1.
#' @param iter_init integer, number of iterations during which the algorithms
#' is run in multi-initialization mode. Default value is 100.
#' @param ncores integer indicating the number of cores to use for
#' parallel computation. Default is 1 and no multi-threading is used.
#' @param reorder_factor boolean indicating if factors should be reordered
#' according to the model-related deviance criterion. Default value is TRUE.
#' @param seed positive integer, seed value for random number generator.
#' Default is NULL and the seed is set based on the current time.
#'
#' @import Rcpp
#' @import RcppEigen
#' @importFrom Rcpp evalCpp
#' @useDynLib pCMF, .registration = TRUE
#'
#' @seealso \code{\link{run_gap_factor}}, \code{\link{run_zi_gap_factor}},
#' \code{\link{run_sparse_gap_factor}}, \code{\link{run_zi_sparse_gap_factor}}
#'
#' @return list of pCMF output, depending on the input parameter, see details.
#'
#' @examples
#' \dontrun{
#' ## generate data
#' n <- 100
#' p <- 500
#' K <- 20
#' factorU <- generate_factor_matrix(n, K, ngroup=3,
#'                                   average_signal=c(250,100,250),
#'                                   group_separation=0.8,
#'                                   distribution="exponential",
#'                                   shuffle_feature=TRUE)
#' factorV <- generate_factor_matrix(p, K, ngroup=2, average_signal=80,
#'                                   group_separation=0.8,
#'                                   distribution="exponential",
#'                                   shuffle_feature=TRUE,
#'                                   prop_noise_feature=0.6)
#' U <- factorU$factor_matrix
#' V <- factorV$factor_matrix
#' count_data <- generate_count_matrix(n, p, K, U, V,
#'                                     ZI=TRUE, prob1=rep(0.3,p))
#' X <- count_data$X
#' ## or use your own data as a count matrix
#' ## of dimension cells x genes (individuals x features)
#'
#' ## pre-filtering
#' kept_cols <- prefilter(X, prop = 0.05, quant_max = 0.95,
#'                        presel = TRUE, threshold = 0.2)
#' X <- X[,kept_cols]
#' ## run pCMF algorithm
#' res <- pCMF(X, K, verbose=FALSE, zero_inflation = TRUE, sparsity = TRUE)
#' ## rerun with genes that contributes
#' res <- pCMF(X[,res$sparse_param$prior_prob_S>0],
#'             K, verbose=FALSE, zero_inflation = TRUE, sparsity = FALSE)
#' }
#'
#' @export
pCMF <- function(X, K,
                 zero_inflation = TRUE, sparsity = TRUE, sel_bound = 0.5,
                 verbose = TRUE, monitor = TRUE,
                 iter_max = 1000, iter_min = NULL, ninit = 1, iter_init = 100,
                 ncores = 1, reorder_factor = TRUE, seed = NULL) {
    res <- NULL
    if(is.null(iter_min)) {
        iter_min <- as.integer(iter_max / 2)
    }
    if(!zero_inflation & !sparsity) {
        res <- run_gap_factor(X, K, verbose = verbose,
                              monitor = monitor, iter_max = iter_max,
                              iter_min = iter_min,
                              init_mode = "random",
                              epsilon = 1e-2, additional_iter = 10L,
                              conv_mode = 1L, ninit = ninit,
                              iter_init = iter_init, ncores = ncores,
                              reorder_factor = reorder_factor,
                              seed = seed)
    } else if(zero_inflation & !sparsity) {
        res <- run_zi_gap_factor(X, K, verbose = verbose,
                                 monitor = monitor, iter_max = iter_max,
                                 iter_min = iter_min,
                                 init_mode = "random",
                                 epsilon = 1e-2, additional_iter = 10L,
                                 conv_mode = 1L, ninit = ninit,
                                 iter_init = iter_init, ncores = ncores,
                                 reorder_factor = reorder_factor,
                                 seed = seed)
    } else if(!zero_inflation & sparsity) {

        prior_S <- 1-exp(-apply(X,2,sd)/mean(X[X!=0]))
        prob_S <- matrix(rep(prior_S, K), ncol=K)

        res <- run_sparse_gap_factor(X, K, sel_bound = sel_bound, verbose = verbose,
                                     monitor = monitor, iter_max = iter_max,
                                     iter_min = iter_min,
                                     init_mode = "random",
                                     epsilon = 1e-2, additional_iter = 10L,
                                     conv_mode = 1L, ninit = ninit,
                                     iter_init = iter_init, ncores = ncores,
                                     reorder_factor = reorder_factor,
                                     seed = seed,
                                     prob_S = prob_S, prior_S = prior_S)
    } else if(zero_inflation & sparsity) {

        prior_S <- 1-exp(-apply(X,2,sd)/mean(X[X!=0]))
        prob_S <- matrix(rep(prior_S, K), ncol=K)

        res <- run_zi_sparse_gap_factor(X, K, sel_bound = sel_bound, verbose = verbose,
                                        monitor = monitor, iter_max = iter_max,
                                        iter_min = iter_min,
                                        init_mode = "random",
                                        epsilon = 1e-2, additional_iter = 10L,
                                        conv_mode = 1L, ninit = ninit,
                                        iter_init = iter_init, ncores = ncores,
                                        reorder_factor = reorder_factor,
                                        seed = seed,
                                        prob_S = prob_S, prior_S = prior_S)
    }

    return(res)
}
