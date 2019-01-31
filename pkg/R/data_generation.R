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

#' @title generate_factor_matrix
#' @name generate_factor_matrix
#'
#' @description
#' Generation of a factor matrix in a latent sub-space of dimension \code{K}
#'
#' @details
#' This function can be used to generate factor matrix in
#' a matrix factorization framework \eqn{X ~ UV^t}:
#'     - U for individual representation in the latent sub-space
#'     - V for variable contribution to the latent sub-space
#'
#' More details regarding data generation can be found in Durif et al. (2017,
#' Appendix).
#'
#' If the uniform distribution is used, entries in \code{U} and \code{V} are
#' simulated from the distribution Uniform(0, 2*\code{average_signal}/\code{K}).
#'
#' If the gamma distribution is used, entries in \code{U} and \code{V} are
#' simulated from the distribution Gamma(\code{average_signal}/\code{K}, 1).
#' If the parameter \code{disp} is given, then the distribution
#' Gamma(\code{average_signal}/\code{K}, \code{disp}) is used.
#'
#' If the exponential distribution is used, entries in \code{U} and \code{V} are
#' simulated from the distribution Gamma(1, \code{K}/\code{average_signal})
#' which corresponds to the distribution
#' Exponential(\code{K}/\code{average_signal}).
#'
#' The noise variables are generated with the same framework, using
#' the parameter \code{noise_level} instead of \code{average_signal}.
#'
#' @references
#' Durif, G., Modolo, L., Mold, J.E., Lambert-Lacroix, S., Picard, F., 2017.
#' Probabilistic Count Matrix Factorization for Single Cell Expression Data
#' Analysis. arXiv:1710.11028 [stat].
#'
#' @author
#' Ghislain Durif, \email{gd.dev@libertymail.net}
#'
#' @param nfeature integer, number of features (individuals or variables),
#' i.e. number of rows in the generated matrix.
#' @param K integer, dimension of the latent sub-space, i.e. number of columns
#' in the generated matrix.
#' @param ngroup integer, number of groups structuring the features
#' (individuals or variables). Such groups induce a structure of dependence
#' between the features. Default value is 1 and the features are not
#' structured into dependent groups. Condition: \code{ngroup} <= \code{K}.
#' @param average_signal real value or vector of real value of size \code{ngroup},
#' average signal level in the structured block of the generated matrix. If a
#' single value, all groups share the same \code{average_signal}. If a
#' vector, each group has the corresponding \code{average_signal}.
#' @param disp real value or vectorof real value of size \code{ngroup}.
#' If distribution is used, \code{disp} defines the dispersion parameter
#' for each/all group(s) (if resp vector/single value) for the gamma distribution.
#' See details. Default value is NULL, and \code{disp} is set to 1.
#' @param group_separation real value in \[0,1\] indicating the level of
#' separation between the groups, 0 = no separation, 1 = high separation.
#' Default value is 0.5.
#' @param distribution character string, distribution used to generate
#' the matrix entries. Default is "uniform", other possible values are
#' "gamma" and "exponential". See details.
#' @param shuffle_feature boolean, should the features (individuals
#' or variables) be shuffled in the output.
#' @param prop_noise_feature real value in \[0,1\], proportion of features
#' that does not contributes to the group structure, i.e. constitute an
#' additional group of noise. Default is 0.
#' @param noise_level real value, average level of the noisy feature. Default
#' is null and the level of noisy feature is set to be
#' (1 - \code{group_separation}) * \code{average_signal} / \code{disp}.
#' @param seed integer, seed value for random number generator. Default value
#' is NULL and no the seed is not specifically set.
#' @param tag string, tag to name feature (i.e. rows in the factor matrix),
#' with "tagXXX" where "XXX" is the row number. Default value is NULL and
#' feature are labelled "1", "2", etc.
#'
#' @importFrom stats rgamma runif
#'
#' @return list containing the following
#' \item{factor_matrix}{the factor matrix of dimension \code{nfeature x K}.}
#' \item{feature_order}{vector of index giving the shuffling order used to
#' generate the data}
#' \item{feature_label}{vector of labels}
#'
#' @examples
#' \dontrun{
#' ## generate data
#' n <- 100
#' K <- 5
#' factorU <- generate_factor_matrix(n, K, ngroup=3, average_signal=0.5*K,
#'                                   group_separation=0.5,
#'                                   distribution="uniform",
#'                                   shuffle_feature=TRUE)
#' }
#'
#' @export
generate_factor_matrix <- function(nfeature, K, ngroup=1, average_signal=1,
                                   disp=NULL,
                                   group_separation=0.5,
                                   distribution="uniform",
                                   shuffle_feature=TRUE,
                                   prop_noise_feature=0,
                                   noise_level=NULL,
                                   seed=NULL,
                                   tag=NULL) {
    # input check
    if(!is.numeric(nfeature) | !(nfeature>0) | (floor(nfeature) - nfeature !=0)) {
        stop("'nfeature' input parameter should be a positive integer")
    }
    if(!is.numeric(K) | !(K>0) | (floor(K) - K != 0)) {
        stop("'K' input parameter should be a positive integer")
    }
    if(!is.numeric(ngroup) | !(ngroup>0) | (ngroup>K)) {
        stop("'ngroup' input parameter should be a positive integer <= 'K'")
    }
    if(!is.numeric(average_signal) | any(average_signal<0) | (length(average_signal) != 1 & length(average_signal) != ngroup)) {
        stop("'average_signal' input parameter should be a real positive value, or a vector of size 'ngroup' of real positive values")
    }
    if(!is.null(disp)) {
        if(!is.numeric(disp) | any(disp<0) | (length(disp) != 1 & length(disp) != ngroup)) {
            stop("'disp' input parameter should be a real positive value, or a vector of size 'ngroup' of real positive values or a NULL value")
        }
    }
    if(!is.numeric(group_separation) | (group_separation<0) | (group_separation>1)) {
        stop("'group_separation' input parameter should be a real value in [0,1]")
    }
    if(!(distribution %in% c("uniform", "gamma", "exponential"))) {
        stop("'distribution' input parameter should be \"uniform\" or \"gamma\"")
    }
    if(!is.logical(shuffle_feature)) {
        stop("'shuffle_feature' input parameter should be a boolean")
    }
    if(!is.numeric(prop_noise_feature) | (prop_noise_feature<0) | (prop_noise_feature>1)) {
        stop("'prop_noise_feature' input parameter should be a real value in [0,1]")
    }
    if(!is.null(noise_level)) {
        if(!is.numeric(noise_level) | (noise_level<0)) {
            stop("'noise_level' input parameter should be a real positive or NULL value")
        }
    }
    if(!is.null(seed)) {
        if(!is.integer(seed) | (seed<0) ) {
            stop("'seed' input parameter should be NULL or a positive integer")
        }
    }
    if(!is.null(tag)) {
        if(length(tag)>1) {
            stop("'tag' should be a single string of characters.")
        }
    }

    # set seed if necessary
    if(!is.null(seed)) {
        set.seed(seed)
    }

    # generate matrix of parameter for each block, dim ngroup x ngroup
    param_block_matrix <- diag(average_signal, nrow=ngroup, ncol=ngroup)
    disp_block_matrix <- NULL
    if(is.null(disp)) {
        disp <- 1
    }
    disp_block_matrix <- diag(1, nrow=ngroup, ncol=ngroup)
    if(ngroup>1) {
        param_block_matrix <- param_block_matrix + mean(average_signal) * (1 - group_separation) * (1 - diag(1, nrow=ngroup, ncol=ngroup))
        disp_block_matrix <- disp_block_matrix + (1 - diag(1, nrow=ngroup, ncol=ngroup))
    }
    # shuffle groups (columns of 'param_block_matrix')
    col_order = sample.int(n=ngroup, size=ngroup, replace=FALSE)
    param_block_matrix <- param_block_matrix[,col_order]
    disp_block_matrix <- disp_block_matrix[,col_order]

    # block id
    id_block <- 1:ngroup

    ### structuring block matrix
    nfeature0 <- floor( (1 - prop_noise_feature) * nfeature)
    mat0 <- NULL
    id_rows <- NULL

    if(nfeature0 > 0) {

        param_matrix <- NULL

        if(ngroup > 1) {
            # assign features (=rows) to a block
            id_rows <- sort(rep(id_block, length=nfeature0))

            # assign components (=cols) to a block
            id_cols <- sort(rep(id_block, length=K))

            # construction of the parameter matrix by block, dim nfeature x K
            param_matrix <- matrix(NA, nrow=nfeature0, ncol=K)
            disp_matrix <- matrix(NA, nrow=nfeature0, ncol=K)
            for(row_block in id_block) {
                for(col_block in id_block) {
                    rows_in_block <- (1:nfeature0)[id_rows == row_block]
                    nrow_block <- length(rows_in_block)
                    cols_in_block <- (1:K)[id_cols == col_block]
                    ncol_block <- length(cols_in_block)

                    param_matrix[rows_in_block, cols_in_block] <- matrix(param_block_matrix[row_block, col_block], nrow=nrow_block, ncol=ncol_block)
                    disp_matrix[rows_in_block, cols_in_block] <- matrix(disp_block_matrix[row_block, col_block], nrow=nrow_block, ncol=ncol_block)
                }
            }
        } else {
            id_rows <- rep(1, nfeature0)
            param_matrix <- matrix(average_signal, nrow=nfeature0, ncol=K)
            disp_matrix <- matrix(disp, nrow=nfeature0, ncol=K)
        }

        # generation of the factor sub-matrix
        if(distribution == "uniform") {
            mat0 <- sapply(1:K, function(k) return(runif(nfeature0, min=0, max=2*param_matrix[,k]/sqrt(K)))) # matrix nfeature0 x K
        } else if(distribution == "gamma") {
            mat0 <- sapply(1:K, function(k) return(rgamma(nfeature0, shape=param_matrix[,k]/sqrt(K), rate=disp_matrix[,k]))) # matrix nfeature0 x K
        } else if(distribution == "exponential") {
            mat0 <- sapply(1:K, function(k) return(rgamma(nfeature0, shape=1, rate=sqrt(K)/param_matrix[,k]))) # matrix nfeature0 x K
        }
    }

    ### noise block
    mat_noise <- NULL
    nfeature_noise <- nfeature - nfeature0
    if(is.null(noise_level)) {
        noise_level <- mean(average_signal) * (1 - group_separation)
    }
    if(nfeature_noise > 0) {
        # generation of the factor sub-matrix
        if(distribution == "uniform") {
            mat_noise <- sapply(1:K, function(k) return(runif(nfeature_noise, min=0, max=2*noise_level/sqrt(K)))) # matrix nfeature_noise x K
        } else if(distribution == "gamma") {
            mat_noise <- sapply(1:K, function(k) return(rgamma(nfeature_noise, shape=noise_level/sqrt(K), rate=min(disp)))) # matrix nfeature_noise x K
        }  else if(distribution == "exponential") {
            mat_noise <- sapply(1:K, function(k) return(rgamma(nfeature_noise, shape=1, rate=sqrt(K)/noise_level))) # matrix nfeature_noise x K
        }
    }

    ### concatenate structuring block matrix and noise block matrix
    mat <- rbind(mat0, mat_noise)

    # shuffle rows
    feature_order <- 1:nfeature
    if(shuffle_feature) {
        feature_order <- sample.int(n=nfeature, size=nfeature, replace=FALSE)
    }
    mat <- mat[feature_order,]
    feature_label <- c(id_rows, rep(0, nfeature_noise))[feature_order]

    # name rows
    rownames(mat) <- paste0(tag, 1:nrow(mat))
    colnames(mat) <- paste0("comp", 1:ncol(mat))

    ### output
    res <- list(factor_matrix=mat, feature_order=feature_order, feature_label=feature_label)
    return(res)

}



#' @title generate_count_matrix
#' @name generate_count_matrix
#'
#' @description
#' Generates a count data matrix \eqn{X} according to a probabilistic factor
#' model based on the Poisson distribution
#' with potential drop-out events (zero-inflation).
#'
#' @details
#' The generated count data matrix \eqn{X} of dimension \eqn{n \times p} stores
#' \eqn{n} obaservations (in rows) of \eqn{p} recorded variables (in columns).
#'
#' The generative process is based on a probabilistic factor model,
#' i.e. individuals and variables are represented in a latent space of
#' dimension \eqn{K} by the input matrices \eqn{U} (observation coordinates
#' of dim \eqn{n \times K}) and \eqn{V} (variable loadings of
#' dim \eqn{p \times K}) respectively.
#'
#' Generative process (without zero-inflation):
#' \deqn{X | U,V ~ Poisson(UV^t)}
#' i.e. \eqn{X_{ij} | U_i, V_j ~ Poison(\sum_k U_{ik}V_{jk})}
#'
#' Generative process (with zero-inflation):
#' \deqn{X_{ij} | D_{ij}, U_i, V_j ~ D_{ij} \times Poison(\sum_k U_{ik}V_{jk})}
#' where \eqn{D_{ij}} is a Bernoulli indicator for drop-out events:
#' \deqn{D_{ij} ~ Bernoulli(p_j)}
#' with \eqn{p_j\in[0,1]}, i.e. \eqn{D_{ij} = 0} in case of a drop-out events.
#' The \eqn{p_j} probability can be directly supplied in input (\code{prob1}
#' input parameter) or computed as \eqn{1 - exp(-rate \times m^2)}
#' where $m$ is the average signal for the corresponding column
#' i.e. \eqn{\frac{\sum_{i,k} U_{ik}V_{jk}}{n}}. This formulation is adapted
#' from the drop-out modeling in Pierson and Yau (2015).
#'
#' Important: the \code{U} and \code{V} input matrices can be generated
#' with the function \code{\link{generate_factor_matrix}}.
#'
#' More details regarding data generation can be found in Durif et al. (2017,
#' Appendix).
#'
#' @author
#' Ghislain Durif, \email{gd.dev@libertymail.net}
#'
#' @param n positive integer, number of observations (nb of rows)
#' @param p positive integer, number of variables (nb of columns)
#' @param K positive integer, number of latent factors (dimension of
#' latent subspace)
#' @param U matrix of dimension \code{n x K}, representation (coordinates)
#' of individuals in the latent space of dimension \code{K}.
#' @param V matrix of dimension \code{p x K}, contributions (loadings)
#' of variables to the latent space of dimension \code{K}.
#' @param ZI boolean, indicating if the data are zero-inflated, i.e.
#' contains drop-out events. Default value is FALSE.
#' @param prob1 vector of length \code{p} of Bernoulli probabilities
#' for zero-inflation. The probability of drop-out events per variable
#' (i.e. column in \code{X}) is \code{1 - prob1}. When NULL and if \code{ZI} is
#' set to TRUE, the \code{rate0} input parameter is used to compute drop-out
#' probabilities. Default value is NULL.
#' @param rate0 vector of length \code{p} of rate used to generate
#' Bernoulli probability for zero-inflation,
#' i.e. \code{prob1 = 1 - exp(-rate0*meanX^2)} where \code{meanX} is the
#' column-average Poisson signal used to generate the data matrix
#' (c.f. details). Default value is NULL. When \code{prob1} input parameter is
#' given \code{rate0} input parameter is not used.
#'
#' @return list containing the following
#' \item{X}{data matrix (dim \code{n x p}) of counts}
#' \item{U}{input parameter matrix (dim \code{n x K}) of factor
#' coordinates (in observation space).}
#' \item{V}{input parameter matrix (dim \code{p x K}) of factor loadings
#' (in variable space).}
#' \item{n}{number of observations (or rows in X)}
#' \item{p}{number of variables (or columns in X)}
#' \item{K}{number of latent factors (dimension of latent subspace)}
#' \item{ZI}{input parameter boolean}
#' \item{prob1}{input parameter vector (NULL if unused) or computed vector
#' if \code{rate0} input parameter is used}
#' \item{rate0}{input parameter vector (NULL if unused)}
#' \item{Xnzi}{NULL if \code{ZI} is FALSE, otherwise observation matrix
#' (dim \code{n x p}) without drop-out events}
#' \item{ZIind}{NULL if \code{ZI} is FALSE, otherwise matrix (dim \code{n x p})
#' of drop-out indicators \eqn{[D_{ij}]_{n\times p}}}
#'
#' @importFrom stats rpois rbinom
#'
#' @references
#' Pierson, E., Yau, C., 2015. ZIFA: Dimensionality reduction for zero-inflated
#' single-cell gene expression analysis. Genome Biology 16, 241.
#' https://doi.org/10.1186/s13059-015-0805-z
#'
#' Durif, G., Modolo, L., Mold, J.E., Lambert-Lacroix, S., Picard, F., 2017.
#' Probabilistic Count Matrix Factorization for Single Cell Expression Data
#' Analysis. arXiv:1710.11028 [stat].
#'
#' @examples
#' \dontrun{
#' ## generate data
#' n <- 100
#' p <- 50
#' K <- 5
#' factorU <- generate_factor_matrix(n, K, ngroup=3, average_signal=30,
#'                                   group_separation=0.8,
#'                                   distribution="gamma",
#'                                   shuffle_feature=TRUE)
#' factorV <- generate_factor_matrix(p, K, ngroup=2, average_signal=30,
#'                                   group_separation=0.8,
#'                                   distribution="gamma",
#'                                   shuffle_feature=TRUE)
#' U <- factorU$factor_matrix
#' V <- factorV$factor_matrix
#' count_data <- generate_count_matrix(n, p, K, U, V)
#' X <- count_data$X
#' }
#'
#' @export
generate_count_matrix <- function(n, p, K, U, V,
                                  ZI=FALSE, prob1=NULL, rate0=NULL) {

    if(!is.numeric(n) | !(n>0) | (floor(n) - n != 0)) {
        stop("'n' input parameter should be a positive value")
    }
    if(!is.numeric(p) | !(p>0) | (floor(p) - p != 0)) {
        stop("'p' input parameter should be a positive value")
    }
    if(!is.numeric(K) | !(K>0) | (floor(K) - K != 0)) {
        stop("'K' input parameter should be a positive value")
    }
    if(!is.matrix(U) | any(dim(U) != c(n,K)) | any(U<0)) {
        stop("'U' input parameter should be a positive-valued matrix of dimension 'n' x 'K'")
    }
    if(!is.matrix(V) | any(dim(V) != c(p,K)) | any(V<0)) {
        stop("'V' input parameter should be a positive-valued matrix of dimension 'p' x 'K'")
    }
    if(!is.logical(ZI)) {
        stop("'ZI' input parameter should be a boolean")
    }
    if(ZI & (is.null(prob1) & is.null(rate0))) {
        stop(paste0("It is required to use a zero-inflated model ",
                    "to generate data ('ZI' input parameter is set to TRUE) ",
                    "however neither 'prob1' nor 'rate0' input parameters ",
                    "are given"))
    }
    if(!is.null(prob1)) {
        if(!is.vector(prob1) | (length(prob1) != p)) {
            stop("if used, 'prob1' input parameter should be a vector of size 'p'")
        }
    }
    if(!is.null(rate0)) {
        if(!is.vector(rate0) | (length(rate0) != p)) {
            stop("if used, 'rate0' input parameter should be a vector of size 'p'")
        }
    }

    ## Poisson rate
    Lambda = U %*% t(V)

    ## generating the count (without drop-out events to start)
    Xnzi <- matrix(rpois(n=n*p, lambda=as.vector(Lambda)), nrow=n, ncol=p)

    ## generating the Bernoulli variables if requested
    if(ZI) {
        if(!is.null(rate0)) {
            meanX <- apply(Xnzi, 2, mean)
            prob1 <- 1 - exp(-rate0*meanX^2)
        }
        D <- sapply(prob1, function(pi) return(rbinom(n=n,size=1,prob=pi)))
    } else {
        D <- matrix(1, nrow=n, ncol=p)
    }

    # name rows and columns
    rownames(Xnzi) <- rownames(U)
    colnames(Xnzi) <- rownames(V)

    ## counts model (with zero-inflation if so)
    X <- Xnzi * D

    if(!ZI) {
        Xnzi <- NULL
        D <- NULL
    }

    ## return
    return(list(X=X, U=U, V=V, n=n, p=p, K=K,
                ZI=ZI, prob1=prob1, rate0=rate0, Xnzi=Xnzi, ZIind=D))
}
