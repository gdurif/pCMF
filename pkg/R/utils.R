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


#' @title getU
#' @name getU
#'
#' @description
#' Getter for the matrix U in the pCMF/NMF framework (\eqn{X ~ UV^t})
#'
#' @details
#' We consider a count data matrix \eqn{X} with
#' \eqn{n} observations/individuals in rows and
#' \eqn{p} recorded variables in columns.
#'
#' In the matrix factorization framework (Poisson NMF or Gamma-Poisson
#' factor model), the data matrix \eqn{X_{n\times p}} is approximated by the
#' product \eqn{UV^t} where \eqn{U_{n \times K}} and \eqn{V_{p\times K}}
#' are respectively the observation coordinates and variable loadings in
#' the latent space of lower dimension \eqn{K}.
#'
#' This function returns the matrix \code{U}.
#'
#' In the case of the Gamma-Poisson factor model, the geometry related to the
#' Gamma distribution in the exponential family is a log-representation,
#' thus, if requested, the matrix \code{log U} can be returned.
#'
#' @author
#' Ghislain Durif, \email{gd.dev@libertymail.net}
#'
#' @seealso \code{\link{pCMF}}, \code{\link{run_poisson_nmf}}
#'
#' @param model a Gamma-Poisson factor model output by the
#' function \code{\link{pCMF}} or a Poisson NMF model output by the
#' function \code{\link{run_poisson_nmf}}
#' @param log_representation boolean, useful only with the Gamma Poisson
#' factor model, it indicates if the representation is in
#' the natural geometry associated to the Gamma distribution (log) or in the
#' Euclidean space, default is TRUE
#'
#' @return the matrix \code{U} of individual coordinates in the lower
#' dimensional sub-space (or \code{log U} if requested)
#'
#' @examples
#' \dontrun{
#' ## generate data
#' n <- 100
#' p <- 200
#' K <- 10
#' factorU <- generate_factor_matrix(n, K, ngroup=3, average_signal=60,
#'                                   group_separation=0.8,
#'                                   distribution="gamma",
#'                                   shuffle_feature=TRUE)
#' factorV <- generate_factor_matrix(p, K, ngroup=2, average_signal=60,
#'                                   group_separation=0.8,
#'                                   distribution="gamma",
#'                                   shuffle_feature=TRUE)
#' U <- factorU$factor_matrix
#' V <- factorV$factor_matrix
#' count_data <- generate_count_matrix(n, p, K, U, V)
#' X <- count_data$X
#' ## or use your own data as a count matrix
#' ## of dimension cells x genes (individuals x features)
#' ## run pCMF algorithm
#' res <- pCMF(X, K, verbose=FALSE)
#' ## get U and V
#' hatU <- getU(res)
#' hatV <- getV(res)
#' }
#'
#' @export
getU <- function(model, log_representation=TRUE) {

    if(class(model) == "NMF") {
        U <- as.matrix(model$factor$U)
    } else if(class(model) == "pCMF") {
        if(log_representation) {
            U <- apply(as.matrix(model$stats$EU), c(1,2), function(x) return(log(x+1E-5)))
        } else {
            U <- as.matrix(model$stats$EU)
        }
    } else {
        stop("wrong model in input")
    }
    return(U)
}


#' @title getV
#' @name getV
#'
#' @description
#' Getter for the matrix V in the pCMF/NMF framework (\eqn{X ~ UV^t})
#'
#' @details
#' We consider a count data matrix \eqn{X} with
#' \eqn{n} observations/individuals in rows and
#' \eqn{p} recorded variables in columns.
#'
#' In the matrix factorization framework (Poisson NMF or Gamma-Poisson
#' factor model), the data matrix \eqn{X_{n\times p}} is approximated by the
#' product \eqn{UV^t} where \eqn{U_{n \times K}} and \eqn{V_{p\times K}}
#' are respectively the observation coordinates and variable loadings in
#' the latent space of lower dimension \eqn{K}.
#'
#' This function returns the matrix \code{V}.
#'
#' In the case of the Gamma-Poisson factor model, the geometry related to the
#' Gamma distribution in the exponential family is a log-representation,
#' thus, if requested, the matrix \code{log V} can be returned.
#'
#' @author
#' Ghislain Durif, \email{gd.dev@libertymail.net}
#'
#' @seealso \code{\link{pCMF}}, \code{\link{run_poisson_nmf}}
#'
#' @param model a Gamma-Poisson factor model output by the
#' function \code{\link{pCMF}} or a Poisson NMF model output by the
#' function \code{\link{run_poisson_nmf}}
#' @param log_representation boolean, useful only with the Gamma Poisson
#' factor model, it indicates if the representation is in
#' the natural geometry associated to the Gamma distribution (log) or in the
#' Euclidean space, default is TRUE
#'
#' @return the matrix \code{V} of individual coordinates in the lower
#' dimensional sub-space (or \code{log V} if requested)
#'
#' @examples
#' \dontrun{
#' ## generate data
#' n <- 100
#' p <- 200
#' K <- 10
#' factorU <- generate_factor_matrix(n, K, ngroup=3, average_signal=60,
#'                                   group_separation=0.8,
#'                                   distribution="gamma",
#'                                   shuffle_feature=TRUE)
#' factorV <- generate_factor_matrix(p, K, ngroup=2, average_signal=60,
#'                                   group_separation=0.8,
#'                                   distribution="gamma",
#'                                   shuffle_feature=TRUE)
#' U <- factorU$factor_matrix
#' V <- factorV$factor_matrix
#' count_data <- generate_count_matrix(n, p, K, U, V)
#' X <- count_data$X
#' ## or use your own data as a count matrix
#' ## of dimension cells x genes (individuals x features)
#' ## run pCMF algorithm
#' res <- pCMF(X, K, verbose=FALSE)
#' ## get U and V
#' hatU <- getU(res)
#' hatV <- getV(res)
#' }
#'
#' @export
getV <- function(model, log_representation=TRUE) {

    if(class(model) == "NMF") {
        V <- as.matrix(model$factor$V)
    } else if(class(model) == "pCMF") {
        if(log_representation) {
            V <- apply(as.matrix(model$stats$EV), c(1,2), function(x) return(log(x+1)))
        } else {
            V <- as.matrix(model$stats$EV)
        }
    } else {
        stop("wrong model in input")
    }
    return(V)
}
