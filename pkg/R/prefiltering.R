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

#' @title prefilter
#' @name prefilter
#'
#' @description
#' Apply pre-filtering to columns of the data matrix \eqn{X} (remove columns
#' with only null values or too much null values or extremely high values).
#' Optional pre-selection according to a variance-based heuristic introduced
#' in Durif et al. (2019).
#'
#' @details
#' This function can be used to filter out columns of the data matrix:
#' \enumerate{
#' \item{columns with only null values are dismissed.}
#' \item{columns with less than \code{prop} in proportion of non null
#' entries are dismissed.}
#' \item{columns with extremely high values are dismissed accoding to the
#' following rule:
#' \verb{
#'     Xmax <- apply(X, 2, max) ## column-wise max
#'     Xmax <= quantile(Xmax, probs=quant_max)
#' }
#' With \code{quant_max=1}, no column are dismissed.
#' }
#' \item{optionally, if \code{presel=TRUE}, columns can removed
#' according to the following heuristic on variance:
#' \verb{
#'     prior_variance <- 1-exp(-apply(X,2,sd)/mean(X[X!=0]))
#'     prior_variance > threshold
#' }
#' This pre-selection step can be useful to save computation time.
#' }
#' }
#'
#' @references
#' Durif, G., Modolo, L., Mold, J.E., Lambert-Lacroix, S., Picard, F., 2017.
#' Probabilistic Count Matrix Factorization for Single Cell Expression Data
#' Analysis. arXiv:1710.11028 [stat].
#'
#' @author
#' Ghislain Durif, \email{gd.dev@libertymail.net}
#'
#' @param X data matrix of dimension \code{nrow x ncol}
#' @param prop real [0,1]-value, minimal proportion of non null entries
#' in kept columns. Default is 0.05. If zero, only columns with ony null
#' values are dismissed.
#' @param quant_max real [0,1]-value, controling the removing of  columns with
#' extremely high values. In particular, each column whose max value is higher
#' than the quantile \code{quant_max} computed on the set of max values
#' across all columns is removed. Default is 1, and no column is dismissed
#' according to this criterion.
#' @param presel boolean, indicating if columns should be filtered out
#' according to the variance-based heuristics introduced in
#' Durif et al. (2019). Default is FALSE.
#' @param threshold real [0,1]-value, controling the level of pre-selection
#' according to the variance-based heuristics introduced in
#' Durif et al. (2019). Only usd if \code{presel=TRUE}. Default is 0.2,
#' meaning that only columns for which the heuristic is higher than 0.2
#' are kept (c.f. details).
#'
#' @importFrom stats quantile sd
#'
#' @return a boolean vector of length \code{ncol} indicating which columns
#' are kept.
#'
#' @examples
#' \dontrun{
#' #' ## generate data
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
#' ## pre-filtering
#' kept_cols <- prefilter(X, prop = 0.05, quant_max = 0.95)
#' X <- X[,kept_cols]
#' }
#'
#' @export
prefilter <- function(X, prop = 0.05, quant_max = 1,
                      presel = FALSE, threshold = 0.2) {

    if(!is.data.frame(X) & !is.matrix(X) & !is.array(X)) {
        stop("`X` should be a matrix or a data.frame or an array")
    }
    if(!is.numeric(prop) & length(prop)!=1 & prop<0 & prop>1) {
        stop("`prop` should be a [0,1]-valued number")
    }
    if(!is.numeric(quant_max) & length(quant_max)!=1 & quant_max<0 & quant_max>1) {
        stop("`quant_max` should be a [0,1]-valued number")
    }
    if(!(presel %in% c(TRUE,FALSE)) & !(presel %in% c(0,1))) {
        stop("`presel` should be a boolean")
    }
    if(!is.numeric(threshold) & length(threshold)!=1 & threshold<0 & threshold>1) {
        stop("`threshold` should be a [0,1]-valued number")
    }

    ## column-wise max
    Xmax <- apply(X, 2, max)
    ## remove columns with only 0
    crit1 <- (Xmax>0)
    ## remove columns with less than `prop` *100% non null entries
    crit2 <- (apply(X, 2, function(x) sum(x!=0)) >= prop * nrow(X))
    ## remove columns with extremely high values
    crit3 <- (Xmax <= quantile(Xmax, probs=quant_max))
    ## remove columns based on deviation/mean criterion
    crit4 <- rep(TRUE, ncol(X))
    if(presel) {
        prior_variance <- 1-exp(-apply(X,2,sd)/mean(X[X!=0]))
        crit4 <- prior_variance > threshold
    }
    ## finally
    kept_cols <- crit1 & crit2 & crit3 & crit4
    return(kept_cols)
}
