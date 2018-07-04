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

#' @title matrix_heatmap
#' @name matrix_heatmap
#'
#' @description
#' Wrapper for the function \code{\link[fields]{image.plot}} from the
#' package \code{fields}
#'
#' @details
#' Plot the entries of the input matrix as a heatmap
#'
#' @author
#' Ghislain Durif, \email{gd.dev@libertymail.net}
#'
#' @importFrom fields image.plot
#'
#' @param mat the matrix to plot
#' @param ... any parameter that can be fed to the function
#' \code{\link[fields]{image.plot}}
#'
#' @examples
#' \dontrun{
#' n <- 100
#' K <- 50
#' mat <- generate_factor_matrix(n, K, ngroup=3, average_signal=0.5*K,
#'                               group_separation=0.5,
#'                               distribution="uniform",
#'                               shuffle_feature=FALSE)
#' matrix_heatmap(mat$factor_matrix)
#' }
#'
#' @export
matrix_heatmap <- function(mat, ...) {
    image.plot(t(apply(mat, 2, rev)), xaxt="n", yaxt="n", ...)
}
