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


#' @title graphU
#' @name graphU
#'
#' @description
#' Graphical representation of the individuals coordinates (matrix U)
#' in the pCMF/NMF framework (\eqn{X ~ UV^t})
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
#' This function can be used to draw the scatterplot of individual coordinates
#' along two of the latent directions, i.e. two columns from the
#' matrix \code{U}.
#'
#' In the case of the Gamma-Poisson factor model, the geometry related to the
#' Gamma distribution in the exponential family is a log-representation,
#' thus, if requested, the matrix \code{log U} can be returned.
#'
#' The graphical representation is based on ggplot2.
#'
#' @author
#' Ghislain Durif, \email{gd.dev@libertymail.net}
#'
#' @seealso \code{\link{pCMF}}, \code{\link{run_poisson_nmf}}
#'
#' @param model model a Gamma-Poisson factor model output by the
#' function \code{\link{pCMF}} or a Poisson NMF model output by the
#' function \code{\link{run_poisson_nmf}}
#' @param axes a vector of 2 indexes corresponding to the 2 directions
#' to represent, Default value is \code{c(1,2)}.
#' @param labels a vector of indidividual/feature labels. Default value
#' is NULL, in this case individuals/features are not labelled.
#' @param log_representation boolean, indicating if the representation is in
#' the natural geometry associated to the Gamma distribution (log) or in the
#' standard Euclidean space. Default value is TRUE.
#' @param edit_theme boolean, indicating if the ggplot2 standard theme
#' should be edited or not. Default value is TRUE.
#' @param graph boolean, indicating if the graph should be drawn or not.
#' Default value is TRUE.
#'
#' @return the ggplot2 graph
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
#' res <- pCMF(X, K=2, verbose=FALSE)
#' ## graphical representation
#' graphU(res, axes=c(1,2), labels=factorU$feature_label) # individual representation
#' graphV(res, axes=c(1,2), labels=factorV$feature_label) # variable representation
#' }
#'
#' @export
graphU <- function(model, axes=c(1,2), labels=NULL, log_representation=TRUE,
                   edit_theme=TRUE, graph=TRUE) {

    return(matrix_plot(mat=getU(model, log_representation=log_representation),
                       axes=axes, labels=labels,
                       edit_theme=edit_theme, graph=graph))
}


#' @title graphV
#' @name graphV
#'
#' @description
#' Graphical representation of the variable loadings (matrix V)
#' in the pCMF/NMF framework (\eqn{X ~ UV^t})
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
#' This function can be used to draw the scatterplot of variable loadings
#' along two of the latent directions, i.e. two columns from the
#' matrix \code{V}.
#'
#' In the case of the Gamma-Poisson factor model, the geometry related to the
#' Gamma distribution in the exponential family is a log-representation,
#' thus, if requested, the matrix \code{log V} can be returned.
#'
#' The graphical representation is based on ggplot2.
#'
#' @author
#' Ghislain Durif, \email{gd.dev@libertymail.net}
#'
#' @seealso \code{\link{pCMF}}, \code{\link{run_poisson_nmf}}
#'
#' @param model model a Gamma-Poisson factor model output by the
#' function \code{\link{pCMF}} or a Poisson NMF model output by the
#' function \code{\link{run_poisson_nmf}}
#' @param axes a vector of 2 indexes corresponding to the 2 directions
#' to represent, Default value is \code{c(1,2)}.
#' @param labels a vector of indidividual/feature labels. Default value
#' is NULL, in this case individuals/features are not labelled.
#' @param log_representation boolean, indicating if the representation is in
#' the natural geometry associated to the Gamma distribution (log) or in the
#' standard Euclidean space. Default value is TRUE.
#' @param edit_theme boolean, indicating if the ggplot2 standard theme
#' should be edited or not. Default value is TRUE.
#' @param graph boolean, indicating if the graph should be drawn or not.
#' Default value is TRUE.
#'
#' @return the ggplot2 graph
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
#' res <- pCMF(X, K=2, verbose=FALSE)
#' ## graphical representation
#' graphU(res, labels=factorU$feature_label)
#' graphV(res, labels=factorV$feature_label)
#' }
#'
#' @export
graphV <- function(model, axes=c(1,2), labels=NULL, log_representation=TRUE,
                   edit_theme=TRUE, graph=TRUE) {

    return(matrix_plot(mat=getV(model, log_representation=log_representation),
                       axes=axes, labels=labels,
                       edit_theme=edit_theme, graph=graph))
}


#' @title matrix_plot
#' @name matrix_plot
#' @keywords internal
#'
#' @description
#' Graphical representation of a bidimensional scatterplot based on columns of
#' the input matrix \code{mat}.
#'
#' @details
#' This function is a wrapper to construct a ggplot2 graph representing
#' coordinates of individuals or features embedded in a \code{K} dimensional
#' latent space. Each row of the input matrix \code{mat} represents an
#' individual or a feature, and each column represents a latent dimension.
#'
#' This function plots the point coordinates (rows) from the matrix \code{mat}
#' according to 2 chosen directions (columns). Default behavior is to use the
#' first two direction (columns).
#'
#' This function is internal and is used by the function \code{\link{graphU}}
#' and \code{\link{graphV}}.
#'
#' @author
#' Ghislain Durif, \email{gd.dev@libertymail.net}
#'
#' @seealso \code{\link{graphU}} \code{\link{graphV}}
#'
#' @importFrom ggplot2 ggplot geom_point theme aes element_text element_rect element_line element_line element_blank
#'
#' @param mat a real-valued matrix of dimension \code{nrow x K} representing
#' the coordinates of individuals/features (rows) along \code{K} latent
#' directions (columns).
#' @param axes a vector of 2 indexes corresponding to the 2 directions
#' to represent, Default value is \code{c(1,2)}.
#' @param labels a vector of indidividual/feature labels. Default value
#' is NULL, in this case individuals/features are not labelled.
#' @param edit_theme boolean, indicating if the ggplot2 standard theme
#' should be edited or not. Default value is TRUE.
#' @param graph boolean, indicating if the graph should be drawn or not.
#' Default value is TRUE.
#'
#' @return the ggplot2 graph
#'
#' @export
matrix_plot <- function(mat, axes=c(1:2), labels=NULL,
                       edit_theme=TRUE, graph=TRUE) {

    ## check input
    Kmax <- max(axes)
    if(Kmax > ncol(mat)) {
        stop("'axes' argument is not compatible with 'mat' dimension")
    }
    if(!is.null(labels)) {
        if(length(labels) != nrow(mat)) {
            stop("'labels' argument length is not compatible with 'mat' dimension")
        }
    }

    ## format the data
    dataToPlot <- data.frame(comp1=mat[,axes[1]], comp2=mat[,axes[2]])

    ## graph representation
    if(!is.null(labels)) {
        dataToPlot$labels <- as.factor(labels)
        g <- ggplot(dataToPlot, aes(x=comp1, y=comp2, color=labels))
    } else {
        g <- ggplot(dataToPlot, aes(x=comp1, y=comp2))
    }
    g <- g + geom_point()
    if(edit_theme) {
        g <- g + theme(legend.text=element_text(size=14),
                       legend.title=element_text(size=14),
                       axis.text.x=element_text(size=10),
                       axis.title.x=element_text(size=14),
                       axis.title.y=element_text(size=14),
                       axis.text.y=element_text(size=10),
                       strip.text.x=element_text(size=14),
                       strip.text.y=element_text(size=14),
                       plot.title=element_text(size=14))
        g <- g + theme(panel.background=element_rect(fill="white", colour="black"),
                       panel.grid.major=element_line(color="grey90"),
                       panel.grid.minor=element_line(color="grey90"),
                       strip.background=element_blank())
    }
    ## plot graph ?
    if(graph) {
        g
    }
    ## output
    return(g)
}

