// Copyright 2018 Ghislain Durif
//
// This file is part of the `pCMF' library for R and related languages.
// It is made available under the terms of the GNU General Public
// License, version 2, or at your option, any later version,
// incorporated herein by reference.
//
// This program is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE.  See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public
// License along with this program; if not, write to the Free
// Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
// MA 02111-1307, USA

/*!
* \brief implementation of the wrapper for the Poisson NMF algorithm
* \author Ghislain Durif
* \version 1.0
* \date 26/02/2018
*/

#include <Rcpp.h>
#include <RcppEigen.h>

#include "algorithm_simple_factor.h"
#include "poisson_nmf.h"
#include "wrapper_matrix_factor.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::MatrixXi;                  // variable size matrix, integer
using Eigen::Map;                       // 'map' rather than copy

using pCMF::simple_factor_algo;
using pCMF::poisson_nmf;

//' @title run_poisson_nmf
//' @name run_poisson_nmf
//'
//' @description
//' Re-implementation of the Non-Negative Matrix Factorization (NMF) in
//' the Poisson framework.
//'
//' @details
//' Wrapper for a Cpp function.
//'
//' In the Non-negative Matrix Factorization framework (NMF), the count data
//' matrix \eqn{X} (dim \eqn{n \times p}) is approximated by a matrix product
//' \eqn{U V^t} where \eqn{U_{n\times K}} and \eqn{V_{p\times K}} respectively
//' represent the individuals (rows of \eqn{X}) and variables (cols of \eqn{X})
//' in a sub-space of dimension \eqn{K}.
//'
//' In the Poisson NMF framework, the approximation between \eqn{X} and
//' \eqn{U V^t} is made regarding the Kullback-Leibler divergence, which
//' corresponds to the Bregman divergence derived from the model
//' \eqn{X \sim P(U V^t)}, i.e. each entry \eqn{X_{ij}} is assumed to follow
//' a Poisson distribution of parameter \eqn{\sum_k U_{ik} V_{jk}}.
//'
//' More details regarding Poisson NMF can be found in Brunet et al. (2004),
//' the original implementation of NMF in R was done by Gaujoux
//' and Seoighe (2010). The generalization of Principal Component Analysis
//' based on Bregman divergence is discussed in Collins et al. (2001).
//'
//' @references
//' Brunet, J.-P., Tamayo, P., Golub, T.R., Mesirov, J.P., 2004.
//' Metagenes and molecular pattern discovery using matrix factorization.
//' PNAS 101, 4164–4169.
//'
//' Collins, M., Dasgupta, S., Schapire, R.E., 2001.
//' A generalization of principal components analysis to the exponential family.
//' Advances in Neural Information Processing Systems. pp. 617–624.
//'
//' Gaujoux, R., Seoighe, C., 2010.
//' A flexible R package for nonnegative matrix factorization.
//' BMC Bioinformatics 11, 367.
//'
//' @author
//' Ghislain Durif, \email{gd.dev@libertymail.net}
//'
//' @param X a count data matrix of dimension \code{n x p}.
//' @param K integer, required dimension of the subspace for the latent
//' representation.
//' @param U matrix of dimension \code{n x K}, initial values for the factor
//' matrix U. Default is NULL and U is randomly initialized from
//' a Uniform distribution on (0,1).
//' @param V matrix of dimension \code{p x K}, initial values for the factor
//' matrix V. Default is NULL and V is randomly initialized from
//' a Uniform distribution on (0,1).
//' @param verbose boolean indicating verbosity. Default is True.
//' @param monitor boolean indicating if model related measures
//' (log-likelihood, deviance, Bregman divergence between \eqn{X}
//' and \eqn{UV^t}) should be computed. Default is True.
//' @param iter_max integer, maximum number of iterations after which the
//' optimization is stopped even when the algorithm did not converge.
//' Default is 1000.
//' @param iter_min integer, minimum number of iterations without checking
//' convergence. Default is 500.
//' @param epsilon double precision parameter to assess convergence, i.e.
//' the convergence is reached when the gap (absolute or normalized) between
//' two iterates becomes smaller than epsilon. Default is 1e-2.
//' @param additional_iter integer, number of successive iterations during
//' which the convergence criterion should be verified to assess convergence.
//' @param conv_mode \{0,1,2\}-valued indicator setting how to assess
//' convergence: 0 for absolute gap, 1 for normalized gap and 2 for
//' RV coefficient between two iterates. Default is 1.
//' @param ninit integer, number of initialization to consider. In multiple
//' initialization mode (>1), the algorithm is run for
//' \code{iter_init} iterations with mutliple seeds and the best one
//' (regarding the optimization criterion) is kept. If \code{U} and \code{V}
//' are supplied, each seed uses a pertubated version of \code{U} and \code{V}
//' as initial values. Default value is 1.
//' @param iter_init integer, number of iterations during which the algorithms
//' is run in multi-initialization mode. Default value is 100.
//' @param ncores integer indicating the number of cores to use for
//' parallel computation. Default is 1 and no multi-threading is used.
//' @param reorder_factor boolean indicating if factors should be reordered
//' according to the model-related deviance criterion. Default value is True.
//' @param seed positive integer, seed value for random number generator.
//' Default is NULL and the seed is set based on the current time.
//'
//' @import Rcpp
//' @import RcppEigen
//' @importFrom Rcpp evalCpp
//' @useDynLib pCMF, .registration = TRUE
//'
//' @return list of Poisson NMF output
//' \item{factor}{list of estimated factor matrices:
//'     \code{U}: matrix of dimension \code{n x K}, representation of
//'     individuals in the subspace of dimension \code{K}.
//'     \code{V}: matrix of dimension \code{p x K}, contributions of variables
//'     to the subspace of dimension \code{K}.}
//' \item{convergence}{list of items related to the algorithm convergence:
//'     \code{converged}: boolean indicating if convergence was reached.
//'     \code{nb_iter}: integer, number of effective iterations.
//'     \code{conv_mode}: input \code{conv_mode} parameter.
//'     \code{conv_crit}: vector of size \code{nb_iter}, values of the
//'     convergence criterion across iterations.}
//' \item{loss}{double, values of the loss after iterations.}
//' \item{monitor}{list of monitored criteria if \code{conv_mode} input
//' parameter was set to 1, not returned otherwise:
//'     \code{norm_gap}: vector of size \code{nb_iter}, values of normalized
//'     gap between iterates across iterations.
//'     \code{abs_gap}: vector of size \code{nb_iter}, values of absolute
//'     gap between iterates across iterations.
//'     \code{loglikelihood}: vector of size \code{nb_iter}, values of the
//'     log-likelihood associated to the Poisson NMF model across iterations.
//'     \code{deviance}: vector of size \code{nb_iter}, values of the
//'     deviance associated to the Poisson NMF model across iterations.
//'     \code{optim_criterion}: vector of size \code{nb_iter}, values of the
//'     loss across iterations.}
//' \item{seed}{positive integer value corresponding to the value given as
//' seed in input parameter for random number generation.}
//'
//' @examples
//' \dontrun{
//' ## generate data
//' n <- 100
//' p <- 50
//' K <- 5
//' factorU <- generate_factor_matrix(n, K, ngroup=3, average_signal=0.5*K,
//'                                   group_separation=0.5,
//'                                   distribution="uniform",
//'                                   shuffle_feature=TRUE)
//' factorV <- generate_factor_matrix(p, K, ngroup=2, average_signal=0.5*K,
//'                                   group_separation=0.5,
//'                                   distribution="uniform",
//'                                   shuffle_feature=TRUE)
//' U <- factorU$factor_matrix
//' V <- factorV$factor_matrix
//' count_data <- generate_count_matrix(n, p, K, U, V)
//' X <- count_data$X
//' ## run matrix factorization algorithm
//' res <- run_poisson_nmf(X, K, verbose=FALSE)
//' }
//'
//' @export
// [[Rcpp::export]]
SEXP run_poisson_nmf(SEXP X, int K,
                     Rcpp::Nullable<MatrixXd> U = R_NilValue,
                     Rcpp::Nullable<MatrixXd> V = R_NilValue,
                     bool verbose = true, bool monitor = true,
                     int iter_max = 1000, int iter_min = 500,
                     double epsilon = 1e-2, int additional_iter = 10,
                     int conv_mode = 1, int ninit = 1,
                     int iter_init = 100, int ncores = 1,
                     bool reorder_factor = true,
                     Rcpp::Nullable<uint32_t> seed = R_NilValue) {

    Rcpp::List output = wrapper_matrix_factor<poisson_nmf, simple_factor_algo>(X, K, U, V,
                                                                               verbose, monitor,
                                                                               iter_max, iter_min,
                                                                               epsilon, additional_iter,
                                                                               conv_mode, ninit,
                                                                               iter_init, ncores,
                                                                               reorder_factor, seed);
    output.attr("class") = "NMF";
    return output;
}
