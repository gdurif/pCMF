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
* \brief implementation of the wrapper for the pCMF algorithm based on Gamma-Poisson factor model
* \author Ghislain Durif
* \version 1.0
* \date 03/04/2018
*/

#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <RcppEigen.h>

#include "algorithm_variational_EM.h"
#include "gap_factor_model.h"
#include "utils/random.h"
#include "wrapper_gap_factor.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::MatrixXi;                  // variable size matrix, integer
using Eigen::Map;                       // 'map' rather than copy

using pCMF::variational_em_algo;
using pCMF::gap_factor_model;

using std::string;

//' @title run_gap_factor
//' @name run_gap_factor
//'
//' @description
//' Implementation of the probabilistic Count Matrix Factorization (pCMF)
//' method based on the Gamma-Poisson hirerarchical factor model. This method
//' is specifically designed to analyze count matrices such as gene expression
//' profiles (RNA-seq) obtained by high throughput sequencing.
//'
//' @details
//' Wrapper for a Cpp function.
//'
//' In the probabilistic Count Matrix Factorization framework (pCMF), the
//' count data matrix \eqn{X} (dim \eqn{n \times p}) is approximated
//' by a matrix product \eqn{U V^t} where \eqn{U_{n\times K}} and
//' \eqn{V_{p\times K}} respectively represent the individuals
//' (rows of \eqn{X}) and variables (cols of \eqn{X})
//' in a sub-space of dimension \eqn{K}.
//'
//' In the pCMF framework, the approximation between \eqn{X} and
//' \eqn{U V^t} is made regarding the Kullback-Leibler divergence, which
//' corresponds to the Bregman divergence derived from the model
//' \eqn{X \sim P(U V^t)}, i.e. each entry \eqn{X_{ij}} is assumed to follow
//' a Poisson distribution of parameter \eqn{\sum_k U_{ik} V_{jk}}.
//' In addition, factors \eqn{U_{ik}} and \eqn{V_{jk}} are assumed to
//' be independent latent random variables following Gamma distributions.
//'
//' More details regarding pCMF can be found in Durif et al. (2017).
//'
//' @references
//' Durif, G., Modolo, L., Mold, J.E., Lambert-Lacroix, S., Picard, F., 2017.
//' Probabilistic Count Matrix Factorization for Single Cell Expression Data
//' Analysis. arXiv:1710.11028 [stat].
//'
//' @author
//' Ghislain Durif, \email{gd.dev@libertymail.net}
//'
//' @param X a count data matrix of dimension \code{n x p}.
//' @param K integer, required dimension of the subspace for the latent
//' representation.
//' @param U matrix of dimension \code{n x K}, initial values for the factor
//' matrix \code{U}. It is used to intialized the variational parameter of the
//' varitional distribution over the factor \code{U}. Default is NULL and
//' variational parameters over \code{U} are intialized otherwise.
//' Note: if you supply \code{U} input parameter, you should supply
//' \code{V} input parameter.
//' @param V matrix of dimension \code{p x K}, initial values for the factor
//' matrix \code{V}. It is used to intialized the variational parameter of the
//' varitional distribution over the factor \code{V}. Default is NULL and
//' variational parameters over \code{V} are intialized otherwise.
//' Note: if you supply \code{V} input parameter, you should supply
//' \code{U} input parameter.
//' @param verbose boolean indicating verbosity. Default is True.
//' @param monitor boolean indicating if model related measures
//' (log-likelihood, deviance, Bregman divergence between \eqn{X}
//' and \eqn{UV^t}) should be computed. Default is True.
//' @param iter_max integer, maximum number of iterations after which the
//' optimization is stopped even when the algorithm did not converge.
//' Default is 1000.
//' @param iter_min integer, minimum number of iterations without checking
//' convergence. Default is 500.
//' @param init_mode string, intialization mode to choose between "random",
//' "nmf". Default value is "random". Unused if initial values are given
//' for U and V or a1, a2, b1 and b2.
//' @param epsilon double precision parameter to assess convergence, i.e.
//' the convergence is reached when the gap (absolute or normalized) between
//' two iterates becomes smaller than epsilon. Default is 1e-2.
//' @param additional_iter integer, number of successive iterations during
//' which the convergence criterion should be verified to assess convergence.
//' @param conv_mode \{0,1,2\}-valued indicator setting how to assess
//' convergence: 0 for absolute gap, 1 for normalized gap between
//' two iterates and 2 for custom criterion (depends on the considered
//' method/model). Default is 1.
//' @param ninit integer, number of initialization to consider. In multiple
//' initialization mode (>1), the algorithm is run for \code{iter_init}
//' iterations with mutliple seeds and the best one (regarding the optimization
//' criterion) is kept. Default value is 1.
//' @param iter_init integer, number of iterations during which the algorithms
//' is run in multi-initialization mode. Default value is 100.
//' @param ncores integer indicating the number of cores to use for
//' parallel computation. Default is 1 and no multi-threading is used.
//' @param reorder_factor boolean indicating if factors should be reordered
//' according to the model-related deviance criterion. Default value is True.
//' @param seed positive integer, seed value for random number generator.
//' Default is NULL and the seed is set based on the current time.
//' @param a1 matrix of dimension \code{n x K}, initial values for the
//' variational shape parameter of the Gamma variational distribution over the
//' factor \code{U}. Default is NULL and each row \code{i} in \code{a1} is
//' randomly initialized from a Gamma distribution of parameters
//' \code{(1,K/mean(X)_i)} where \code{mean(X)_i} is the rowwise mean of
//' the corresponding row in the input data matrix \code{X}.
//' This input parameter is not used if the input parameter \code{U} is not NULL.
//' @param a2 matrix of dimension \code{n x K}, initial values for the
//' variational rate parameter of the Gamma variational distribution over the
//' factor \code{U}. Default is NULL and \code{a2} is intialized with
//' a matrix of 1. This input parameter is not used if the input
//' parameter \code{U} is not NULL.
//' @param b1 matrix of dimension \code{p x K}, initial values for the
//' variational shape parameter of the Gamma variational distribution over the
//' factor \code{V}. Default is NULL and each row \code{j} in \code{b1} is
//' randomly initialized from a Gamma distribution of parameters
//' \code{(1,K/mean(X)_j)} where \code{mean(X)_j} is the colwise mean of
//' the corresponding column in the input data matrix \code{X}.
//' This input parameter is not used if the input parameter \code{V} is not NULL.
//' @param b2 matrix of dimension \code{p x K}, initial values for the
//' variational rate parameter of the Gamma variational distribution over the
//' factor \code{V}. Default is NULL and \code{a2} is intialized with
//' a matrix of 1. This input parameter is not used if the input
//' parameter \code{V} is not NULL.
//' @param alpha1 matrix of dimension \code{n x K}, initial values for the
//' prior shape parameter of the Gamma variational distribution over the
//' factor \code{U}. Default is NULL and \code{alpha1} is initialized based
//' on update rules derived in Durif et al. (2017). This input parameter is
//' not used if the input parameter \code{U} is not NULL. Note: in the
//' standard GaP factor model, all row of \code{alpha1} should be identical.
//' @param alpha2 matrix of dimension \code{n x K}, initial values for the
//' prior rate parameter of the Gamma variational distribution over the
//' factor \code{U}. Default is NULL and \code{alpha2} is initialized based
//' on update rules derived in Durif et al. (2017). This input parameter is
//' not used if the input parameter \code{U} is not NULL. Note: in the
//' standard GaP factor model, all row of \code{alpha2} should be identical.
//' @param beta1 matrix of dimension \code{p x K}, initial values for the
//' prior shape parameter of the Gamma variational distribution over the
//' factor \code{V}. Default is NULL and \code{beta1} is initialized based
//' on update rules derived in Durif et al. (2017). This input parameter is
//' not used if the input parameter \code{U} is not NULL. Note: in the
//' standard GaP factor model, all row of \code{beta1} should be identical.
//' @param beta2 matrix of dimension \code{p x K}, initial values for the
//' prior rate parameter of the Gamma variational distribution over the
//' factor \code{V}. Default is NULL and \code{beta2} is initialized based
//' on update rules derived in Durif et al. (2017). This input parameter is
//' not used if the input parameter \code{U} is not NULL. Note: in the
//' standard GaP factor model, all row of \code{beta2} should be identical.
//'
//' Note: input parameters \code{a1}, \code{a2}, \code{b1}, \code{b2} should
//' all be set to be used. Similarly, \code{alpha1}, \code{alpha2},
//' \code{beta1}, \code{beta2} should all be set to be used.
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
//' \item{loss}{vector of size \code{nb_iter}, values of the loss
//' across iterations.}
//' \item{monitor}{list of monitored criteria if \code{conv_mode} input
//' parameter was set to 1, not returned otherwise:
//'     \code{norm_gap}: vector of size \code{nb_iter}, values of normalized
//'     gap between iterates across iterations.
//'     \code{abs_gap}: vector of size \code{nb_iter}, values of absolute
//'     gap between iterates across iterations.
//'     \code{loglikelihood}: vector of size \code{nb_iter}, values of the
//'     log-likelihood associated to the Poisson NMF model across iterations.
//'     \code{deviance}: vector of size \code{nb_iter}, values of the
//'     deviance associated to the Poisson NMF model across iterations.}
//' \item{variational_params}{list of estimated value for variational
//' parameters:
//'     \code{a1} and \code{a2}: matrix of dimension \code{n x K}.
//'     \code{b1} and \code{b2}: matrix of dimension \code{p x K}.}
//' \item{hyper_params}{list of estimated value for hyper parameters:
//'     \code{alpha1} and \code{alpha2}: matrix of dimension \code{n x K}.
//'     \code{beta1} and \code{beta2}: matrix of dimension \code{p x K}.}
//' \item{stats}{list of exhaustive statistics relaed to the variational
//' distribution:
//'     \code{EU}: expectation of U, matrix of dimension \code{n x K}.
//'     \code{ElogU}: expectation of log U, matrix of dimension \code{n x K}.
//'     \code{EV}: expectation of V, matrix of dimension \code{p x K}.
//'     \code{ElogV}: expectation of log V, matrix of dimension \code{p x K}.}
//' \item{seed}{positive integer value corresponding to the value given as
//' seed in input parameter for random number generation.}
//'
//' @examples
//' \dontrun{
//' ## generate data
//' n <- 100
//' p <- 200
//' K <- 10
//' factorU <- generate_factor_matrix(n, K, ngroup=3, average_signal=60,
//'                                   group_separation=0.8,
//'                                   distribution="gamma",
//'                                   shuffle_feature=TRUE)
//' factorV <- generate_factor_matrix(p, K, ngroup=2, average_signal=60,
//'                                   group_separation=0.8,
//'                                   distribution="gamma",
//'                                   shuffle_feature=TRUE)
//' U <- factorU$factor_matrix
//' V <- factorV$factor_matrix
//' count_data <- generate_count_matrix(n, p, K, U, V)
//' X <- count_data$X
//' ## or use your own data as a count matrix
//' ## of dimension cells x genes (individuals x features)
//' ## run matrix factorization algorithm
//' res <- run_gap_factor(X, K, verbose=FALSE)
//' }
//'
//' @export
// [[Rcpp::export]]
SEXP run_gap_factor(SEXP X, int K,
                    Rcpp::Nullable<MatrixXd> U = R_NilValue,
                    Rcpp::Nullable<MatrixXd> V = R_NilValue,
                    bool verbose = true, bool monitor = true,
                    int iter_max = 1000, int iter_min = 500,
                    Rcpp::CharacterVector init_mode = "random",
                    double epsilon = 1e-2, int additional_iter = 10,
                    int conv_mode = 1, int ninit = 1,
                    int iter_init = 100, int ncores = 1,
                    bool reorder_factor = true,
                    Rcpp::Nullable<uint32_t> seed = R_NilValue,
                    Rcpp::Nullable<MatrixXd> a1 = R_NilValue,
                    Rcpp::Nullable<MatrixXd> a2 = R_NilValue,
                    Rcpp::Nullable<MatrixXd> b1 = R_NilValue,
                    Rcpp::Nullable<MatrixXd> b2 = R_NilValue,
                    Rcpp::Nullable<MatrixXd> alpha1 = R_NilValue,
                    Rcpp::Nullable<MatrixXd> alpha2 = R_NilValue,
                    Rcpp::Nullable<MatrixXd> beta1 = R_NilValue,
                    Rcpp::Nullable<MatrixXd> beta2 = R_NilValue) {

    Rcpp::List output = wrapper_gap_factor<gap_factor_model, variational_em_algo>(X, K, U, V, verbose, monitor,
                                                                                  iter_max, iter_min, init_mode,
                                                                                  epsilon, additional_iter, conv_mode,
                                                                                  ninit, iter_init, ncores,
                                                                                  reorder_factor, seed,
                                                                                  a1, a2, b1, b2,
                                                                                  alpha1, alpha2, beta1, beta2);
    output.attr("class") = "pCMF";
    return output;
}
