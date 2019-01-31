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
* \brief wrapper template for variational EM algorithm for matrix factorization
* (such as sparse Gamma-Poisson factor model)
* \author Ghislain Durif
* \version 1.0
* \date 10/04/2018
*/

#ifndef SPARSE_GAP_FACTOR_WRAPPER_H
#define SPARSE_GAP_FACTOR_WRAPPER_H

#if defined(_OPENMP)
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#endif

#include <Rcpp.h>
#include <RcppEigen.h>
#include <stdio.h>
#include <string>

#include "utils/random.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::MatrixXi;                  // variable size matrix, integer
using Eigen::Map;                       // 'map' rather than copy
using Eigen::VectorXd;                  // variable size vector, double precision

using std::string;

/*!
 * \fn template wrapper to run a matrix factorization algorithm based on the
 * hierarchical Gamma Poisson factor model (and derivatives)
 * in its sparse version (over factor V)
 *
 * \tparam model the statistical model considered for the inference framework
 * \tparam algo an inference/optimization algorithm
 *
 * \param X a count data matrix of dimension \code{n x p}.
 * \param K integer, required dimension of the subspace for the latent
 * representation.
 * \param sel_bound real value in [0,1] used to threshold sparsity probabilities
 * for factor V
 * \param U matrix of dimension \code{n x K}, initial values for the factor
 * matrix \code{U}. It is used to intialized the variational parameter of the
 * varitional distribution over the factor \code{U}. Default is NULL and
 * variational parameters over \code{U} are intialized otherwise.
 * Note: if you supply \code{U} input parameter, you should supply
 * \code{V} input parameter.
 * \param V matrix of dimension \code{p x K}, initial values for the factor
 * matrix \code{V}. It is used to intialized the variational parameter of the
 * varitional distribution over the factor \code{V}. Default is NULL and
 * variational parameters over \code{V} are intialized otherwise.
 * Note: if you supply \code{V} input parameter, you should supply
 * \code{U} input parameter.
 * \param verbose boolean indicating verbosity. Default is True.
 * \param monitor boolean indicating if model related measures
 * (log-likelihood, deviance, Bregman divergence between \eqn{X}
 * and \eqn{UV^t}) should be computed. Default is True.
 * \param iter_max integer, maximum number of iterations after which the
 * optimization is stopped even when the algorithm did not converge.
 * Default is 1000.
 * \param iter_min integer, minimum number of iterations without checking
 * convergence. Default is 500.
 * \param init_mode string, intialization mode to choose between "random",
 * "nmf". Default value is "random". Unused if initial values are given
 * for U and V or a1, a2, b1 and b2.
 * \param epsilon double precision parameter to assess convergence, i.e.
 * the convergence is reached when the gap (absolute or normalized) between
 * two iterates becomes smaller than epsilon. Default is 1e-2.
 * \param additional_iter integer, number of successive iterations during
 * which the convergence criterion should be verified to assess convergence.
 * \param conv_mode \{0,1,2\}-valued indicator setting how to assess
 * convergence: 0 for absolute gap, 1 for normalized gap between
 * two iterates and 2 for custom criterion (depends on the considered
 * method/model). Default is 1.
 * \param ninit integer, number of initialization to consider. In multiple
 * initialization mode (>1), the algorithm is run for \code{iter_init}
 * iterations with mutliple seeds and the best one (regarding the optimization
 * criterion) is kept. Default value is 1.
 * \param iter_init integer, number of iterations during which the algorithms is
 * run in multi-initialization mode. Default value is 100.
 * \param ncores integer indicating the number of cores to use for
 * parallel computation. Default is 1 and no multi-threading is used.
 * \param reorder_factor boolean indicating if factors should be reordered
 * according to the model-related deviance criterion. Default value is True.
 * \param seed positive integer, seed value for random number generator.
 * Default is NULL and the seed is set based on the current time.
 * \param a1 matrix of dimension \code{n x K}, initial values for the
 * variational shape parameter of the Gamma variational distribution over the
 * factor \code{U}. Default is NULL and each row \code{i} in \code{a1} is
 * randomly initialized from a Gamma distribution of parameters
 * \code{(1,K/mean(X)_i)} where \code{mean(X)_i} is the rowwise mean of
 * the corresponding row in the input data matrix \code{X}.
 * This input parameter is not used if the input parameter \code{U} is not NULL.
 * \param a2 matrix of dimension \code{n x K}, initial values for the
 * variational rate parameter of the Gamma variational distribution over the
 * factor \code{U}. Default is NULL and \code{a2} is intialized with
 * a matrix of 1. This input parameter is not used if the input
 * parameter \code{U} is not NULL.
 * \param b1 matrix of dimension \code{p x K}, initial values for the
 * variational shape parameter of the Gamma variational distribution over the
 * factor \code{V}. Default is NULL and each row \code{j} in \code{b1} is
 * randomly initialized from a Gamma distribution of parameters
 * \code{(1,K/mean(X)_j)} where \code{mean(X)_j} is the colwise mean of
 * the corresponding column in the input data matrix \code{X}.
 * This input parameter is not used if the input parameter \code{V} is not NULL.
 * \param a2 matrix of dimension \code{p x K}, initial values for the
 * variational rate parameter of the Gamma variational distribution over the
 * factor \code{V}. Default is NULL and \code{a2} is intialized with
 * a matrix of 1. This input parameter is not used if the input
 * parameter \code{V} is not NULL.
 * \param alpha1 matrix of dimension \code{n x K}, initial values for the
 * prior shape parameter of the Gamma variational distribution over the
 * factor \code{U}. Default is NULL and \code{alpha1} is initialized based
 * on update rules derived in Durif et al. (2018). This input parameter is
 * not used if the input parameter \code{U} is not NULL.
 * \param alpha2 matrix of dimension \code{n x K}, initial values for the
 * prior rate parameter of the Gamma variational distribution over the
 * factor \code{U}. Default is NULL and \code{alpha2} is initialized based
 * on update rules derived in Durif et al. (2018). This input parameter is
 * not used if the input parameter \code{U} is not NULL.
 * \param beta1 matrix of dimension \code{p x K}, initial values for the
 * prior shape parameter of the Gamma variational distribution over the
 * factor \code{V}. Default is NULL and \code{beta1} is initialized based
 * on update rules derived in Durif et al. (2018). This input parameter is
 * not used if the input parameter \code{U} is not NULL.
 * \param beta2 matrix of dimension \code{p x K}, initial values for the
 * prior rate parameter of the Gamma variational distribution over the
 * factor \code{V}. Default is NULL and \code{beta2} is initialized based
 * on update rules derived in Durif et al. (2018). This input parameter is
 * not used if the input parameter \code{U} is not NULL.
 * \param prob_S matrix of dimension \code{p x K}, initial values for the
 * variational probability parameter for the sparsity indicator matrix \code{S}
 * over the factor \code{V}. Default is NULL and \code{prob_S} is initialized
 * randomly.
 * \param prior_S vector of length \code{p}, initial values for the
 * prior probability parameter for the sparsity indicator matrix \code{S} over
 * the factor \code{V}. Default is NULL and \code{prior_S} is initialized
 * randomly.
 * \param prob_D matrix of dimension \code{n x p}, initial values for the
 * variational probability parameter for the drop-out indicator matrix \code{D}
 * accounting for zero-inflation in \code{X}. Default is NULL
 * and \code{prob_D} is initialized by the frequence of zero in the
 * corresponding column of \code{X}. This parameter is not used for model
 * without zero-inflation.
 * \param prior_D vector of length \code{p}, initial values for the
 * prior probability parameter for the drop-out indicator matrix \code{D}
 * accounting for zero-inflation in \code{X}. Default is NULL and
 * \code{prior_D} is initialized by the frequence of zero in the
 * corresponding column of \code{X}. This parameter is not used for model
 * without zero-inflation.
 *
 * Note: input parameters \code{a1}, \code{a2}, \code{b1}, \code{b2} should
 * be all set to be used.
 */
template <class model, template<typename> class algo>
SEXP wrapper_sparse_gap_factor(SEXP X, int K, double sel_bound,
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
                               Rcpp::Nullable<MatrixXd> beta2 = R_NilValue,
                               Rcpp::Nullable<MatrixXd> prob_S = R_NilValue,
                               Rcpp::Nullable<VectorXd> prior_S = R_NilValue,
                               Rcpp::Nullable<MatrixXd> prob_D = R_NilValue,
                               Rcpp::Nullable<VectorXd> prior_D = R_NilValue) {

    MatrixXd Xin;
    MatrixXd Uin;
    MatrixXd Vin;
    MatrixXd a1in;
    MatrixXd a2in;
    MatrixXd b1in;
    MatrixXd b2in;
    MatrixXd alpha1in;
    MatrixXd alpha2in;
    MatrixXd beta1in;
    MatrixXd beta2in;
    MatrixXd prob_Sin;
    MatrixXd prior_Sin;
    MatrixXd prob_Din;
    MatrixXd prior_Din;

    std::string init_mode_st = Rcpp::as<std::string>(init_mode);

    // check input
    if(TYPEOF(X) == INTSXP) {
        MatrixXi Xtmp = Rcpp::as< Map<MatrixXi> >(X);
        Xin = Xtmp.cast<double>();
    } else if(TYPEOF(X) == REALSXP) {
        Xin = Rcpp::as< Map<MatrixXd> >(X);
    } else {
        Rcpp::stop("Input matrix X should be integer or real valued.");
    }

    int n = Xin.rows();
    int p = Xin.cols();

    if(K>std::min(n, p)) {
        Rcpp::stop("Input paramter K should be lower than min(n,p) where n and p are the number of rows and columns in input matrix X");
    }

    if(sel_bound<0 || sel_bound>1) {
        Rcpp::stop("Input parameter sel_bound should be a real value between 0 and 1");
    }

    if((U.isNotNull() && !V.isNotNull()) || (!U.isNotNull() && V.isNotNull())) {
        Rcpp::stop("You should supply initial values for both U and V or for neither of them");
    }

    if(U.isNotNull() && V.isNotNull()) {
        Uin = Rcpp::as< Map<MatrixXd> >(U);
        Vin = Rcpp::as< Map<MatrixXd> >(V);
        if((Uin.rows() != n) || (Uin.cols() != K) || (Vin.rows() != p) || (Vin.cols() != K)) {
            Rcpp::stop("Wrong dimension for U and/or V input matrices");
        }
    }

    if(a1.isNotNull() || a2.isNotNull() || b1.isNotNull() || b2.isNotNull()) {
        if(!(a1.isNotNull() && a2.isNotNull() && b1.isNotNull() && b2.isNotNull())) {
            Rcpp::stop("You should supply initial values for all a1, a2, b1, b2 or for none of them");
        } else {
            a1in = Rcpp::as< Map<MatrixXd> >(a1);
            a2in = Rcpp::as< Map<MatrixXd> >(a2);
            b1in = Rcpp::as< Map<MatrixXd> >(b1);
            b2in = Rcpp::as< Map<MatrixXd> >(b2);
            if((a1in.rows() != n) || (a1in.cols() != K) || (a2in.rows() != n) || (a2in.cols() != K)
                   || (b1in.rows() != p) || (b1in.cols() != K) || (b2in.rows() != p) || (b2in.cols() != K)) {
                Rcpp::stop("Wrong dimension for a1, a2, b1 and/or b2 input matrices");
            }
        }
    }

    if(alpha1.isNotNull() || alpha2.isNotNull() || beta1.isNotNull() || beta2.isNotNull()) {
        if(!(alpha1.isNotNull() && alpha2.isNotNull() && beta1.isNotNull() && beta2.isNotNull())) {
            Rcpp::stop("You should supply initial values for all alpha1, alpha2, beta1, beta2 or for none of them");
        } else {
            alpha1in = Rcpp::as< Map<MatrixXd> >(alpha1);
            alpha2in = Rcpp::as< Map<MatrixXd> >(alpha2);
            beta1in = Rcpp::as< Map<MatrixXd> >(beta1);
            beta2in = Rcpp::as< Map<MatrixXd> >(beta2);
            if((alpha1in.rows() != n) || (alpha1in.cols() != K) || (alpha2in.rows() != n) || (alpha2in.cols() != K)
                   || (beta1in.rows() != p) || (beta1in.cols() != K) || (beta2in.rows() != p) || (beta2in.cols() != K)) {
                Rcpp::stop("Wrong dimension for alpha1, alpha2, beta1 and/or beta2 input matrices");
            }
        }
    }

    if(prob_S.isNotNull() && prior_S.isNotNull()) {
        prob_Sin = Rcpp::as< Map<MatrixXd> >(prob_S);
        prior_Sin = Rcpp::as< Map<VectorXd> >(prior_S);

        // Rcpp::Rcout << "prob_S dim = = " << prob_Sin.rows() << ", " << prob_Sin.cols() << std::endl;
        // Rcpp::Rcout << "prior_S size = " << prior_Sin.size() << std::endl;

        if((prob_Sin.rows() != p) || (prob_Sin.cols() != K) || (prior_Sin.size() != p)) {
            Rcpp::stop("Wrong dimension for prob_S and/or prior_S input matrix/vector");
        }
    }

    if(prob_D.isNotNull() && prior_D.isNotNull()) {
        prob_Din = Rcpp::as< Map<MatrixXd> >(prob_D);
        prior_Din = Rcpp::as< Map<VectorXd> >(prior_D);
        if((prob_Din.rows() != n) || (prob_Din.cols() != p) || (prior_Din.size() != p)) {
            Rcpp::stop("Wrong dimension for prob_D and/or prior_D input matrix/vector");
        }
    }

    // parallelizing
#if defined(_OPENMP)
    omp_set_num_threads(ncores);
    Eigen::initParallel();
#endif

    // random number generator
    myRandom::RNGType rng;

    if(seed.isNotNull()) {
        uint32_t mySeed = Rcpp::as<uint32_t>(seed);
        rng = myRandom::rngInit(mySeed);
    } else {
        rng = myRandom::rngInit();
    }

    // set algo
    if(verbose) Rcpp::Rcout << "Declaration" << std::endl;
    algo<model> my_model(n, p, K, Xin,
                         verbose, monitor, iter_max, iter_min,
                         epsilon, additional_iter, conv_mode);

    // init algo
    if(verbose) Rcpp::Rcout << "Initialization" << std::endl;

    // multiple init ?
    if(ninit > 1) {
        if(verbose) Rcpp::Rcout << "Multiple initialization run" << std::endl;

        algo<model> tmp_model(n, p, K, Xin,
                              verbose, monitor, iter_max, iter_min,
                              epsilon, additional_iter, conv_mode);
        tmp_model.set_iter(iter_init, iter_init, 1e-6);
        tmp_model.set_verbosity(false);
        double best_loss;

        for(int nrun=0; nrun<ninit; nrun++) {
            if(verbose) Rcpp::Rcout << "Run " << nrun << std::endl;
            if(prob_S.isNotNull() && prior_S.isNotNull()) {
                tmp_model.init_sparse_param_gap(sel_bound, prob_Sin, prior_Sin);
            } else {
                tmp_model.init_sparse_param_gap(sel_bound, rng);
            }
            if(prob_D.isNotNull() && prior_D.isNotNull()) {
                tmp_model.init_zi_param_gap(prob_Din, prior_Din);
            } else {
                tmp_model.init_zi_param_gap();
            }
            if(U.isNotNull() && V.isNotNull()) {
                tmp_model.init(Uin, Vin);
                tmp_model.perturb_param(rng, std::max(Uin.maxCoeff(), Vin.maxCoeff())/K);
            } else if(a1.isNotNull() && a2.isNotNull() && b1.isNotNull() && b2.isNotNull()) {
                tmp_model.init_variational_param_gap(a1in, a2in, b1in, b2in);
                tmp_model.perturb_param(rng, std::max(Uin.maxCoeff(), Vin.maxCoeff())/K);
            } else if(alpha1.isNotNull() && alpha2.isNotNull() && beta1.isNotNull() && beta2.isNotNull()) {
                tmp_model.init_hyper_param_gap(alpha1in, alpha2in, beta1in, beta2in);
                tmp_model.perturb_param(rng, std::max(Uin.maxCoeff(), Vin.maxCoeff())/K);
            } else if(a1.isNotNull() && a2.isNotNull() && b1.isNotNull() && b2.isNotNull()
                          && alpha1.isNotNull() && alpha2.isNotNull() && beta1.isNotNull() && beta2.isNotNull()) {
                tmp_model.init_all_param_gap(alpha1in, alpha2in, beta1in, beta2in,
                                             a1in, a2in, b1in, b2in);
                tmp_model.perturb_param(rng, std::max(Uin.maxCoeff(), Vin.maxCoeff())/K);
            }  else if(init_mode_st == "random") {
                tmp_model.init_random(rng);
            } else if(init_mode_st == "nmf") {
                Rcpp::stop("FixMe not implemented yet");
            } else {
                Rcpp::stop("`init_mode` input parameter should be 'random' or 'nmf'");
            }
            tmp_model.set_current_iter(0);
            tmp_model.optimize();
            double tmp_loss = tmp_model.get_loss();
            if(verbose) Rcpp::Rcout << "loss " << tmp_loss << std::endl;
            if(nrun==0) {
                best_loss = tmp_loss;
                my_model = tmp_model;
            } else {
                if(tmp_loss < best_loss) {
                    best_loss = tmp_loss;
                    my_model = tmp_model;
                }
            }
        }

        my_model.set_iter(iter_min, iter_max, epsilon);
        my_model.set_verbosity(verbose);

    } else {
        if(prob_S.isNotNull() && prior_S.isNotNull()) {
            my_model.init_sparse_param_gap(sel_bound, prob_Sin, prior_Sin);
        } else {
            my_model.init_sparse_param_gap(sel_bound, rng);
        }
        if(prob_D.isNotNull() && prior_D.isNotNull()) {
            my_model.init_zi_param_gap(prob_Din, prior_Din);
        } else {
            my_model.init_zi_param_gap();
        }
        if(U.isNotNull() && V.isNotNull()) {
            my_model.init(Uin, Vin);
        } else if(a1.isNotNull() && a2.isNotNull() && b1.isNotNull() && b2.isNotNull()) {
            my_model.init_variational_param_gap(a1in, a2in, b1in, b2in);
        } else if(alpha1.isNotNull() && alpha2.isNotNull() && beta1.isNotNull() && beta2.isNotNull()) {
            my_model.init_hyper_param_gap(alpha1in, alpha2in, beta1in, beta2in);
        } else if(a1.isNotNull() && a2.isNotNull() && b1.isNotNull() && b2.isNotNull()
                      && alpha1.isNotNull() && alpha2.isNotNull() && beta1.isNotNull() && beta2.isNotNull()) {
            my_model.init_all_param_gap(alpha1in, alpha2in, beta1in, beta2in,
                                        a1in, a2in, b1in, b2in);
        } else if(init_mode_st == "random") {
            my_model.init_random(rng);
        } else if(init_mode_st == "nmf") {
            Rcpp::stop("FixMe not implemented yet");
        } else {
            Rcpp::stop("`init_mode` input parameter should be 'random' or 'nmf'");
        }
    }

    // run algo
    if(verbose) Rcpp::Rcout << "Optimization" << std::endl;
    my_model.optimize();

    // order factor
    if(reorder_factor) {
        if(verbose) Rcpp::Rcout << "Order factor" << std::endl;
        my_model.order_factor();
    }

    // Return
    if(verbose) Rcpp::Rcout << "Output" << std::endl;
    Rcpp::List output;
    my_model.get_output(output);
    output.push_back(seed, "seed");
    return output;
};

#endif // SPARSE_GAP_FACTOR_WRAPPER_H
