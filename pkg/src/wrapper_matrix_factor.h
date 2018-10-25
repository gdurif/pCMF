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
* \brief wrapper template for matrix factorization algorithm
* (such as Poisson NMF or Gamma Poisson Factor model)
* \author Ghislain Durif
* \version 1.0
* \date 06/03/2018
*/

#ifndef MATRIX_FACTOR_WRAPPER_H
#define MATRIX_FACTOR_WRAPPER_H

#if defined(_OPENMP)
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#endif

#include <Rcpp.h>
#include <RcppEigen.h>
#include <stdio.h>

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::MatrixXi;                  // variable size matrix, integer
using Eigen::Map;                       // 'map' rather than copy

/*!
 * \fn template wrapper to run a matrix factorization algorithm
 *
 * \tparam model the statistical model considered for the inference framework
 * \tparam algo an inference/optimization algorithm
 *
 * \param X a count data matrix of dimension \code{n x p}.
 * \param K integer, required dimension of the subspace for the latent
 * representation.
 * \param U matrix of dimension \code{n x K}, initial values for the factor
 * matrix \code{U}. Default is NULL and U is randomly initialized from
 * a Uniform distribution on (0,1). Note: if you supply \code{U} input
 * parameter, you should supply \code{V} input parameter.
 * \param V matrix of dimension \code{p x K}, initial values for the factor
 * matrix \code{V}. Default is NULL and V is randomly initialized from
 * a Uniform distribution on (0,1). Note: if you supply \code{U} input
 * parameter, you should supply \code{V} input parameter.
 * \param verbose boolean indicating verbosity. Default is True.
 * \param monitor boolean indicating if model related measures
 * (log-likelihood, deviance, Bregman divergence between \eqn{X}
 * and \eqn{UV^t}) should be computed. Default is True.
 * \param iter_max integer, maximum number of iterations after which the
 * optimization is stopped even when the algorithm did not converge.
 * Default is 1000.
 * \param iter_min integer, minimum number of iterations without checking
 * convergence. Default is 500.
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
 * criterion) is kept. If \code{U} and \code{V} are supplied, each seed uses
 * a pertubated version of \code{U} and \code{V} as initial values.
 * Default value is 1.
 * \param iter_init integer, number of iterations during which the algorithms is
 * run in multi-initialization mode. Default value is 100.
 * \param ncores integer indicating the number of cores to use for
 * parallel computation. Default is 1 and no multi-threading is used.
 * \param reorder_factor boolean indicating if factors should be reordered
 * according to the model-related deviance criterion. Default value is True.
 * \param seed positive integer, seed value for random number generator.
 * Default is NULL and the seed is set based on the current time.
 */
template <class model, template<typename> class algo>
SEXP wrapper_matrix_factor(SEXP X, int K,
                           Rcpp::Nullable<MatrixXd> U = R_NilValue,
                           Rcpp::Nullable<MatrixXd> V = R_NilValue,
                           bool verbose = true, bool monitor = true,
                           int iter_max = 1000, int iter_min = 500,
                           double epsilon = 1e-2, int additional_iter = 10,
                           int conv_mode = 1, int ninit = 1,
                           int iter_init = 100, int ncores = 1,
                           bool reorder_factor = true,
                           Rcpp::Nullable<uint32_t> seed = R_NilValue) {

    MatrixXd Xin;
    MatrixXd Uin;
    MatrixXd Vin;

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

    if((U.isNotNull() && !V.isNotNull()) || (!U.isNotNull() && V.isNotNull())) {
        Rcpp::stop("You should supply initial values for both U and V or for neither of them");
    }

    if(U.isNotNull() && V.isNotNull()) {
        Uin = Rcpp::as< Map<MatrixXd> >(U);
        Vin = Rcpp::as< Map<MatrixXd> >(V);
        if((Uin.rows() != n) || (Uin.cols() != K) || (Vin.rows() != p) || (Vin.cols() != K)) {
            Rcpp::stop("Wrong dimension for U and/or V input matrix");
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
            if(U.isNotNull() && V.isNotNull()) {
                tmp_model.init(Uin, Vin);
                tmp_model.perturb_param(rng, std::max(Uin.maxCoeff(), Vin.maxCoeff())/K);
            } else {
                tmp_model.init_random(rng);
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

        if(U.isNotNull() && V.isNotNull()) {
            my_model.init(Uin, Vin);
        } else {
            my_model.init_random(rng);
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

#endif // MATRIX_FACTOR_WRAPPER_H
