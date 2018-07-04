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

#ifdef _DEV

#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <stdio.h>
#include <time.h>

#include "algorithm_simple_factor.h"
#include "poisson_nmf.h"
#include "utils/matrix.h"
#include "utils/random.h"

#undef context          // context defined in boost
#include <testthat.h>   // last to avoid definition of context from boost included from "utils/internal.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision

using pCMF::simple_factor_algo;
using pCMF::poisson_nmf;

// Testing functions from utils/likelihood.h
context("Simple matrix factorization algorithm unit tests") {

    test_that("poisson nmf") {
        Rcpp::Rcout << "poisson nmf" << std::endl;

        bool verbose = true;
        bool monitor = true;
        int iter_max = 100;
        int iter_min = 50;
        double epsilon = 1e-2;
        int additional_iter = 10;
        int conv_mode = 1;
        int ncores = 1;
        uint32_t seed = 600;

        int n = 80;
        int p = 50;
        int K = 5;

        // parallelizing
#if defined(_OPENMP)
        omp_set_num_threads(ncores);
        Eigen::initParallel();
#endif

        Rcpp::Rcout << "Data generation" << std::endl;
        Rcpp::Rcout << "n = " << n << std::endl;
        Rcpp::Rcout << "p = " << p  << std::endl;
        Rcpp::Rcout << "K = " << K << std::endl;

        myRandom::RNGType rng = myRandom::rngInit(3);

        MatrixXd X(n, p);
        MatrixXd U(n, K);
        MatrixXd V(p, K);

        myRandom::rUnif(U, n, K, 0, 1, rng);
        myRandom::rUnif(V, p, K, 0, 1, rng);

        myRandom::rPoisson(X, n, p, U * V.transpose(), rng);

        // set algo
        Rcpp::Rcout << "Declaration" << std::endl;
        simple_factor_algo<poisson_nmf> my_model(X.rows(), X.cols(), K, X,
                                                 verbose, monitor, iter_max, iter_min,
                                                 epsilon, additional_iter, conv_mode);

        // init algo
        Rcpp::Rcout << "Init with given values" << std::endl;
        my_model.init(U, V);

        // random init
        Rcpp::Rcout << "Random init" << std::endl;
        my_model.init_random(rng);

        // run algo
        Rcpp::Rcout << "Optimization" << std::endl;
        Rcpp::Rcout << "ncores = " << ncores << std::endl;
        time_t start,end;
        time (&start);
        my_model.optimize();
        time (&end);
        double dif = difftime (end,start);

        // over
        Rcpp::Rcout << "Optimization done in " << dif << " sec" << std::endl;

        // factor
        MatrixXd hatU;
        MatrixXd hatV;
        my_model.get_factor(hatU, hatV);
        expect_true(hatU.rows() == n);
        expect_true(hatU.cols() == K);
        expect_true(hatV.rows() == p);
        expect_true(hatV.cols() == K);

    }
}

#endif // _DEV
