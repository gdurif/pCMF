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

#include <math.h>
#include <RcppEigen.h>

#include <stdio.h>
#include <Rcpp.h>

#include "utils/likelihood.h"

#undef context          // context defined in boost
#include <testthat.h>   // last to avoid definition of context from boost included from "utils/internal.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::VectorXd;                  // variable size vector, double precision

// Testing functions from utils/likelihood.h
context("Likelihood unit tests") {

    test_that("Gamma log-likelihood") {
        expect_true(likelihood::gamma_loglike(1, 1, 1) == -1);
        double a = 5;
        double b = 3;
        double x = 10;
        double res = log( exp(a * log(b)) * exp((a-1) * log(x)) * exp(-b * x) / tgamma(a));
        expect_true(std::abs(likelihood::gamma_loglike(x, a, b) - res) < 1E-8);
    }

    test_that("Gamma log-likelihood matrix version") {
        double a = 5;
        double b = 3;
        double x = 10;
        double res = log( exp(a * log(b)) * exp((a-1) * log(x)) * exp(-b * x) / tgamma(a));
        MatrixXd X(1,1);
        MatrixXd param1(1,1);
        MatrixXd param2(1,1);
        X(0,0) = x;
        param1(0,0) = a;
        param2(0,0) = b;
        double res0 = likelihood::gamma_loglike(X, param1, param2);
        expect_true(std::abs(res0 - res) < 1E-8);
    }

    test_that("Gamma log-likelihood matrix version with X and logX given") {
        double a = 5;
        double b = 3;
        double x = 10;
        double res = log( exp(a * log(b)) * exp((a-1) * log(x)) * exp(-b * x) / tgamma(a));
        MatrixXd X(1,1);
        MatrixXd logX(1,1);
        MatrixXd param1(1,1);
        MatrixXd param2(1,1);
        X(0,0) = x;
        logX(0,0) = log(x);
        param1(0,0) = a;
        param2(0,0) = b;
        double res0 = likelihood::gamma_loglike(X, logX, param1, param2);
        expect_true(std::abs(res0 - res) < 1E-8);
    }

    test_that("Poisson log-likelihood") {
        expect_true(likelihood::poisson_loglike(1, 1) == -1);
        double r = 5;
        double x = 10;
        double res = log( exp(x*log(r)) * exp(-r) / tgamma(x+1));
        expect_true(std::abs(likelihood::poisson_loglike(x, r) - res) < 1E-8);
    }

    test_that("Poisson saturated log-likelihood") {
        expect_true(likelihood::poisson_loglike(0, 0) == 0);
        expect_true(likelihood::poisson_loglike(1, 1) == -1);
        double x = 10;
        double res = log( exp(x*log(x>0 ? x : 1)) * exp(-x) / tgamma(x+1));
        expect_true(std::abs(likelihood::poisson_loglike(x, x) - res) < 1E-8);
    }

    test_that("Poisson log-likelihood matrix version") {
        double r = 5;
        double x = 10;
        double res = log( exp(x*log(r)) * exp(-r) / tgamma(x+1));
        MatrixXd X(1,1);
        MatrixXd rate(1,1);
        X(0,0) = x;
        rate(0,0) = r;
        double res0 = likelihood::poisson_loglike(X, rate);
        expect_true(std::abs(res0 - res) < 1E-8);
    }

    test_that("Poisson log-likelihood matrix version with colwise intensity") {
        double r = 5;
        double x = 10;
        double res = log( exp(x*log(r)) * exp(-r) / tgamma(x+1));
        MatrixXd X(1,1);
        VectorXd rate(1);
        X(0,0) = x;
        rate(0) = r;
        double res0 = likelihood::poisson_loglike_vec(X, rate);
        expect_true(std::abs(res0 - res) < 1E-8);
    }

    test_that("ZI Poisson log-likelihood matrix version") {
        double r = 5;
        double x = 10;
        double p = 0.5;
        double res = log( p * exp(x*log(r)) * exp(-r) / tgamma(x+1));
        MatrixXd X(1,1);
        MatrixXd rate(1,1);
        MatrixXd prob(1,1);
        X(0,0) = x;
        rate(0,0) = r;
        prob(0,0) = p;
        double res0 = likelihood::zi_poisson_loglike(X, rate, prob);
        expect_true(std::abs(res0 - res) < 1E-8);
    }

    test_that("ZI Poisson log-likelihood matrix version with colwise intensity and drop-out probability") {
        double r = 5;
        double x = 10;
        double p = 0.5;
        double res = log( p * exp(x*log(r)) * exp(-r) / tgamma(x+1));
        MatrixXd X(1,1);
        VectorXd rate(1);
        VectorXd prob(1,1);
        X(0,0) = x;
        rate(0,0) = r;
        prob(0,0) = p;
        double res0 = likelihood::zi_poisson_loglike_vec(X, rate, prob);
        expect_true(std::abs(res0 - res) < 1E-8);
    }

    test_that("Bernoulli log-likelihood matrix version") {
        double p = 0.4;
        double x = 1;
        double res = log( exp(x*log(p)) * exp((1-x)*log(1-p)));
        MatrixXd X(1,1);
        MatrixXd prob(1,1);
        X(0,0) = x;
        prob(0,0) = p;
        double res0 = likelihood::bernoulli_loglike(X, prob);
        expect_true(std::abs(res0 - res) < 1E-8);
    }

    test_that("Bernoulli log-likelihood matrix version with colwise probability") {
        double p = 0.4;
        double x = 1;
        double res = log( exp(x*log(p)) * exp((1-x)*log(1-p)));
        MatrixXd X(1,1);
        VectorXd prob(1);
        X(0,0) = x;
        prob(0) = p;
        double res0 = likelihood::bernoulli_loglike(X, prob);
        expect_true(std::abs(res0 - res) < 1E-8);
    }

}

#endif // _DEV
