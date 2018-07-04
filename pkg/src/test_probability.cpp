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

#include <boost/math/special_functions/digamma.hpp>
#include <math.h>
#include <RcppEigen.h>
#include <testthat.h>

#include "utils/matrix.h"
#include "utils/probability.h"

// [[Rcpp::depends(BH)]]
using boost::math::digamma;

// [[rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision

// Testing functions from utils/likelihood.h
context("Probability unit tests") {

    test_that("Gamma expectation") {
        double shape = 5;
        double rate = 10;
        double res = shape/rate;
        expect_true(std::abs(probability::Egamma(shape, rate) - res) < 1E-8);
    }

    test_that("Gamma expectation matrix version") {
        double shape = 5;
        double rate = 10;
        double res = shape/rate;
        MatrixXd param1(1,1);
        MatrixXd param2(1,1);
        MatrixXd res0(1,1);
        param1(0,0) = shape;
        param2(0,0) = rate;
        probability::Egamma(param1, param2, res0);
        expect_true(std::abs(res0(0,0) - res) < 1E-8);
    }

    test_that("Gamma log-expectation") {
        double shape = 5;
        double rate = 10;
        double res = digamma(shape) - log(rate);
        expect_true(std::abs(probability::ElogGamma(shape, rate) - res) < 1E-8);
    }

    test_that("Gamma log-expectation matrix version") {
        double shape = 5;
        double rate = 10;
        double res = digamma(shape) - log(rate);
        MatrixXd param1(1,1);
        MatrixXd param2(1,1);
        MatrixXd res0(1,1);
        param1(0,0) = shape;
        param2(0,0) = rate;
        probability::ElogGamma(param1, param2, res0);
        expect_true(std::abs(res0(0,0) - res) < 1E-8);
    }

    test_that("Gamma entropy") {
        double shape = 5;
        double rate = 10;
        double res = shape - log(rate) + lgamma(shape) + (1-shape) * digamma(shape);
        expect_true(std::abs(probability::gamma_entropy(shape, rate) - res) < 1E-8);
    }

    test_that("Gamma entropy matrix version") {
        double shape = 5;
        double rate = 10;
        double res = shape - log(rate) + lgamma(shape) + (1-shape) * digamma(shape);
        MatrixXd param1(1,1);
        MatrixXd param2(1,1);
        MatrixXd res0(1,1);
        param1(0,0) = shape;
        param2(0,0) = rate;
        probability::gamma_entropy(param1, param2, res0);
        expect_true(std::abs(res0(0,0) - res) < 1E-8);
    }

    test_that("sum of Gamma entropy matrix version") {
        double shape = 5;
        double rate = 10;
        double res = shape - log(rate) + lgamma(shape) + (1-shape) * digamma(shape);
        MatrixXd param1(1,1);
        MatrixXd param2(1,1);
        param1(0,0) = shape;
        param2(0,0) = rate;
        double res0 = probability::gamma_entropy(param1, param2);
        expect_true(std::abs(res0 - res) < 1E-8);
    }

    test_that("Bregman Poisson") {
        expect_true(probability::bregman_poisson(1,1) == 0);
        expect_true(probability::bregman_poisson(10,10) == 0);
        double x = 5;
        double y = 10;
        double res = x * log(x/y) - x + y;
        expect_true(std::abs(probability::bregman_poisson(x,y) - res) < 1E-8);
    }

    test_that("Matrix Bregman divergence") {
        int nrow = 4;
        int ncol = 3;
        MatrixXd X(nrow,ncol);
        MatrixXd Y(nrow,ncol);
        X <<  2,  4,  3,
             10,  7,  3,
              9,  3, 11,
              3,  9,  3;
        Y <<  7, 11,  1,
              2, 10,  4,
              3,  2,  6,
              7,  6,  2;
        double res1 = matrix::elem_wise_bivar_func_sum(X, Y, nrow, ncol, probability::bregman_poisson);
        double res2 = 23.57360306;
        expect_true(std::abs(res1 - res2) < 1E-8);
    }

}

#endif // _DEV
