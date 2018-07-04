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

#include "utils/internal.h"

#undef context          // context defined in boost
#include <testthat.h>   // last to avoid definition of context from boost included from "utils/internal.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::VectorXd;                  // variable size vector, double precision

// Testing functions from utils/likelihood.h
context("Internal function unit tests") {

    test_that("Digamma inverse") {
        double res;
        res = 7.8834286312;
        expect_true(std::abs(internal::digammaInv(2,5) - res) < 1E-8);
        res = 22026.9657929151;
        expect_true(std::abs(internal::digammaInv(10,5) - res) < 1E-8);
    }

    test_that("Compute estimated variance") {
        double res;
        VectorXd sample;
        sample.setLinSpaced(9,1,9);
        res = 7.5;
        expect_true(std::abs(internal::variance(sample) - res) < 1E-8);
    }

    test_that("Custom log") {
        double x = 0;
        expect_true(internal::custom_log(x) == 0);
        x = 7.5;
        expect_true(internal::custom_log(x) == std::log(x));
    }

    test_that("Custom logit function") {
        expect_true(internal::logit(0.5) == 0);
        expect_true(internal::logit(0) == -30);
        expect_true(internal::logit(1) == 30);
        double x = 0.2;
        double res = std::log(x/(1-x));
        expect_true(std::abs(internal::logit(x) - res) < 1E-8);
        x = 0.2;
        res = std::log(x/(1-x));
        expect_true(std::abs(internal::logit(x) - res) < 1E-8);
    }

    test_that("Custom logit inverse function") {
        expect_true(internal::expit(0) == 0.5);
        expect_true(internal::expit(-30) == 0);
        expect_true(internal::expit(30) == 1);
        double x = -2;
        double res = 1.0 / (1 + std::exp(-x));
        expect_true(std::abs(internal::expit(x) - res) < 1E-8);
        x = 2;
        res = 1.0 / (1 + std::exp(-x));
        expect_true(std::abs(internal::expit(x) - res) < 1E-8);
    }

}

#endif // _DEV
