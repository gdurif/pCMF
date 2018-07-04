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

#include <algorithm>
#include <math.h>
#include <RcppEigen.h>
#include <vector>

#include "utils/internal.h"
#include "utils/random.h"

#undef context          // context defined in boost
#include <testthat.h>   // last to avoid definition of context from boost included from "utils/internal.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::VectorXd;                  // variable size vector, double precision

// Testing functions from utils/likelihood.h
context("Random function unit tests") {

    test_that("Gamma distribution random generator") {
        myRandom::RNGType rng = myRandom::rngInit();

        int n = 100000;
        VectorXd sample(n);

        double shape = 5;
        double rate = 10;

        myRandom::rGamma(sample, n, shape, rate, rng);

        double mean = shape / rate;
        double var = shape / double(std::pow(rate,2));

        double estim_var = internal::variance(sample);

        expect_true(std::abs(sample.mean() - mean) < 1E-1);
        expect_true(std::abs(estim_var - var) < 5E-1);
    }

    test_that("uint32 random generator") {
        myRandom::RNGType rng = myRandom::rngInit();

        int n = 100;
        uint32_t* sample0 = new uint32_t[n];
        myRandom::rInt32(sample0, n, rng);

        std::vector<uint32_t> sample;
        sample.assign(sample0, sample0 + n);

        expect_true(*std::min_element(sample.begin(), sample.end()) >= 0);
        expect_true(*std::max_element(sample.begin(), sample.end()) < std::pow(2,32));

        delete[] sample0;
    }

    test_that("Poisson distribution random generator") {
        myRandom::RNGType rng = myRandom::rngInit();

        int n = 100000;
        VectorXd sample(n);

        double rate = 10;

        myRandom::rPoisson(sample, n, rate, rng);

        double mean = rate;
        double var = rate;

        double estim_var = internal::variance(sample);

        expect_true(std::abs(sample.mean() - mean) < 1E-1);
        expect_true(std::abs(estim_var - var) < 5E-1);
    }

    test_that("Uniform distribution random generator") {
        myRandom::RNGType rng = myRandom::rngInit();

        int n = 1000000;
        VectorXd sample(n);

        double mini = 5;
        double maxi = 20;

        myRandom::rUnif(sample, n, mini, maxi, rng);

        double mean = (mini + maxi) / 2;
        double var = std::pow(maxi-mini,2) / double(12);

        double estim_var = internal::variance(sample);

        expect_true(std::abs(sample.mean() - mean) < 1E-1);
        expect_true(std::abs(estim_var - var) < 5E-1);
    }

}

#endif // _DEV
