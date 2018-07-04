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

#include <RcppEigen.h>
#include <testthat.h>

#include "utils/matrix.h"


// [[rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision

// internal functions for test
double mySum(double x, double y) {
    return(x+y);
}

// Testing functions from utils/likelihood.h
context("Matrix unit tests") {

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
        double res1 = matrix::elem_wise_bivar_func_sum(X, Y, nrow, ncol, mySum);
        double res2 = 128;
        expect_true(std::abs(res1 - res2) < 1E-8);
    }

}

#endif // _DEV
