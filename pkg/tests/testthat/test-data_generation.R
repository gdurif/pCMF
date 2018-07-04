### Copyright 2018 Ghislain DURIF
###
### This file is part of the `pCMF' library for R and related languages.
### It is made available under the terms of the GNU General Public
### License, version 2, or at your option, any later version,
### incorporated herein by reference.
###
### This program is distributed in the hope that it will be
### useful, but WITHOUT ANY WARRANTY; without even the implied
### warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
### PURPOSE.  See the GNU General Public License for more
### details.
###
### You should have received a copy of the GNU General Public
### License along with this program; if not, write to the Free
### Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
### MA 02111-1307, USA

context("data generation")
test_that("Generation of factor matrix unit tests", {

    n <- 100
    K <- 5

    res <- generate_factor_matrix(nfeature=n, K=K, ngroup=2, average_signal=1,
                                  group_separation=0.5,
                                  distribution="uniform",
                                  shuffle_feature=TRUE,
                                  prop_noise_feature=0,
                                  noise_level=0.5,
                                  seed=NULL)

    expect_true(is.list(res))
    expect_true(is.matrix(res$factor_matrix))
    expect_true(is.vector(res$feature_order))
    expect_true(is.vector(res$feature_label))

    expect_true(all(res$feature_label %in% c(1,2)))

    U <- res$factor_matrix

    expect_true(nrow(U) == n)
    expect_true(ncol(U) == K)

})

test_that("Generation of count matrix unit tests", {

    n <- 100
    p <- 50
    K <- 5

    U <- matrix(runif(n*K), n, K)
    V <- matrix(runif(p*K), p, K)

    res <- generate_count_matrix(n, p, K, U, V,
                                 ZI=FALSE, prob1=NULL, rate0=NULL)

    expect_true(is.list(res))
    expect_true(is.matrix(res$X))
    expect_true(is.matrix(res$U))
    expect_true(is.matrix(res$V))
    expect_true(is.logical(res$ZI))
    expect_true(is.null(res$prob1))
    expect_true(is.null(res$rate0))
    expect_true(is.null(res$Xnzi))
    expect_true(is.null(res$ZIind))

    expect_true(nrow(res$X) == n)
    expect_true(ncol(res$X) == p)

    res <- generate_count_matrix(n, p, K, U, V,
                                 ZI=TRUE, rate0=rep(5, p))

    expect_true(is.list(res))
    expect_true(is.matrix(res$X))
    expect_true(is.matrix(res$U))
    expect_true(is.matrix(res$V))
    expect_true(is.logical(res$ZI))
    expect_true(is.vector(res$prob1))
    expect_true(length(res$prob1) == p)
    expect_true(is.vector(res$rate0))
    expect_true(length(res$rate0) == p)
    expect_true(is.matrix(res$Xnzi))
    expect_true(is.matrix(res$ZIind))

    expect_true(nrow(res$Xnzi) == n)
    expect_true(ncol(res$Xnzi) == p)

    expect_true(nrow(res$ZIind) == n)
    expect_true(ncol(res$ZIind) == p)

})
