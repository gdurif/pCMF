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

context("utils functions")
test_that("getter unit tests", {

    res <- 0

    expect_that(getU(res), throws_error("wrong model in input"))
    expect_that(getV(res), throws_error("wrong model in input"))

    n <- 100
    p <- 40
    K <- 5
    U <- matrix(runif(n*K), nrow=n, ncol=K)
    V <- matrix(runif(n*K), nrow=p, ncol=K)

    UVt <- U %*% t(V)

    X <- matrix(rpois(n*p, as.vector(UVt)), nrow=n, ncol=p)

    res <- run_poisson_nmf(X, K)

    hatU <- getU(res)
    hatV <- getV(res)

    expect_true(nrow(hatU) == n)
    expect_true(ncol(hatU) == K)
    expect_true(nrow(hatV) == p)
    expect_true(ncol(hatV) == K)
})
