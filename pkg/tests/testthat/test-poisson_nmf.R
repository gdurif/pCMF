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

context("Poisson NMF")
test_that("Poisson NMF unit tests", {

    n <- 100
    p <- 40
    K <- 5
    U <- matrix(runif(n*K), nrow=n, ncol=K)
    V <- matrix(runif(p*K), nrow=p, ncol=K)

    UVt <- U %*% t(V)

    X <- matrix(rpois(n*p, as.vector(UVt)), nrow=n, ncol=p)

    ## check standard run
    res <- run_poisson_nmf(X, K)

    expect_true(class(res) == "NMF")

    expect_true(is.list(res))
    expect_true(is.list(res$factor))
    expect_true(is.matrix(res$factor$U))
    expect_true(is.matrix(res$factor$V))
    expect_true(is.list(res$convergence))
    expect_true(is.logical(res$convergence$converged))
    expect_true(is.integer(res$convergence$nb_iter))
    expect_true(is.integer(res$convergence$conv_mode))
    expect_true(is.vector(res$convergence$conv_crit))
    expect_true(is.numeric(res$loss))
    expect_true(is.numeric(res$exp_dev))
    expect_true(is.list(res$monitor))
    expect_true(is.vector(res$monitor$abs_gap))
    expect_true(is.vector(res$monitor$norm_gap))
    expect_true(is.vector(res$monitor$loglikelihood))
    expect_true(is.vector(res$monitor$deviance))
    expect_true(is.vector(res$monitor$optim_criterion))
    expect_true(is.null(res$seed))

    rv_coeff <- function(A, B) {
        res1 <- sum(diag(A %*% t(A) %*% B %*% t(B)))
        res2 <- sum(diag(A %*% t(A) %*% A %*% t(A)))
        res3 <- sum(diag(B %*% t(B) %*% B %*% t(B)))
        return(res1 / (sqrt(res2*res3)))
    }

    hatU <- getU(res)
    hatV <- getV(res)

    expect_true(rv_coeff(U, hatU) > 0.8)
    expect_true(rv_coeff(V, hatV) > 0.8)

    ## computation of the deviance percentage
    poisLogLike <- function(X, lambda) {
        X.ij <- as.vector(X)
        lambda.ij <- as.vector(lambda)
        lambda.ij.nnull <- ifelse(lambda.ij==0, 1, lambda.ij)
        return(sum(X.ij * log(lambda.ij.nnull) - lambda.ij - lfactorial(X.ij)))
    }

    lambda <- hatU %*% t(hatV)
    lambda0 <- matrix(rep(apply(X,2,mean), times=n),nrow=n, ncol=p, byrow=TRUE)
    res1 <- poisLogLike(X, lambda)
    res2 <- poisLogLike(X, X)
    res3 <- poisLogLike(X, lambda0)
    expect_true( abs(res$exp_dev - ( (res1 - res3) / (res2 - res3))) < 0.01 )

    ###### check different input parameter combination
    ## full args
    res <- run_poisson_nmf(X, K, U = NULL, V = NULL,
                           verbose = FALSE, monitor = TRUE,
                           iter_max = 100, iter_min = 50,
                           epsilon = 1e-2, additional_iter = 10,
                           conv_mode = 1, ninit = 1,
                           iter_init = 10, ncores = 1,
                           reorder_factor = FALSE,
                           seed = NULL)
    ## reorder factor
    res <- run_poisson_nmf(X, K, U = NULL, V = NULL,
                           verbose = FALSE, monitor = TRUE,
                           iter_max = 100, iter_min = 50,
                           epsilon = 1e-2, additional_iter = 10,
                           conv_mode = 1, ninit = 1,
                           iter_init = 10, ncores = 1,
                           reorder_factor = TRUE,
                           seed = NULL)
    ## convergence mode
    res <- run_poisson_nmf(X, K, U = U, V = V,
                           verbose = FALSE, monitor = TRUE,
                           iter_max = 100, iter_min = 50,
                           epsilon = 1e-2, additional_iter = 10,
                           conv_mode = 2, ninit = 1,
                           iter_init = 10, ncores = 1,
                           reorder_factor = TRUE,
                           seed = NULL)
    ## supply U and V
    res <- run_poisson_nmf(X, K, U = U, V = V,
                           verbose = FALSE, monitor = TRUE,
                           iter_max = 100, iter_min = 50,
                           epsilon = 1e-2, additional_iter = 10,
                           conv_mode = 1, ninit = 1,
                           iter_init = 10, ncores = 1,
                           reorder_factor = TRUE,
                           seed = NULL)

    ## mutli-init
    res <- run_poisson_nmf(X, K, U = NULL, V = NULL,
                           verbose = FALSE, monitor = TRUE,
                           iter_max = 100, iter_min = 50,
                           epsilon = 1e-2, additional_iter = 10,
                           conv_mode = 1, ninit = 10,
                           iter_init = 10, ncores = 1,
                           reorder_factor = TRUE,
                           seed = NULL)

    ## multi-init with U and V
    res <- run_poisson_nmf(X, K, U = U, V = V,
                           verbose = FALSE, monitor = TRUE,
                           iter_max = 100, iter_min = 50,
                           epsilon = 1e-2, additional_iter = 10,
                           conv_mode = 1, ninit = 10,
                           iter_init = 10, ncores = 1,
                           reorder_factor = TRUE,
                           seed = NULL)

})
