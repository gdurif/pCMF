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

context("pCMF (Gamma-Poisson factor model)")
test_that("pCMF with standard GaP factor model unit tests", {

    n <- 100
    p <- 40
    K <- 5
    U <- matrix(rgamma(n*K, shape=10, rate=1), nrow=n, ncol=K)
    V <- matrix(rgamma(p*K, shape=10, rate=1), nrow=p, ncol=K)

    UVt <- U %*% t(V)

    X <- matrix(rpois(n*p, as.vector(UVt)), nrow=n, ncol=p)

    ## check standard run
    res <- run_gap_factor(X, K)

    expect_true(class(res) == "pCMF")

    expect_true(is.list(res))
    expect_true(is.list(res$factor))
    expect_true(is.matrix(res$factor$U))
    expect_true(is.matrix(res$factor$V))

    expect_true(is.list(res$variational_params))
    expect_true(is.matrix(res$variational_params$a1))
    expect_true(is.matrix(res$variational_params$a2))
    expect_true(is.matrix(res$variational_params$b1))
    expect_true(is.matrix(res$variational_params$b2))

    expect_true(is.list(res$hyper_params))
    expect_true(is.matrix(res$hyper_params$alpha1))
    expect_true(is.matrix(res$hyper_params$alpha2))
    expect_true(is.matrix(res$hyper_params$beta1))
    expect_true(is.matrix(res$hyper_params$beta2))

    expect_true(is.list(res$stats))
    expect_true(is.matrix(res$stats$EU))
    expect_true(is.matrix(res$stats$EV))
    expect_true(is.matrix(res$stats$ElogU))
    expect_true(is.matrix(res$stats$ElogV))

    expect_true(is.list(res$convergence))
    expect_true(is.logical(res$convergence$converged))
    expect_true(is.integer(res$convergence$nb_iter))
    expect_true(is.integer(res$convergence$conv_mode))
    expect_true(is.vector(res$convergence$conv_crit))

    expect_true(is.numeric(res$loss))
    expect_true(is.numeric(res$exp_dev))

    expect_true(is.vector(res$monitor$abs_gap))
    expect_true(is.vector(res$monitor$norm_gap))
    expect_true(is.vector(res$monitor$loglikelihood))
    expect_true(is.vector(res$monitor$deviance))
    expect_true(is.vector(res$monitor$optim_criterion))

    expect_true(is.null(res$seed))

    ###### check different input parameter combination
    ## full args
    res <- run_gap_factor(X, K, U = NULL, V = NULL,
                          verbose = FALSE, monitor = TRUE,
                          iter_max = 100, iter_min = 50,
                          init_mode = "random",
                          epsilon = 1e-2, additional_iter = 10,
                          conv_mode = 1, ninit = 1,
                          iter_init = 10, ncores = 1,
                          reorder_factor = FALSE, seed = NULL,
                          a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                          alpha1 = NULL, alpha2 = NULL,
                          beta1 = NULL, beta2 = NULL)
    ## reorder factor
    res <- run_gap_factor(X, K, U = NULL, V = NULL,
                          verbose = FALSE, monitor = TRUE,
                          iter_max = 100, iter_min = 50,
                          init_mode = "random",
                          epsilon = 1e-2, additional_iter = 10,
                          conv_mode = 1, ninit = 1,
                          iter_init = 10, ncores = 1,
                          reorder_factor = TRUE, seed = NULL,
                          a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                          alpha1 = NULL, alpha2 = NULL,
                          beta1 = NULL, beta2 = NULL)
    ## convergence mode
    res <- run_gap_factor(X, K, U = NULL, V = NULL,
                          verbose = FALSE, monitor = TRUE,
                          iter_max = 100, iter_min = 50,
                          init_mode = "random",
                          epsilon = 1e-2, additional_iter = 10,
                          conv_mode = 0, ninit = 1,
                          iter_init = 10, ncores = 1,
                          reorder_factor = FALSE, seed = NULL,
                          a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                          alpha1 = NULL, alpha2 = NULL,
                          beta1 = NULL, beta2 = NULL)
    ## supply U and V
    res <- run_gap_factor(X, K, U = U, V = V,
                          verbose = FALSE, monitor = TRUE,
                          iter_max = 100, iter_min = 50,
                          init_mode = "random",
                          epsilon = 1e-2, additional_iter = 10,
                          conv_mode = 1, ninit = 1,
                          iter_init = 10, ncores = 1,
                          reorder_factor = FALSE, seed = NULL,
                          a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                          alpha1 = NULL, alpha2 = NULL,
                          beta1 = NULL, beta2 = NULL)

    ## mutli-init
    res <- run_gap_factor(X, K, U = NULL, V = NULL,
                          verbose = FALSE, monitor = TRUE,
                          iter_max = 100, iter_min = 50,
                          init_mode = "random",
                          epsilon = 1e-2, additional_iter = 10,
                          conv_mode = 1, ninit = 10,
                          iter_init = 10, ncores = 1,
                          reorder_factor = FALSE, seed = NULL,
                          a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                          alpha1 = NULL, alpha2 = NULL,
                          beta1 = NULL, beta2 = NULL)

    ## multi-init with U and V
    res <- run_gap_factor(X, K, U = U, V = V,
                          verbose = FALSE, monitor = TRUE,
                          iter_max = 100, iter_min = 50,
                          init_mode = "random",
                          epsilon = 1e-2, additional_iter = 10,
                          conv_mode = 1, ninit = 10,
                          iter_init = 10, ncores = 1,
                          reorder_factor = FALSE, seed = NULL,
                          a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                          alpha1 = NULL, alpha2 = NULL,
                          beta1 = NULL, beta2 = NULL)

})


test_that("pCMF with standard ZI GaP factor model unit tests", {

    n <- 100
    p <- 40
    K <- 5
    U <- matrix(rgamma(n*K, shape=10, rate=1), nrow=n, ncol=K)
    V <- matrix(rgamma(p*K, shape=10, rate=1), nrow=p, ncol=K)

    UVt <- U %*% t(V)

    X <- matrix(rpois(n*p, as.vector(UVt)), nrow=n, ncol=p)

    D <- matrix(rbinom(n*p, size=1, prob=0.5), nrow=n, ncol=p)

    X <- X * D

    ## check standard run
    res <- run_zi_gap_factor(X, K)

    expect_true(class(res) == "pCMF")

    expect_true(is.list(res))
    expect_true(is.list(res$factor))
    expect_true(is.matrix(res$factor$U))
    expect_true(is.matrix(res$factor$V))

    expect_true(is.list(res$variational_params))
    expect_true(is.matrix(res$variational_params$a1))
    expect_true(is.matrix(res$variational_params$a2))
    expect_true(is.matrix(res$variational_params$b1))
    expect_true(is.matrix(res$variational_params$b2))

    expect_true(is.list(res$hyper_params))
    expect_true(is.matrix(res$hyper_params$alpha1))
    expect_true(is.matrix(res$hyper_params$alpha2))
    expect_true(is.matrix(res$hyper_params$beta1))
    expect_true(is.matrix(res$hyper_params$beta2))

    expect_true(is.list(res$stats))
    expect_true(is.matrix(res$stats$EU))
    expect_true(is.matrix(res$stats$EV))
    expect_true(is.matrix(res$stats$ElogU))
    expect_true(is.matrix(res$stats$ElogV))

    expect_true(is.list(res$ZI_param))
    expect_true(is.matrix(res$ZI_param$prob_D))
    expect_true(is.vector(res$ZI_param$freq_D))
    expect_true(is.vector(res$ZI_param$prior_prob_D))

    expect_true(is.list(res$convergence))
    expect_true(is.logical(res$convergence$converged))
    expect_true(is.integer(res$convergence$nb_iter))
    expect_true(is.integer(res$convergence$conv_mode))
    expect_true(is.vector(res$convergence$conv_crit))

    expect_true(is.numeric(res$loss))
    expect_true(is.numeric(res$exp_dev))

    expect_true(is.vector(res$monitor$abs_gap))
    expect_true(is.vector(res$monitor$norm_gap))
    expect_true(is.vector(res$monitor$loglikelihood))
    expect_true(is.vector(res$monitor$deviance))
    expect_true(is.vector(res$monitor$optim_criterion))

    expect_true(is.null(res$seed))

    ###### check different input parameter combination
    ## full args
    res <- run_zi_gap_factor(X, K, U = NULL, V = NULL,
                             verbose = FALSE, monitor = TRUE,
                             iter_max = 100, iter_min = 50,
                             init_mode = "random",
                             epsilon = 1e-2, additional_iter = 10,
                             conv_mode = 1, ninit = 1,
                             iter_init = 10, ncores = 1,
                             reorder_factor = FALSE, seed = NULL,
                             a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                             alpha1 = NULL, alpha2 = NULL,
                             beta1 = NULL, beta2 = NULL,
                             prob_D = NULL, prior_D = NULL)
    ## reorder factor
    res <- run_zi_gap_factor(X, K, U = NULL, V = NULL,
                             verbose = FALSE, monitor = TRUE,
                             iter_max = 100, iter_min = 50,
                             init_mode = "random",
                             epsilon = 1e-2, additional_iter = 10,
                             conv_mode = 1, ninit = 1,
                             iter_init = 10, ncores = 1,
                             reorder_factor = TRUE, seed = NULL,
                             a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                             alpha1 = NULL, alpha2 = NULL,
                             beta1 = NULL, beta2 = NULL,
                             prob_D = NULL, prior_D = NULL)
    ## convergence mode
    res <- run_zi_gap_factor(X, K, U = NULL, V = NULL,
                             verbose = FALSE, monitor = TRUE,
                             iter_max = 100, iter_min = 50,
                             init_mode = "random",
                             epsilon = 1e-2, additional_iter = 10,
                             conv_mode = 0, ninit = 1,
                             iter_init = 10, ncores = 1,
                             reorder_factor = FALSE, seed = NULL,
                             a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                             alpha1 = NULL, alpha2 = NULL,
                             beta1 = NULL, beta2 = NULL,
                             prob_D = NULL, prior_D = NULL)
    ## supply U and V
    res <- run_zi_gap_factor(X, K, U = U, V = V,
                             verbose = FALSE, monitor = TRUE,
                             iter_max = 100, iter_min = 50,
                             init_mode = "random",
                             epsilon = 1e-2, additional_iter = 10,
                             conv_mode = 1, ninit = 1,
                             iter_init = 10, ncores = 1,
                             reorder_factor = FALSE, seed = NULL,
                             a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                             alpha1 = NULL, alpha2 = NULL,
                             beta1 = NULL, beta2 = NULL,
                             prob_D = NULL, prior_D = NULL)

    ## mutli-init
    res <- run_zi_gap_factor(X, K, U = NULL, V = NULL,
                             verbose = FALSE, monitor = TRUE,
                             iter_max = 100, iter_min = 50,
                             init_mode = "random",
                             epsilon = 1e-2, additional_iter = 10,
                             conv_mode = 1, ninit = 10,
                             iter_init = 10, ncores = 1,
                             reorder_factor = FALSE, seed = NULL,
                             a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                             alpha1 = NULL, alpha2 = NULL,
                             beta1 = NULL, beta2 = NULL,
                             prob_D = NULL, prior_D = NULL)

    ## multi-init with U and V
    res <- run_zi_gap_factor(X, K, U = U, V = V,
                             verbose = FALSE, monitor = TRUE,
                             iter_max = 100, iter_min = 50,
                             init_mode = "random",
                             epsilon = 1e-2, additional_iter = 10,
                             conv_mode = 1, ninit = 10,
                             iter_init = 10, ncores = 1,
                             reorder_factor = FALSE, seed = NULL,
                             a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                             alpha1 = NULL, alpha2 = NULL,
                             beta1 = NULL, beta2 = NULL,
                             prob_D = NULL, prior_D = NULL)
    ## init prob and prior over D
    prob_D <- matrix(0.5, n, p)
    prior_D <- rep(0.5, p)
    res <- run_zi_gap_factor(X, K, U = NULL, V = NULL,
                             verbose = FALSE, monitor = TRUE,
                             iter_max = 100, iter_min = 50,
                             init_mode = "random",
                             epsilon = 1e-2, additional_iter = 10,
                             conv_mode = 1, ninit = 1,
                             iter_init = 10, ncores = 1,
                             reorder_factor = FALSE, seed = NULL,
                             a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                             alpha1 = NULL, alpha2 = NULL,
                             beta1 = NULL, beta2 = NULL,
                             prob_D = prob_D, prior_D = prior_D)

})


test_that("pCMF with sparse GaP factor model unit tests", {

    n <- 100
    p <- 40
    K <- 5
    U <- matrix(rgamma(n*K, shape=10, rate=1), nrow=n, ncol=K)
    V <- matrix(rgamma(p*K, shape=10, rate=1), nrow=p, ncol=K)

    UVt <- U %*% t(V)

    X <- matrix(rpois(n*p, as.vector(UVt)), nrow=n, ncol=p)

    ## check standard run
    res <- run_sparse_gap_factor(X, K)

    expect_true(class(res) == "pCMF")

    expect_true(is.list(res))
    expect_true(is.list(res$factor))
    expect_true(is.matrix(res$factor$U))
    expect_true(is.matrix(res$factor$V))

    expect_true(is.list(res$variational_params))
    expect_true(is.matrix(res$variational_params$a1))
    expect_true(is.matrix(res$variational_params$a2))
    expect_true(is.matrix(res$variational_params$b1))
    expect_true(is.matrix(res$variational_params$b2))

    expect_true(is.list(res$hyper_params))
    expect_true(is.matrix(res$hyper_params$alpha1))
    expect_true(is.matrix(res$hyper_params$alpha2))
    expect_true(is.matrix(res$hyper_params$beta1))
    expect_true(is.matrix(res$hyper_params$beta2))

    expect_true(is.list(res$stats))
    expect_true(is.matrix(res$stats$EU))
    expect_true(is.matrix(res$stats$EV))
    expect_true(is.matrix(res$stats$ElogU))
    expect_true(is.matrix(res$stats$ElogV))

    expect_true(is.list(res$sparse_param))
    expect_true(is.matrix(res$sparse_param$prob_S))
    expect_true(is.vector(res$sparse_param$prior_prob_S))
    expect_true(is.matrix(res$sparse_param$S))

    expect_true(is.list(res$convergence))
    expect_true(is.logical(res$convergence$converged))
    expect_true(is.integer(res$convergence$nb_iter))
    expect_true(is.integer(res$convergence$conv_mode))
    expect_true(is.vector(res$convergence$conv_crit))

    expect_true(is.numeric(res$loss))
    expect_true(is.numeric(res$exp_dev))

    expect_true(is.vector(res$monitor$abs_gap))
    expect_true(is.vector(res$monitor$norm_gap))
    expect_true(is.vector(res$monitor$loglikelihood))
    expect_true(is.vector(res$monitor$deviance))
    expect_true(is.vector(res$monitor$optim_criterion))

    expect_true(is.null(res$seed))

    ###### check different input parameter combination
    ## full args
    res <- run_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                 verbose = FALSE, monitor = TRUE,
                                 iter_max = 100, iter_min = 50,
                                 init_mode = "random",
                                 epsilon = 1e-2, additional_iter = 10,
                                 conv_mode = 1, ninit = 1,
                                 iter_init = 10, ncores = 1,
                                 reorder_factor = FALSE, seed = NULL,
                                 a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                 alpha1 = NULL, alpha2 = NULL,
                                 beta1 = NULL, beta2 = NULL,
                                 prob_S = NULL, prior_S = NULL)
    ## reorder factor
    res <- run_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                 verbose = FALSE, monitor = TRUE,
                                 iter_max = 100, iter_min = 50,
                                 init_mode = "random",
                                 epsilon = 1e-2, additional_iter = 10,
                                 conv_mode = 1, ninit = 1,
                                 iter_init = 10, ncores = 1,
                                 reorder_factor = TRUE, seed = NULL,
                                 a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                 alpha1 = NULL, alpha2 = NULL,
                                 beta1 = NULL, beta2 = NULL,
                                 prob_S = NULL, prior_S = NULL)
    ## convergence mode
    res <- run_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                 verbose = FALSE, monitor = TRUE,
                                 iter_max = 100, iter_min = 50,
                                 init_mode = "random",
                                 epsilon = 1e-2, additional_iter = 10,
                                 conv_mode = 0, ninit = 1,
                                 iter_init = 10, ncores = 1,
                                 reorder_factor = FALSE, seed = NULL,
                                 a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                 alpha1 = NULL, alpha2 = NULL,
                                 beta1 = NULL, beta2 = NULL,
                                 prob_S = NULL, prior_S = NULL)
    ## supply U and V
    res <- run_sparse_gap_factor(X, K, U = U, V = V,
                                 verbose = FALSE, monitor = TRUE,
                                 iter_max = 100, iter_min = 50,
                                 init_mode = "random",
                                 epsilon = 1e-2, additional_iter = 10,
                                 conv_mode = 1, ninit = 1,
                                 iter_init = 10, ncores = 1,
                                 reorder_factor = FALSE, seed = NULL,
                                 a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                 alpha1 = NULL, alpha2 = NULL,
                                 beta1 = NULL, beta2 = NULL,
                                 prob_S = NULL, prior_S = NULL)

    ## mutli-init
    res <- run_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                 verbose = FALSE, monitor = TRUE,
                                 iter_max = 100, iter_min = 50,
                                 init_mode = "random",
                                 epsilon = 1e-2, additional_iter = 10,
                                 conv_mode = 1, ninit = 10,
                                 iter_init = 10, ncores = 1,
                                 reorder_factor = FALSE, seed = NULL,
                                 a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                 alpha1 = NULL, alpha2 = NULL,
                                 beta1 = NULL, beta2 = NULL,
                                 prob_S = NULL, prior_S = NULL)

    ## multi-init with U and V
    res <- run_sparse_gap_factor(X, K, U = U, V = V,
                                 verbose = FALSE, monitor = TRUE,
                                 iter_max = 100, iter_min = 50,
                                 init_mode = "random",
                                 epsilon = 1e-2, additional_iter = 10,
                                 conv_mode = 1, ninit = 10,
                                 iter_init = 10, ncores = 1,
                                 reorder_factor = FALSE, seed = NULL,
                                 a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                 alpha1 = NULL, alpha2 = NULL,
                                 beta1 = NULL, beta2 = NULL,
                                 prob_S = NULL, prior_S = NULL)
    ## init prob and prior over S
    prob_S <- matrix(0.5, p, K)
    prior_S <- rep(0.5, p)
    res <- run_zi_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                    verbose = FALSE, monitor = TRUE,
                                    iter_max = 100, iter_min = 50,
                                    init_mode = "random",
                                    epsilon = 1e-2, additional_iter = 10,
                                    conv_mode = 1, ninit = 1,
                                    iter_init = 10, ncores = 1,
                                    reorder_factor = FALSE, seed = NULL,
                                    a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                    alpha1 = NULL, alpha2 = NULL,
                                    beta1 = NULL, beta2 = NULL,
                                    prob_S = prob_S, prior_S = prior_S)

})

test_that("pCMF with ZI sparse GaP factor model unit tests", {

    n <- 100
    p <- 40
    K <- 5
    U <- matrix(rgamma(n*K, shape=10, rate=1), nrow=n, ncol=K)
    V <- matrix(rgamma(p*K, shape=10, rate=1), nrow=p, ncol=K)

    UVt <- U %*% t(V)

    X <- matrix(rpois(n*p, as.vector(UVt)), nrow=n, ncol=p)

    D <- matrix(rbinom(n*p, size=1, prob=0.5), nrow=n, ncol=p)

    X <- X * D

    ## check standard run
    res <- run_zi_sparse_gap_factor(X, K)

    expect_true(class(res) == "pCMF")

    expect_true(is.list(res))
    expect_true(is.list(res$factor))
    expect_true(is.matrix(res$factor$U))
    expect_true(is.matrix(res$factor$V))

    expect_true(is.list(res$variational_params))
    expect_true(is.matrix(res$variational_params$a1))
    expect_true(is.matrix(res$variational_params$a2))
    expect_true(is.matrix(res$variational_params$b1))
    expect_true(is.matrix(res$variational_params$b2))

    expect_true(is.list(res$hyper_params))
    expect_true(is.matrix(res$hyper_params$alpha1))
    expect_true(is.matrix(res$hyper_params$alpha2))
    expect_true(is.matrix(res$hyper_params$beta1))
    expect_true(is.matrix(res$hyper_params$beta2))

    expect_true(is.list(res$stats))
    expect_true(is.matrix(res$stats$EU))
    expect_true(is.matrix(res$stats$EV))
    expect_true(is.matrix(res$stats$ElogU))
    expect_true(is.matrix(res$stats$ElogV))

    expect_true(is.list(res$ZI_param))
    expect_true(is.matrix(res$ZI_param$prob_D))
    expect_true(is.vector(res$ZI_param$freq_D))
    expect_true(is.vector(res$ZI_param$prior_prob_D))

    expect_true(is.list(res$sparse_param))
    expect_true(is.matrix(res$sparse_param$prob_S))
    expect_true(is.vector(res$sparse_param$prior_prob_S))
    expect_true(is.matrix(res$sparse_param$S))

    expect_true(is.list(res$convergence))
    expect_true(is.logical(res$convergence$converged))
    expect_true(is.integer(res$convergence$nb_iter))
    expect_true(is.integer(res$convergence$conv_mode))
    expect_true(is.vector(res$convergence$conv_crit))

    expect_true(is.numeric(res$loss))
    expect_true(is.numeric(res$exp_dev))

    expect_true(is.vector(res$monitor$abs_gap))
    expect_true(is.vector(res$monitor$norm_gap))
    expect_true(is.vector(res$monitor$loglikelihood))
    expect_true(is.vector(res$monitor$deviance))
    expect_true(is.vector(res$monitor$optim_criterion))

    expect_true(is.null(res$seed))

    ###### check different input parameter combination
    ## full args
    res <- run_zi_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                    verbose = FALSE, monitor = TRUE,
                                    iter_max = 100, iter_min = 50,
                                    init_mode = "random",
                                    epsilon = 1e-2, additional_iter = 10,
                                    conv_mode = 1, ninit = 1,
                                    iter_init = 10, ncores = 1,
                                    reorder_factor = FALSE, seed = NULL,
                                    a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                    alpha1 = NULL, alpha2 = NULL,
                                    beta1 = NULL, beta2 = NULL,
                                    prob_S = NULL, prior_S = NULL,
                                    prob_D = NULL, prior_D = NULL)
    ## reorder factor
    res <- run_zi_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                    verbose = FALSE, monitor = TRUE,
                                    iter_max = 100, iter_min = 50,
                                    init_mode = "random",
                                    epsilon = 1e-2, additional_iter = 10,
                                    conv_mode = 1, ninit = 1,
                                    iter_init = 10, ncores = 1,
                                    reorder_factor = TRUE, seed = NULL,
                                    a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                    alpha1 = NULL, alpha2 = NULL,
                                    beta1 = NULL, beta2 = NULL,
                                    prob_S = NULL, prior_S = NULL,
                                    prob_D = NULL, prior_D = NULL)
    ## convergence mode
    res <- run_zi_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                    verbose = FALSE, monitor = TRUE,
                                    iter_max = 100, iter_min = 50,
                                    init_mode = "random",
                                    epsilon = 1e-2, additional_iter = 10,
                                    conv_mode = 0, ninit = 1,
                                    iter_init = 10, ncores = 1,
                                    reorder_factor = FALSE, seed = NULL,
                                    a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                    alpha1 = NULL, alpha2 = NULL,
                                    beta1 = NULL, beta2 = NULL,
                                    prob_S = NULL, prior_S = NULL,
                                    prob_D = NULL, prior_D = NULL)
    ## supply U and V
    res <- run_zi_sparse_gap_factor(X, K, U = U, V = V,
                                    verbose = FALSE, monitor = TRUE,
                                    iter_max = 100, iter_min = 50,
                                    init_mode = "random",
                                    epsilon = 1e-2, additional_iter = 10,
                                    conv_mode = 1, ninit = 1,
                                    iter_init = 10, ncores = 1,
                                    reorder_factor = FALSE, seed = NULL,
                                    a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                    alpha1 = NULL, alpha2 = NULL,
                                    beta1 = NULL, beta2 = NULL,
                                    prob_S = NULL, prior_S = NULL,
                                    prob_D = NULL, prior_D = NULL)

    ## mutli-init
    res <- run_zi_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                    verbose = FALSE, monitor = TRUE,
                                    iter_max = 100, iter_min = 50,
                                    init_mode = "random",
                                    epsilon = 1e-2, additional_iter = 10,
                                    conv_mode = 1, ninit = 10,
                                    iter_init = 10, ncores = 1,
                                    reorder_factor = FALSE, seed = NULL,
                                    a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                    alpha1 = NULL, alpha2 = NULL,
                                    beta1 = NULL, beta2 = NULL,
                                    prob_S = NULL, prior_S = NULL,
                                    prob_D = NULL, prior_D = NULL)

    ## multi-init with U and V
    res <- run_zi_sparse_gap_factor(X, K, U = U, V = V,
                                    verbose = FALSE, monitor = TRUE,
                                    iter_max = 100, iter_min = 50,
                                    init_mode = "random",
                                    epsilon = 1e-2, additional_iter = 10,
                                    conv_mode = 1, ninit = 10,
                                    iter_init = 10, ncores = 1,
                                    reorder_factor = FALSE, seed = NULL,
                                    a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                    alpha1 = NULL, alpha2 = NULL,
                                    beta1 = NULL, beta2 = NULL,
                                    prob_S = NULL, prior_S = NULL,
                                    prob_D = NULL, prior_D = NULL)
    ## init prob and prior over S
    prob_S <- matrix(0.5, p, K)
    prior_S <- rep(0.5, p)
    res <- run_zi_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                    verbose = FALSE, monitor = TRUE,
                                    iter_max = 100, iter_min = 50,
                                    init_mode = "random",
                                    epsilon = 1e-2, additional_iter = 10,
                                    conv_mode = 1, ninit = 1,
                                    iter_init = 10, ncores = 1,
                                    reorder_factor = FALSE, seed = NULL,
                                    a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                    alpha1 = NULL, alpha2 = NULL,
                                    beta1 = NULL, beta2 = NULL,
                                    prob_S = prob_S, prior_S = prior_S,
                                    prob_D = NULL, prior_D = NULL)
    ## init prob and prior over D
    prob_D <- matrix(0.5, n, p)
    prior_D <- rep(0.5, p)
    res <- run_zi_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                    verbose = FALSE, monitor = TRUE,
                                    iter_max = 100, iter_min = 50,
                                    init_mode = "random",
                                    epsilon = 1e-2, additional_iter = 10,
                                    conv_mode = 1, ninit = 1,
                                    iter_init = 10, ncores = 1,
                                    reorder_factor = FALSE, seed = NULL,
                                    a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                    alpha1 = NULL, alpha2 = NULL,
                                    beta1 = NULL, beta2 = NULL,
                                    prob_S = NULL, prior_S = NULL,
                                    prob_D = prob_D, prior_D = prior_D)
    ## init prob and prior over both D  and S
    prob_D <- matrix(0.5, n, p)
    prior_D <- rep(0.5, p)
    prob_S <- matrix(0.5, p, K)
    prior_S <- rep(0.5, p)
    res <- run_zi_sparse_gap_factor(X, K, U = NULL, V = NULL,
                                    verbose = FALSE, monitor = TRUE,
                                    iter_max = 100, iter_min = 50,
                                    init_mode = "random",
                                    epsilon = 1e-2, additional_iter = 10,
                                    conv_mode = 1, ninit = 1,
                                    iter_init = 10, ncores = 1,
                                    reorder_factor = FALSE, seed = NULL,
                                    a1 = NULL, a2 = NULL, b1 = NULL, b2 = NULL,
                                    alpha1 = NULL, alpha2 = NULL,
                                    beta1 = NULL, beta2 = NULL,
                                    prob_S = prob_S, prior_S = prior_S,
                                    prob_D = prob_D, prior_D = prior_D)

})

test_that("pCMF wrapper with standard GaP factor model unit tests", {

    n <- 100
    p <- 40
    K <- 5
    U <- matrix(rgamma(n*K, shape=10, rate=1), nrow=n, ncol=K)
    V <- matrix(rgamma(p*K, shape=10, rate=1), nrow=p, ncol=K)

    UVt <- U %*% t(V)

    X <- matrix(rpois(n*p, as.vector(UVt)), nrow=n, ncol=p)

    ## check standard run
    res <- pCMF(X, K, zero_inflation = FALSE, sparsity = FALSE)

})

test_that("pCMF wrapper with standard ZI GaP factor model unit tests", {

    n <- 100
    p <- 40
    K <- 5
    U <- matrix(rgamma(n*K, shape=10, rate=1), nrow=n, ncol=K)
    V <- matrix(rgamma(p*K, shape=10, rate=1), nrow=p, ncol=K)

    UVt <- U %*% t(V)

    X <- matrix(rpois(n*p, as.vector(UVt)), nrow=n, ncol=p)

    D <- matrix(rbinom(n*p, size=1, prob=0.5), nrow=n, ncol=p)

    X <- X * D

    ## check standard run
    res <- pCMF(X, K, zero_inflation = TRUE, sparsity = FALSE)

})

test_that("pCMF wrapper with sparse GaP factor model unit tests", {

    n <- 100
    p <- 40
    K <- 5
    U <- matrix(rgamma(n*K, shape=10, rate=1), nrow=n, ncol=K)
    V <- matrix(rgamma(p*K, shape=10, rate=1), nrow=p, ncol=K)

    UVt <- U %*% t(V)

    X <- matrix(rpois(n*p, as.vector(UVt)), nrow=n, ncol=p)

    ## check standard run
    res <- pCMF(X, K, zero_inflation = FALSE, sparsity = TRUE)

})

test_that("pCMF wrapper with ZI sparse GaP factor model unit tests", {

    n <- 100
    p <- 40
    K <- 5
    U <- matrix(rgamma(n*K, shape=10, rate=1), nrow=n, ncol=K)
    V <- matrix(rgamma(p*K, shape=10, rate=1), nrow=p, ncol=K)

    UVt <- U %*% t(V)

    X <- matrix(rpois(n*p, as.vector(UVt)), nrow=n, ncol=p)

    D <- matrix(rbinom(n*p, size=1, prob=0.5), nrow=n, ncol=p)

    X <- X * D

    ## check standard run
    res <- pCMF(X, K, zero_inflation = TRUE, sparsity = TRUE)

})
