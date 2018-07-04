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

/*!
 * \brief functions for likelihood computation
 * \author Ghislain Durif
 * \version 1.0
 * \date 06/02/2018
 */

#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H

#include <boost/math/special_functions/digamma.hpp>
#include <math.h>
#include <RcppEigen.h>

#include "internal.h"

#define mclog() unaryExpr(std::ptr_fun<double,double>(internal::custom_log))
#define mlgamma() unaryExpr(std::ptr_fun<double,double>(lgamma))
#define mlog() unaryExpr(std::ptr_fun<double,double>(std::log))

// [[Rcpp::depends(BH)]]
using boost::math::digamma;

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::RowVectorXd;                  // variable size row vector, double precision
using Eigen::VectorXd;                  // variable size vector, double precision

/*!
 * \namespace likelihood
 *
 * A specific namespace for all related likelihood functions
 */
namespace likelihood {

/*!
 * \fn log-likelihood for the Gamma distribution Gamma(param1, param2)
 *
 * `param1` is the shape parameter and `param2` is the rate parameter.
 * The density function is defined as
 * \f[
 * x \mapsto f(x; a_1, a_2) = \frac{(a_2)^{a_1} x^{a_1-1} e^{-a_2 x}}{\Gamma(a_1)}
 * \f]
 * where f\$ z \mapsto \Gamma(z) \f$ is the Gamma function.
 *
 * \param[in] x observed value
 * \param[in] param1 shape parameter for the Gamma distribution
 * \param[in] param2 rate parameter for the Gamma distribution
 *
 * \return Gamma log-likelihood value
 */
inline double gamma_loglike(double x, double param1, double param2) {
    double res=0;
    res = param1 * log(param2) + (param1 - 1) * log(x) - param2 * x - lgamma(param1);
    return res;
};

/*!
 * \fn log-likelihood for the Gamma distribution Gamma(param1, param2) matrix-wise
 *
 * `param1` is the shape parameter and `param2` is the rate parameter.
 * The density function is defined as
 * \f[
 * x \mapsto f(x; a_1, a_2) = \frac{(a_2)^{a_1} x^{a_1-1} e^{-a_2 x}}{\Gamma(a_1)}
 * \f]
 * where f\$ z \mapsto \Gamma(z) \f$ is the Gamma function.
 *
 * \param[in] x matrix rows x cols of observed values
 * \param[in] param1 matrix rows x cols of shape parameters
 * \param[in] param2 matrix rows x cols of rate parameters
 *
 * \return sum of the Gamma log-likelihood values over the rows x cols matrix
 */
inline double gamma_loglike(const MatrixXd &X, const MatrixXd &param1, const MatrixXd &param2) {
    double res = ( param1.array() * param2.mlog().array() + (param1.array() - 1) * X.mlog().array()
                       - param2.array() * X.array() - param1.mlgamma().array() ).sum();
    return res;
};

/*!
 * \fn log-likelihood for the Gamma distribution Gamma(param1, param2) matrix-wise
 * where X and logX are given
 *
 * `param1` is the shape parameter and `param2` is the rate parameter.
 * The density function is defined as
 * \f[
 * x \mapsto f(x; a_1, a_2) = \frac{(a_2)^{a_1} x^{a_1-1} e^{-a_2 x}}{\Gamma(a_1)}
 * \f]
 * where f\$ z \mapsto \Gamma(z) \f$ is the Gamma function.
 *
 * \param[in] x matrix rows x cols of observed values
 * \param[in] param1 matrix rows x cols of shape parameters
 * \param[in] param2 matrix rows x cols of rate parameters
 *
 * \return sum of the Gamma log-likelihood values over the rows x cols matrix
 */
inline double gamma_loglike(const MatrixXd &X, const MatrixXd &logX,
                            const MatrixXd &param1, const MatrixXd &param2) {
    double res = ( param1.array() * param2.mlog().array() + (param1.array() - 1) * logX.array()
                       - param2.array() * X.array() - param1.mlgamma().array() ).sum();
    return res;
};

/*!
 * \fn log-likelihood for the Poisson distribution P(rate)
 *
 * `rate` is the Poisson intensity parameter.
 * The density function is defined as
 * \f[
 * x \mapsto f(x; r) = \frac{r^x e^{-r}}{r!}
 * \f]
 *
 * \param[in] x observed value
 * \param[in] rate Poisson intensity
 *
 * \return Poisson log-likelihood value
 */
inline double poisson_loglike(double x, double rate) {
    double res = 0;
    res = x * log(rate>0 ? rate : 1) - rate - lgamma(x+1);
    return res;
};

/*!
 * \fn log-likelihood for the Poisson distribution P(rate) matrix-wise
 *
 * `rate` is the Poisson intensity parameter.
 * The density function is defined as
 * \f[
 * x \mapsto f(x; r) = \frac{r^x e^{-r}}{r!}
 * \f]
 *
 * \param[in] x matrix rows x cols of observed values
 * \param[in] rate matrix rows x cols of Poisson intensities
 *
 * \return sum of the Poisson log-likelihood values over the rows x cols matrix
 */
inline double poisson_loglike(const MatrixXd &X, const MatrixXd &rate) {
    // double res = ( X.array() * ((rate.array() > 0).select(rate, 1)).mlog().array()
    //                 - rate.array() - (X.array() + 1).mlgamma() ).sum();
    double res = ( X.array() * rate.mclog().array()
                       - rate.array() - (X.array() + 1).mlgamma() ).sum();
    return res;
};

/*!
 * \fn log-likelihood for the Poisson distribution P(rate) matrix-wise
 * with a column-wise rate
 *
 * `rate` is the Poisson intensity parameter.
 * The density function is defined as
 * \f[
 * x \mapsto f(x; r) = \frac{r^x e^{-r}}{r!}
 * \f]
 *
 * \param[in] x matrix rows x cols of observed values
 * \param[in] rate vector of size cols of Poisson intensities
 *
 * \return sum of the Poisson log-likelihood values over the rows x cols matrix
 */
inline double poisson_loglike_vec(const MatrixXd &X, const RowVectorXd &rate) {
    // double res = ( ( X.array().rowwise()
    //                      * ((rate.array() > 0).select(rate, 1)).mlog().array() ).rowwise() - rate.array() ).sum()
    //                 - (X.array() + 1).mlgamma().sum();
    double res = ( ( X.array().rowwise()
                            * rate.mclog().array() ).rowwise() - rate.array() ).sum()
                    - (X.array() + 1).mlgamma().sum();
    return res;
};

/*!
 * \fn log-likelihood for the Zero-inflated Poisson distribution matrix-wise
 *
 * `rate` is the Poisson intensity parameter and `prob` the
 * drop-out probability. The density function is defined as
 * \f[
 * x \mapsto f(x; r, prob) = (1-prob) delta_0(x) + prob \frac{r^x e^{-r}}{r!}
 * \f]
 *
 * \param[in] x matrix rows x cols of observed values
 * \param[in] rate matrix rows x cols of Poisson intensities
 * \param[in] prob matrix rows x cols of drop-out probabilities
 *
 * \return sum of the ZI Poisson log-likelihood values over the rows x cols matrix
 */
inline double zi_poisson_loglike(const MatrixXd &X, const MatrixXd &rate,
                                 const MatrixXd &prob) {
    int i, j;
    VectorXd tmp = VectorXd::Zero(X.cols());
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
    for(j=0; j<X.cols(); j++) {
        for(i=0; i<X.rows(); i++) {
            tmp(j) += X(i,j) > 0 ?
                        std::log(prob(i,j)) + poisson_loglike(X(i,j), rate(i,j)) :
                        std::log( (1-prob(i,j) + prob(i,j) * std::exp(-rate(i,j))));
        }
    }
    return tmp.sum();
};

/*!
 * \fn log-likelihood for the Zero-inflated Poisson distribution matrix-wise
 * with column-wise parameters
 *
 * `rate` is the Poisson intensity parameter and `prob` the
 * drop-out probability. The density function is defined as
 * \f[
 * x \mapsto f(x; r, prob) = (1-prob) delta_0(x) + prob \frac{r^x e^{-r}}{r!}
 * \f]
 *
 * \param[in] x matrix rows x cols of observed values
 * \param[in] rate vector of size cols of Poisson intensities
 * \param[in] prob vector of size cols of drop-out probabilities
 *
 * \return sum of the Poisson log-likelihood values over the rows x cols matrix
 */
inline double zi_poisson_loglike_vec(const MatrixXd &X, const VectorXd &rate,
                                     const VectorXd &prob) {
    int i, j;
    VectorXd tmp = VectorXd::Zero(X.cols());
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
    for(j=0; j<X.cols(); j++) {
        for(i=0; i<X.rows(); i++) {
            tmp(j) += X(i,j) > 0 ?
                        std::log(prob(j)) + poisson_loglike(X(i,j), rate(j)) :
                        std::log( (1-prob(j) + prob(j) * std::exp(-rate(j))));
        }
    }
    return tmp.sum();
};

/*!
 * \fn log-likelihood for the Bernoulli distribution P(prob) matrix-wise
 *
 * `prob` is the Bernoulli parameter.
 * The density function is defined as
 * \f[
 * x \mapsto f(x; prob) = prob^x * (1-prob)^{(1 - x)}
 * \f]
 *
 * \param[in] x matrix rows x cols of observed values
 * \param[in] prob matrix rows x cols of Bernoulli probabilities
 *
 * \return sum of the Bernnoulli log-likelihood values over the rows x cols matrix
 */
inline double bernoulli_loglike(const MatrixXd &X, const MatrixXd &prob) {
    // double res = ( X.array() * prob.mclog().array()
    //                    + (1 - X.array()).array() * (1 - prob.array()).mclog().array() ).sum();
    // return res;
    int i, j;
    VectorXd tmp = VectorXd::Zero(X.cols());
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
    for(j=0; j<X.cols(); j++) {
        for(i=0; i<X.rows(); i++) {
            // if(prob(i,j)>1e-44 && prob(i,j)<1-1e-44) {
            if(prob(i,j)>0 && prob(i,j)<1) {
                tmp(j) += X(i,j) * std::log(prob(i,j)) + (1-X(i,j)) * std::log(1-prob(i,j));
            }
        }
    }
    return tmp.sum();
};

/*!
 * \fn log-likelihood for the Bernoulli distribution P(prob) matrix-wise
 * with column-wise parameters
 *
 * `prob` is the Bernoulli parameter.
 * The density function is defined as
 * \f[
 * x \mapsto f(x; prob) = prob^x * (1-prob)^{(1 - x)}
 * \f]
 *
 * \param[in] x matrix rows x cols of observed values
 * \param[in] prob prob vector of size cols of Bernoulli probabilities
 *
 * \return sum of the Bernnoulli log-likelihood values over the rows x cols matrix
 */
inline double bernoulli_loglike_vec(const MatrixXd &X, const RowVectorXd &prob) {
    // double res = ( X.array().rowwise() * prob.mclog().array()
    //                    + (1 - X.array()).array().rowwise() * (1 - prob.array()).mclog().array() ).sum();
    // return res;
    int i, j;
    VectorXd tmp = VectorXd::Zero(X.cols());
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
    for(j=0; j<X.cols(); j++) {
        for(i=0; i<X.rows(); i++) {
            if(prob(i,j)>0 && prob(i,j)<1) {
                tmp(j) += X(i,j) * std::log(prob(j)) + (1-X(i,j)) * std::log(1-prob(j));
            }
        }
    }
    return tmp.sum();
};

}

#endif // LIKELIHOOD_H
