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
 * \brief definitions of internal functions
 * \author Ghislain Durif
 * \version 1.0
 * \date 07/02/2018
 */

#ifndef INTERNAL_H
#define INTERNAL_H

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <math.h>
#include <RcppEigen.h>

// [[Rcpp::depends(BH)]]
using boost::math::digamma;
using boost::math::trigamma;

// [[Rcpp::depends(RcppEigen)]]
using Eigen::VectorXd;                  // variable size vector, double precision

/*!
 * \namespace internal
 *
 * A specific namespace for internal functions
 */
namespace internal {

/*!
 * \fn inverse digamma (psi) function
 *
 * find the solution regarding x with y given of y = digamma(x)
 *
 * \param[in] y the value of the digamma function
 * @return the corresponding x
 */
inline double digammaInv(double y, int nbIter) {
    double x0 = 0;
    double x = 0;

    // init
    if(y >= -2.22) {
        x0 = std::exp(y) + 0.5;
    } else {
        x0 = -1/(y-digamma(1));
    }

    // iter
    for(int i=0; i<nbIter; i++) {
        x = x0 - (digamma(x0) - y)/trigamma(x0);
        x0 = x;
    }

    return x;
};

/*!
 * \fn compute the empirical variance of a serie of observations
 *
 * \param[in] sample vector of observations
 *
 * \return value of the (unbiased) variance estimator on the sample of observations
 */
inline double variance(const VectorXd &sample) {
    int n = sample.size();
    double var;
    VectorXd centered(n);
    centered = sample.array() - sample.mean();
    var = centered.squaredNorm() / double(n - 1);
    return(var);
};

/*!
 * \fn custom logarithm
 *
 * return 0 if applied to 0
 *
 * It is designed to be used in Poisson log-likelihood where we can have
 * 0*log(0) (which should be 0)
 *
 * \param[in] x real positive value
 *
 * \return log(x) if x>0 else 0
 */
inline double custom_log(double x) {
    if(x > 0) {
        return(std::log(x));
    } else {
        return(0);
    }
};

/*!
 * \fn custom logit function to avoid under and over-flow
 *
 * log(x/(1-x))
 *
 * @param[in] x real between 0 and 1
 * @return the value of logit(x)
 */
inline double logit(double x) {
    if (x >= 1 - 1e-12) return 30;
    if (x <= 1e-12) return -30;
    return std::log(x/(1-x));
};

/*!
 * \fn custom logit inverse function to avoid under and over-flow
 *
 * exp(x)/(1+exp(x)) = 1/(1+exp(-x))
 *
 * @param[in] x real between 0 and 1
 * @return the value of logit(x)
 */
inline double expit(double x) {
    if (x >= 30) return 1;
    if (x <= -30) return 0;
    return 1.0 / (1 + std::exp(-x));
};

}

#endif // INTERNAL_H
