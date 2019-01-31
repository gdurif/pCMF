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
* \brief functions related to probability distribution and metric
* \author Ghislain Durif
* \version 1.0
* \date 22/02/2018
*/

#ifndef PROBABILITY_H
#define PROBABILITY_H

#include <boost/math/special_functions/digamma.hpp>
#include <math.h>
#include <RcppEigen.h>

#include "macros.h"

// [[Rcpp::depends(BH)]]
using boost::math::digamma;

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
// using Eigen::Map;                       // 'maps' rather than copies
// using Eigen::VectorXd;                  // variable size vector, double precision

/*!
* \namespace probability
*
* A specific namespace for all related probability functions
*/
namespace probability {

/*!
* \fn Compute expection of Gamma distribution
*
* E[U] = alpha_1/alpha_2
* where alpha_1=param1 (shape) and alpha_2=param2 (rate)
*
* \param[in] param1 shape parameter
* \param[in] param2 rate parameter
*
* \return computed value
*/
inline double Egamma(double param1, double param2) {
    return(param1 / param2);
};

/*!
 * \fn Compute expection of Gamma distribution matrix element-wise
 *
 * E[U] = alpha_1/alpha_2
 * where alpha_1=param1 (shape) and alpha_2=param2 (rate)
 *
 * \param[in] param1 matrix rows x cols of shape parameters
 * \param[in] param2 matrix rows x cols of rate parameter
 * \param[out] res matrix rows x cols of expectation for each couple
 * of parameters
 */
inline void Egamma(const MatrixXd &param1, const MatrixXd &param2, MatrixXd &res) {
    res = param1.array() / param2.array();
};

/*!
* \fn Compute expection of log of Gamma distribution
*
* E[log U] = digamma(alpha_1) - log(alpha_2)
* where alpha_1=param1 (shape) and alpha_2=param2 (rate)
*
* \param[in] param1 shape parameter
* \param[in] param2 rate parameter
*
* \return computed value
*/
inline double ElogGamma(double param1, double param2) {
    return(digamma(param1) - log(param2));
};

/*!
 * \fn Compute expection of log of Gamma distribution matrix element-wise
 *
 * E[log U] = digamma(alpha_1) - log(alpha_2)
 * where alpha_1=param1 (shape) and alpha_2=param2 (rate)
 *
 * \param[in] param1 matrix rows x cols of shape parameters
 * \param[in] param2 matrix rows x cols of rate parameter
 * \param[out] res matrix rows x cols of log expectation for each couple
 * of parameters
 */
inline void ElogGamma(const MatrixXd &param1, const MatrixXd &param2, MatrixXd &res) {
    res = param1.mdigamma().array() - param2.mlog().array();
};

/*!
 * \fn Compute entropy of Gamma distribution
 *
 * E[-log p(U)] = (1-alpha)*digamma(alpha) + alpha - log(beta)
 *               + log(gamma(alpha))
 * where alpha_1=param1 (shape) and alpha_2=param2 (rate)
 *
 * \param[in] param1 shape parameter
 * \param[in] param2 rate parameter
 *
 * \return computed value
 */
inline double gamma_entropy(double param1, double param2) {
    return((1-param1) * digamma(param1) + param1
            + lgamma(param1) - std::log(param2));
};

/*!
 * \fn Compute entropy of Gamma distribution
 *
 * E[-log p(U)] = (1-alpha)*digamma(alpha) + alpha - log(beta)
 *               + log(gamma(alpha))
 * where alpha_1=param1 (shape) and alpha_2=param2 (rate)
 *
 * \param[in] param1 matrix rows x cols of shape parameters
 * \param[in] param2 matrix rows x cols of rate parameter
 * \param[out] res matrix rows x cols of entropy for each couple
 * of parameters
 */
inline void gamma_entropy(const MatrixXd &param1, const MatrixXd &param2, MatrixXd &res) {
    res = (1-param1.array()) * param1.mdigamma().array() + param1.array()
            + param1.mlgamma().array() - param2.mlog().array();
};

/*!
 * \fn Compute the sum of the entropy of Gamma distributions
 *
 * E[-log p(U)] = (1-alpha)*digamma(alpha) + alpha - log(beta)
 *               + log(gamma(alpha))
 * where alpha_1=param1 (shape) and alpha_2=param2 (rate)
 *
 * \param[in] param1 matrix rows x cols of shape parameters
 * \param[in] param2 matrix rows x cols of rate parameter
 *
 * \returns the value corresponding to the sum of all Gamma's entropy over
 * the input parameter matrices
 */
inline double gamma_entropy(const MatrixXd &param1, const MatrixXd &param2) {
    return( ((1-param1.array()) * param1.mdigamma().array() + param1.array()
            + param1.mlgamma().array() - param2.mlog().array()).sum());
};

/*!
 * \fn compute the Bregman divergence associated to the Poisson distribution
 *
 * \f[
 * d(x,y) = x \log\frac{x}{y} - x + y
 * \f]
 *
 * \param[in] x positive value
 * \param[in] y positive value
 *
 * \return value of the Bregman divergence in the Poisson setting
 */
inline double bregman_poisson(double x, double y) {
    return( x * log((x>0 ? x : 1)/y) - x + y );
};

// /*!
//  * \fn compute the element-wise Bregman divergence associated to the Poisson distribution between two matrices (of same dimension)
//  *
//  * \f[
//  * d(X,Y) = \sum_{i,j} d(x_{i,j}, y_{i,j})
//  * \f]
//  *
//  * with \f$ d(x,y) = x \log\frac{x}{y} - x + y \f$
//  *
//  * \param[in] X real matrix
//  * \param[in] Y real matrix
//  * \param[in] nrow number of rows in X and Y
//  * \param[in] ncol number of cols in X and Y
//  *
//  * \return value of the Bregman divergence in the Poisson setting
//  */
// inline double mat_bregman_poisson(MatrixXd X, MatrixXd Y, int nrow, int ncol) {
//     return( bregman_div(X, Y, nrow, ncol, &probability::bregman_poisson) );
// };
//
// /*!
//  * \fn compute the element-wise Bregman divergence between two matrices (of same dimensions)
//  *
//  * \f[
//  * d(X,Y) = \sum_{i,j} d(x_{i,j}, y_{i,j})
//  * \f]
//  *
//  * where \f$ (x,y) \mapsto d(x,y) \f$ is a bregman divergence on real values
//  *
//  * \param[in] X real matrix
//  * \param[in] Y real matrix
//  * \param[in] nrow number of rows in X and Y
//  * \param[in] ncol number of cols in X and Y
//  * \param[in] div a bregman divergence on real values
//  *
//  * \return value of the Bregman divergence between the two matrix entries
//  */
// inline double bregman_div(MatrixXd X, MatrixXd Y, int nrow, int ncol, double (*div)(double,double)) {
//
//     // parallel loop over i if nrow>ncol and over j if ncol>nrow
//     int outer_max = nrow >= ncol ? nrow : ncol;
//     int inner_max = nrow >= ncol ? ncol : nrow;
//     int outer_index, inner_index;
//
//     VectorXd res = VectorXd::Zero(outer_max);
//
// #if defined(_OPENMP)
// #pragma omp parallel for private(inner_index)
// #endif
//     for(outer_index=0; outer_index<outer_max; outer_index++) {
//         if(nrow >= ncol) {
//             for(inner_index=0; inner_index<inner_max; inner_index++) {
//                 res(outer_index) += div(X(outer_index,inner_index), Y(outer_index,inner_index));
//             }
//         } else {
//             for(inner_index=0; inner_index<inner_max; inner_index++) {
//                 res(outer_index) += div(X(inner_index,outer_index), Y(inner_index,outer_index));
//             }
//         }
//
//     }
//
//     return(res.sum());
// };


}

#endif // PROBABILITY_H
