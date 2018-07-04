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
* \brief functions related to matrix computations
* \author Ghislain Durif
* \version 1.0
* \date 23/02/2018
*/

#ifndef MATRIX_H
#define MATRIX_H

#include <math.h>
#include <RcppEigen.h>

// [[rcpp::depends(RcppEigen)]]
using Eigen::Map;                       // 'maps' rather than copies
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::VectorXd;                  // variable size vector, double precision

/*!
* \namespace matrix
*
* A specific namespace for all related matrix functions
*/
namespace matrix {

/*!
* \fn compute the sum of results from a real-valued bivariate function ()taking two real-valued arguments)
*
* Real valued arguments are given by two matrices (of same dimensions),
* and the function is applied element-wise:
*
* \f[
* \sum_{i,j} f(x_{i,j}, y_{i,j})
* \f]
*
* where \f$ (x,y) \mapsto f(x,y) \f$ is a real-valued function taking
* two real-valued arguments
*
* \param[in] X real matrix
* \param[in] Y real matrix
* \param[in] nrow number of rows in X and Y
* \param[in] ncol number of cols in X and Y
* \param[in] fun a real-valued function taking two real-valued arguments
*
* \return value of the sum of the bivariate function applied element wise
*/
inline double elem_wise_bivar_func_sum(const MatrixXd &X, const MatrixXd &Y,
                                       int nrow, int ncol,
                                       double (*fun)(double,double)) {

    int outer_index, inner_index;
    VectorXd res = VectorXd::Zero(ncol);

    // column major storage for matrix in Eigen
#if defined(_OPENMP)
#pragma omp parallel for private(inner_index)
#endif
    for(outer_index=0; outer_index<ncol; outer_index++) {
        for(inner_index=0; inner_index<nrow; inner_index++) {
            res(outer_index) += fun(X(inner_index,outer_index), Y(inner_index,outer_index));
        }
    }
    return(res.sum());
};

/*!
 * \fn compute thr RV coefficients between two matrices (of same shape)
 *
 * The RV coefficients measures the closeness of the two set of points stored
 * in two matrices [1]
 *
 * \param[in] A first matrix
 * \param[in] B second matrix
 *
 * Note: A and B should have the same dimension.
 *
 * \return value of the RV coefficients
 *
 * [1] Friguet, C., 2010. Impact de la dépendance dans les procédures de tests
 * multiples en grande dimension. Rennes, AGROCAMPUS-OUEST.
 */
inline double rv_coeff(const MatrixXd &A, const MatrixXd &B) {
    double res = ( (A * A.transpose() * B * B.transpose()).trace())
    / sqrt( (A * A.transpose() * A * A.transpose()).trace() * (B * B.transpose() * B * B.transpose()).trace() );
    return res;
};


// inline void cwiseMax(MatrixXd &X, double value) {
//     int outer_index, inner_index;
//
//     int nrow = X.rows();
//     int ncol = X.cols();
//
//     // column major storage for matrix in Eigen
// #if defined(_OPENMP)
// #pragma omp parallel for private(inner_index)
// #endif
//     for(outer_index=0; outer_index<nrow; outer_index++) {
//         for(inner_index=0; inner_index<ncol; inner_index++) {
//             X(outer_index,inner_index) = std::max(X(outer_index,inner_index), value);
//         }
//     }
// };
//
// inline void elemwiseDiv(MatrixXd &res, const MatrixXd &A, const MatrixXd &B, int nrow, int ncol) {
//     int outer_index, inner_index;
//
//     // column major storage for matrix in Eigen
// #if defined(_OPENMP)
// #pragma omp parallel for private(inner_index)
// #endif
//     for(outer_index=0; outer_index<ncol; outer_index++) {
//         for(inner_index=0; inner_index<nrow; inner_index++) {
//             res(inner_index,outer_index) = A(inner_index,outer_index) / B(inner_index,outer_index);
//         }
//     }
// };



}

#endif // PROBABILITY_H
