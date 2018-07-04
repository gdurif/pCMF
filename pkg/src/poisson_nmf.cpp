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
* \brief Implementation of the 'poisson_nmf' class (Non-negative Matrix Factorization in the Poisson setting)
* \author Ghislain Durif
* \version 1.0
* \date 21/02/2018
*/

#include <algorithm>
#include <RcppEigen.h>

#include "model.h"
#include "poisson_nmf.h"
#include "utils/likelihood.h"
#include "utils/matrix.h"
#include "utils/probability.h"
#include "utils/random.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::RowVectorXd;               // variable size row vector, double precision
using Eigen::VectorXd;                  // variable size vector, double precision

namespace pCMF {

// Constructor for the class `poisson_nmf`
poisson_nmf::poisson_nmf(int n, int p, int K, const MatrixXd &X)
    : matrix_factor_model(n, p, K, X) {
    m_oldU = MatrixXd::Ones(n,K);
    m_oldV = MatrixXd::Ones(p,K);
}

// randomly perturb parameters
void poisson_nmf::perturb_param(myRandom::RNGType &rng, double noise_level) {
    MatrixXd noise_U(m_n, m_K);
    MatrixXd noise_V(m_p, m_K);
    myRandom::rUnif(noise_U, m_n, m_K, -noise_level, noise_level, rng);
    myRandom::rUnif(noise_V, m_p, m_K, -noise_level, noise_level, rng);

    m_U += noise_U;
    m_V += noise_V;
    // remove negative values
    m_U = m_U.cwiseMax(1e-6);
    m_V = m_V.cwiseMax(1e-6);
    // compute UV^t
    m_UVt = m_U * m_V.transpose();
}

// update rules for parameters in the optimization process
void poisson_nmf::update_param() {

    // ------------- update V ------------- //
    // colsum on U
    RowVectorXd colsumU = m_U.colwise().sum();

    // compute UV^t => done at previous step (or during init for first step, c.f. 'matrix_factor_model' implementation in 'model.cpp')
    // element-wise X / UVt
    MatrixXd tmp = (m_X.array() / m_UVt.array()).matrix();

    // update rule
    m_V.array() *= (( tmp.transpose() * m_U ).array().rowwise() / colsumU.array()).array();

    // ------------- update U ------------- //
    // colsum on V
    RowVectorXd colsumV = m_V.colwise().sum();

    // compute UV^t
    m_UVt = m_U * m_V.transpose();
    // element-wise X / UVt
    tmp = (m_X.array() / m_UVt.array()).matrix();

    // update rule
    m_U.array() *= (( tmp * m_V ).array().rowwise() / colsumV.array()).array();

    // ------------- post-processing ------------- //

    // avoid under-flowing
    m_U = m_U.cwiseMax(1e-18);
    m_V = m_V.cwiseMax(1e-18);

    // compute UV^t
    m_UVt = m_U * m_V.transpose();
}

// update parameters values between iterations
void poisson_nmf::prepare_next_iterate() {
    m_oldU = m_U;
    m_oldV = m_V;
}

// compute absolute and normalized gap of parameters between two iterates
void poisson_nmf::gap_between_iterates(double &abs_gap, double& norm_gap) {
    double paramNorm = sqrt(m_oldU.squaredNorm() + m_oldV.squaredNorm());
    double diffNorm = sqrt((m_oldU - m_U).squaredNorm() + (m_V - m_oldV).squaredNorm());
    abs_gap = diffNorm;
    norm_gap = diffNorm / paramNorm;
}

// compute a convergence criterion to assess convergence based on the RV coefficients
double poisson_nmf::custom_conv_criterion() {
    double res1 = matrix::rv_coeff(m_U, m_oldU);
    double res2 = matrix::rv_coeff(m_V, m_oldV);
    return(1 - std::min(res1, res2));
}

// compute the optimization criterion associated to the Poisson NMF model
double poisson_nmf::optim_criterion() {
    return matrix::elem_wise_bivar_func_sum(m_X, m_UVt, m_n, m_p, probability::bregman_poisson);
}

// compute the Poisson log-likelihood associated to the Poisson NMF model
double poisson_nmf::loglikelihood() {
    return likelihood::poisson_loglike(m_X, m_UVt);
}

// compute the deviance associated to the Poisson NMF model
double poisson_nmf::deviance() {
    double res1 = likelihood::poisson_loglike(m_X, m_UVt);
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    return(-2 * (res1 - res2));
}

// compute the percentage of explained deviance associated to the Poisson NMF model
double poisson_nmf::exp_deviance() {
    double res1 = likelihood::poisson_loglike(m_X, m_UVt);
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    double res3 = likelihood::poisson_loglike_vec(m_X, m_X.colwise().mean());
    return( (res1 - res3) / (res2 - res3) );
}

// compute partial deviance using a sub-set of factors (among 1...K)
double poisson_nmf::partial_deviance(const vector<int> &factor, int k) {

    // permutation based on 'factor'
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(m_K);
    for(int l=0; l<m_K; l++) {
        perm.indices()[l] = factor[l];
    }

    MatrixXd tmp_U = (m_U * perm).leftCols(k);
    MatrixXd tmp_V = (m_V * perm).leftCols(k);

    // MatrixXd tmp_U(m_n, k);
    // MatrixXd tmp_V(m_p, k);
    // for(int l=0; l<k; l++) {
    //     tmp_U.col(l) = m_U.col(factor[l]);
    //     tmp_V.col(l) = m_V.col(factor[l]);
    // }

    double res1 = likelihood::poisson_loglike(m_X, tmp_U * tmp_V.transpose());
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    return(-2 * (res1 - res2));
}

}
