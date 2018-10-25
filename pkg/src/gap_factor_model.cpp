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
* \brief implementations of the Gamma Poisson Factor model and derivatives
* \author Ghislain Durif
* \version 1.0
* \date 07/02/2018
*/

#if defined(_OPENMP)
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#endif

#include <Rcpp.h>
#include <RcppEigen.h>

#include <stdio.h>

#include "gap_factor_model.h"
#include "utils/internal.h"
#include "utils/likelihood.h"
#include "utils/matrix.h"
#include "utils/probability.h"
#include "utils/random.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::Map;                       // 'maps' rather than copies
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::PermutationMatrix;         // variable size permutation matrix

#define mdigammaInv() unaryExpr(std::bind2nd(std::pointer_to_binary_function<double,int,double>(internal::digammaInv),6))
#define mlgamma() unaryExpr(std::ptr_fun<double,double>(lgamma))
#define mlog() unaryExpr(std::ptr_fun<double,double>(std::log))

using internal::digammaInv;

namespace pCMF {

// Constructor for the class `gap_factor_model`
gap_factor_model::gap_factor_model(int n, int p, int K, const MatrixXd &X)
    : variational_matrix_factor_model(n, p, K, X) {

    // prior parameter
    m_alpha1cur = MatrixXd::Ones(n, K);
    m_alpha2cur = MatrixXd::Ones(n, K);
    m_beta1cur = MatrixXd::Ones(p, K);
    m_beta2cur = MatrixXd::Ones(p, K);

    m_alpha1old = MatrixXd::Ones(n, K);
    m_alpha2old = MatrixXd::Ones(n, K);
    m_beta1old = MatrixXd::Ones(p, K);
    m_beta2old = MatrixXd::Ones(p, K);

    // prior parameter
    m_a1cur = MatrixXd::Ones(n, K);
    m_a2cur = MatrixXd::Ones(n, K);
    m_b1cur = MatrixXd::Ones(p, K);
    m_b2cur = MatrixXd::Ones(p, K);

    m_a1old = MatrixXd::Ones(n, K);
    m_a2old = MatrixXd::Ones(n, K);
    m_b1old = MatrixXd::Ones(p, K);
    m_b2old = MatrixXd::Ones(p, K);

    // additional sufficient statistics
    m_EZ_i = MatrixXd::Zero(p,K);
    m_EZ_j = MatrixXd::Zero(n,K);

    m_exp_ElogU_ElogV_k = MatrixXd::Zero(n,p);

    // remove useless attributes U and V (replaced by EU and EV)
    m_U.resize(0,0);
    m_V.resize(0,0);

}

// destructor for the class `gap_factor_model`
gap_factor_model::~gap_factor_model() {}

/*!
 * \brief Initialize variational and hyper-parameters with given values
 */
void gap_factor_model::init_all_param(const MatrixXd &alpha1, const MatrixXd &alpha2,
                                      const MatrixXd &beta1, const MatrixXd &beta2,
                                      const MatrixXd &a1, const MatrixXd &a2,
                                      const MatrixXd &b1, const MatrixXd &b2) {
    m_alpha1cur = alpha1;
    m_alpha2cur = alpha2;
    m_beta1cur = beta1;
    m_beta2cur = beta2;
    m_a1cur = a1;
    m_a2cur = a2;
    m_b1cur = b1;
    m_b2cur = b2;
    this->U_stats();
    this->V_stats();
    this->update_poisson_param();
}

// Initialize variational parameters with given values
void gap_factor_model::init_variational_param(const MatrixXd &a1, const MatrixXd &a2,
                                              const MatrixXd &b1, const MatrixXd &b2) {
    m_a1cur = a1;
    m_a2cur = a2;
    m_b1cur = b1;
    m_b2cur = b2;
    this->U_stats();
    this->V_stats();
    this->update_poisson_param();
    this->update_hyper_param();
}

// Initialize variational and hyper-parameters with given values
void gap_factor_model::init_hyper_param(const MatrixXd &alpha1, const MatrixXd &alpha2,
                                        const MatrixXd &beta1, const MatrixXd &beta2) {
    m_alpha1cur = alpha1;
    m_alpha2cur = alpha2;
    m_beta1cur = beta1;
    m_beta2cur = beta2;
}

// Initialize variational parameters with from given factor matrices U and V
void gap_factor_model::init_from_factor(const MatrixXd &U, const MatrixXd &V) {
    m_a1cur = U;
    m_a2cur = MatrixXd::Ones(m_n, m_K);
    m_b1cur = V;
    m_b2cur = MatrixXd::Ones(m_p, m_K);
    this->U_stats();
    this->V_stats();
    this->update_poisson_param();
    this->update_hyper_param();
}

// Initialize variational and hyper-parameters with random values
void gap_factor_model::random_init_model_param(myRandom::RNGType &rng) {
    m_a1cur = MatrixXd::Ones(m_n, m_K);
    m_a2cur = MatrixXd::Ones(m_n, m_K);
    m_b1cur = MatrixXd::Ones(m_p, m_K);
    m_b2cur = MatrixXd::Ones(m_p, m_K);

    VectorXd X_row_mean = m_X.rowwise().mean();
    VectorXd X_col_mean = m_X.colwise().mean();

    VectorXd tmp = VectorXd::Ones(m_K);

    // init a1cur_{ik} from a Gamma distribution of parameters \code{(1,sqrt(K)/sqrt(mean(X_j)_i))}
    for(int i=0; i<m_n; i++) {
        myRandom::rGamma(tmp, m_K, 1, std::sqrt(m_K/X_row_mean(i)), rng);
        m_a1cur.row(i) = tmp;
    }

    // init b1cur_{jk} from a Gamma distribution of parameters \code{(1,sqrt(K)/sqrt(mean(X_i)_j))}
    for(int j=0; j<m_p; j++) {
        myRandom::rGamma(tmp, m_K, 1, std::sqrt(m_K/X_col_mean(j)), rng);
        m_b1cur.row(j) = tmp;
    }

    this->U_stats();
    this->V_stats();
    this->update_poisson_param();
    this->update_hyper_param();
}

// randomly perturb parameters
void gap_factor_model::perturb_param(myRandom::RNGType &rng, double noise_level) {
    MatrixXd noise_a1(m_n, m_K);
    MatrixXd noise_b1(m_p, m_K);
    myRandom::rUnif(noise_a1, m_n, m_K, -noise_level, noise_level, rng);
    myRandom::rUnif(noise_b1, m_p, m_K, -noise_level, noise_level, rng);

    m_a1cur += noise_a1;
    m_b1cur += noise_b1;
    // remove negative values
    m_a1cur = m_a1cur.cwiseMax(1e-6);
    m_b1cur = m_b1cur.cwiseMax(1e-6);
    this->U_stats();
    this->V_stats();
    this->update_poisson_param();
    this->update_hyper_param();
}

// update variational parameters
void gap_factor_model::update_variational_param() {
    // Multinomial parameters
#if defined(_DEV)
    Rcpp::Rcout << "Multinomial parameters" << std::endl;
#endif
    this->update_variational_multinomial_param();
    // Gamma parameters
#if defined(_DEV)
    Rcpp::Rcout << "Gamma parameters" << std::endl;
#endif
    this->update_variational_gamma_param();
    // Poisson rate
#if defined(_DEV)
    Rcpp::Rcout << "Poisson rate" << std::endl;
#endif
    this->update_poisson_param();

    // Rcpp::Rcout << "### m_EU = " << std::endl;
    // Rcpp::Rcout << m_EU << std::endl;
    // Rcpp::Rcout << "### m_ElogU = " << std::endl;
    // Rcpp::Rcout << m_ElogU << std::endl;
    // Rcpp::Rcout << "### m_ElogV = " << std::endl;
    // Rcpp::Rcout << m_ElogV << std::endl;
    //
    // Rcpp::Rcout << "### m_UVt = " << std::endl;
    // Rcpp::Rcout << m_UVt << std::endl;
    //
    // Rcpp::Rcout << "### m_alpha1 = " << std::endl;
    // Rcpp::Rcout << m_alpha1cur << std::endl;
    // Rcpp::Rcout << "### m_alpha2 = " << std::endl;
    // Rcpp::Rcout << m_alpha2cur << std::endl;
    // Rcpp::Rcout << "### m_beta1 = " << std::endl;
    // Rcpp::Rcout << m_beta1cur << std::endl;
    // Rcpp::Rcout << "### m_beta2 = " << std::endl;
    // Rcpp::Rcout << m_beta2cur << std::endl;
}

// update prior hyper-parameters
void gap_factor_model::update_hyper_param() {
    // Gamma parameters
    this->update_prior_gamma_param();
}

// update variational parameters values between iterations
void gap_factor_model::prepare_next_iterate_variational_param() {
    m_a1old = m_a1cur;
    m_a2old = m_a2cur;
    m_b1old = m_b1cur;
    m_b2old = m_b2cur;
}

// update prior hyper-parameters values between iterations
void gap_factor_model::prepare_next_iterate_hyper_param() {
    m_alpha1old = m_alpha1cur;
    m_alpha2old = m_alpha2cur;
    m_beta1old = m_beta1cur;
    m_beta2old = m_beta2cur;
}

// compute absolute and normalized gap of parameters between two iterates
void gap_factor_model::gap_between_iterates(double &abs_gap, double& norm_gap) {
    double paramNorm = sqrt(m_a1old.squaredNorm() + m_a2old.squaredNorm()
                                + m_b1old.squaredNorm() + m_b2old.squaredNorm());
    double diffNorm = sqrt((m_a1cur - m_a1old).squaredNorm() + (m_a2cur - m_a2old).squaredNorm()
                               + (m_b1cur - m_b1old).squaredNorm() + (m_b2cur - m_b2old).squaredNorm());
    abs_gap = diffNorm;
    norm_gap = diffNorm / paramNorm;
}

// compute a convergence criterion to assess convergence based on the RV coefficients
double gap_factor_model::custom_conv_criterion() {
    double res1 = matrix::rv_coeff(m_EU, m_a1old.array()/m_a2old.array());
    double res2 = matrix::rv_coeff(m_EV, m_b1old.array()/m_b2old.array());
    return(1 - std::min(res1, res2));
}

// compute the optimization criterion associated to the GaP factor model
// in the variational framework corresponding to the ELBO
double gap_factor_model::optim_criterion() {
    return(this->elbo());
}

// compute the joint log-likelihood associated to the Gamma-Poisson factor model
double gap_factor_model::loglikelihood() {
    double res1 = likelihood::poisson_loglike(m_X, m_UVt);
    double res2 = likelihood::gamma_loglike(m_EU, m_a1cur, m_a2cur);
    double res3 = likelihood::gamma_loglike(m_EV, m_b1cur, m_b2cur);
    return res1 + res2 + res3;

}

// compute the evidence lower bound for the model
double gap_factor_model::elbo() {
    double res = 0;

    // Gamma compartment
    res += likelihood::gamma_loglike(m_EU, m_ElogU, m_alpha1cur, m_alpha2cur);
    res += probability::gamma_entropy(m_a1cur, m_a2cur);
    res += likelihood::gamma_loglike(m_EV, m_ElogV, m_beta1cur, m_beta2cur);
    res += probability::gamma_entropy(m_b1cur, m_b2cur);

    // Poisson/Mutinomial compartment (after simplification)
    // Rcpp::Rcout << "Poisson/Mutinomial compartment" << std::endl;
    res -= m_UVt.sum();
    // \sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})])
    // Rcpp::Rcout << "sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})]) " << std::endl;
    this->intermediate_update_variational_multinomial_param();
    // res += (m_X.array() * m_exp_ElogU_ElogV_k.mlog().array()).sum();
    int i,j;
    VectorXd tmp = VectorXd::Zero(m_p);
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
    for(j=0; j<m_p; j++) {
        for(i=0; i<m_n; i++) {
            tmp(j) += m_X(i,j) * std::log(m_exp_ElogU_ElogV_k(i,j));
        }
    }
    res += tmp.sum();
    // res -= (m_X.array() + 1).mlgamma().sum();
    tmp = VectorXd::Zero(m_p);
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
    for(j=0; j<m_p; j++) {
        for(i=0; i<m_n; i++) {
            tmp(j) +=  lgamma(m_X(i,j) + 1);
        }
    }
    res -= tmp.sum();
    return(res);
}

// compute the deviance associated to the GaP factor model
double gap_factor_model::deviance() {
    double res1 = likelihood::poisson_loglike(m_X, m_UVt);
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    return(-2 * (res1 - res2));
}

// compute the percentage of explained deviance associated to the GaP factor model
double gap_factor_model::exp_deviance() {
    double res1 = likelihood::poisson_loglike(m_X, m_UVt);
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    double res3 = likelihood::poisson_loglike_vec(m_X, m_X.colwise().mean());
    return( (res1 - res3) / (res2 - res3) );
}

// reorder factor according to the 'm_factor_order' attribute
void gap_factor_model::reorder_factor() {

    // permutation based on 'm_factor_order'
    PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(m_K);
    for(int k=0; k<m_K; k++) {
        perm.indices()[k] = m_factor_order[k];
    }

    m_EU *= perm;
    m_EV *= perm;
    m_ElogU *= perm;
    m_ElogV *= perm;

    m_a1cur *= perm;
    m_a2cur *= perm;
    m_b1cur *= perm;
    m_b2cur *= perm;

    m_alpha1cur *= perm;
    m_alpha2cur *= perm;
    m_beta1cur *= perm;
    m_beta2cur *= perm;
}

// create list of object to return
void gap_factor_model::get_output(Rcpp::List &results) {
    Rcpp::List output = Rcpp::List::create(Rcpp::Named("U") = m_EU,
                                           Rcpp::Named("V") = m_EV);
    results.push_back(output, "factor");

    Rcpp::List var_params = Rcpp::List::create(Rcpp::Named("a1") = m_a1cur,
                                               Rcpp::Named("a2") = m_a2cur,
                                               Rcpp::Named("b1") = m_b1cur,
                                               Rcpp::Named("b2") = m_b2cur);

    Rcpp::List hyper_params = Rcpp::List::create(Rcpp::Named("alpha1") = m_alpha1cur,
                                                 Rcpp::Named("alpha2") = m_alpha2cur,
                                                 Rcpp::Named("beta1") = m_beta1cur,
                                                 Rcpp::Named("beta2") = m_beta2cur);

    results.push_back(var_params, "variational_params");
    results.push_back(hyper_params, "hyper_params");

    Rcpp::List stats = Rcpp::List::create(Rcpp::Named("EU") = m_EU,
                                          Rcpp::Named("EV") = m_EV,
                                          Rcpp::Named("ElogU") = m_ElogU,
                                          Rcpp::Named("ElogV") = m_ElogV);

    results.push_back(stats, "stats");
}

// define getters
// getter for U and V
void gap_factor_model::get_factor(MatrixXd &U, MatrixXd &V) {
    U = m_EU;
    V = m_EV;
}


//--------------------------------------------//
// parameter updates for standard variational //
//--------------------------------------------//

// Compute sufficient statistics for U regarding the variational distribution
void gap_factor_model::U_stats() {
    probability::Egamma(m_a1cur, m_a2cur, m_EU);
    probability::ElogGamma(m_a1cur, m_a2cur, m_ElogU);
}

// Compute sufficient statistics for V regarding the variational distribution
void gap_factor_model::V_stats() {
    probability::Egamma(m_b1cur, m_b2cur, m_EV);
    probability::ElogGamma(m_b1cur, m_b2cur, m_ElogV);
}

// compute partial deviance using a sub-set of factors (among 1...K)
double gap_factor_model::partial_deviance(const vector<int> &factor, int k) {

    // permutation based on 'factor'
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(m_K);
    for(int l=0; l<m_K; l++) {
        perm.indices()[l] = factor[l];
    }

    MatrixXd tmp_U = (m_EU * perm).leftCols(k);
    MatrixXd tmp_V = (m_EV * perm).leftCols(k);

    // Rcpp::Rcout << "k = " << k << std::endl;
    // Rcpp::Rcout << "factor order" << std::endl;
    // for(int l=0; l<k; l++) {
    //     Rcpp::Rcout << factor[l] << " ";
    // }
    // Rcpp::Rcout << std::endl;
    //
    // MatrixXd tmp_U(m_n, k);
    // MatrixXd tmp_V(m_p, k);
    // for(int l=0; l<k; l++) {
    //     tmp_U.col(l) = m_EU.col(factor[l]);
    //     tmp_V.col(l) = m_EV.col(factor[l]);
    // }
    //
    // Rcpp::Rcout << "dim tmpU = " << tmp_U.rows() << "," << tmp_U.cols() << std::endl;

    double res1 = likelihood::poisson_loglike(m_X, tmp_U * tmp_V.transpose());
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    return(-2 * (res1 - res2));
}

// update rule for Poisson intensity matrix
void gap_factor_model::update_poisson_param() {
    m_UVt = m_EU * m_EV.transpose();
}

// update rule for the multinomial parameters in variational framework
void gap_factor_model::update_variational_multinomial_param() {
    int i, j, k;

    // \sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})])
    // Rcpp::Rcout << "sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})]) " << std::endl;
    this->intermediate_update_variational_multinomial_param();

    // \sum_j X_{ij} r_{ijk} = \sum_j E_q[Z_{ijk}]
    // Rcpp::Rcout << "sum_j X_{ij} r_{ijk} = sum_j E_q[Z_{ijk}] " << std::endl;
#if defined(_OPENMP)
#pragma omp parallel for private(j,k)
#endif
    for(i=0; i<m_n; i++) {
        double max_value;
        double res;
        double tmp;
        VectorXd tmpVec(m_K);
        for(k = 0; k<m_K; k++) {
            res = 0;
            for(j=0; j<m_p; j++) {
                tmpVec = m_ElogU.row(i) + m_ElogV.row(j);
                max_value = tmpVec.maxCoeff();
                if(max_value < 100) max_value = 0;
                tmp = tmpVec(k) - max_value;
                res += m_X(i,j) * (tmp >= -100 ? std::exp(tmp) : 3e-44) / (m_exp_ElogU_ElogV_k(i,j) > 0 ? m_exp_ElogU_ElogV_k(i,j) : 1);
            }
            m_EZ_j(i,k) = res;

            // Rcpp::Rcout << "### m_EZ_j(" << i << ", " << k <<") = " << m_EZ_j(i,k) << std::endl;
        }
    }

    // Rcpp::Rcout << "m_EZ_j = " << std::endl;
    // Rcpp::Rcout << m_EZ_j << std::endl;

    // \sum_i X_{ij} r_{ijk} = \sum_i E_q[Z_{ijk}]
    // Rcpp::Rcout << "sum_i X_{ij} r_{ijk} = sum_i E_q[Z_{ijk}] " << std::endl;
#if defined(_OPENMP)
#pragma omp parallel for private(i,k)
#endif
    for(j=0; j<m_p; j++) {
        double max_value;
        double res;
        double tmp;
        VectorXd tmpVec(m_K);
        for(k = 0; k<m_K; k++) {
            res = 0;
            for(i=0; i<m_n; i++) {
                tmpVec = m_ElogU.row(i) + m_ElogV.row(j);
                max_value = tmpVec.maxCoeff();
                if(max_value < 100) max_value = 0;
                tmp = tmpVec(k) - max_value;
                res += m_X(i,j) * (tmp >= -100 ? std::exp(tmp) : 3e-44) / (m_exp_ElogU_ElogV_k(i,j) > 0 ? m_exp_ElogU_ElogV_k(i,j) : 1);
            }
            m_EZ_i(j,k) = res;

            // Rcpp::Rcout << "### m_EZ_i(" << j << ", " << k <<") = " << m_EZ_i(j,k) << std::endl;
        }
    }

    // Rcpp::Rcout << "m_EZ_i = " << std::endl;
    // Rcpp::Rcout << m_EZ_i << std::endl;
}

// intemrediate computation when updating the multinomial parameters in variational framework
void gap_factor_model::intermediate_update_variational_multinomial_param() {
    int i, j, k;
    // \sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})])
    // Rcpp::Rcout << "sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})]) " << std::endl;
#if defined(_OPENMP)
#pragma omp parallel for private(i,k)
#endif
    for(j=0; j<m_p; j++) {
        double max_value;
        double res;
        double tmp;
        VectorXd tmpVec(m_K);
        for(i=0; i<m_n; i++) {
            res = 0;
            tmpVec = m_ElogU.row(i) + m_ElogV.row(j);
            max_value = tmpVec.maxCoeff();
            if(max_value < 100) max_value = 0;
            for(k=0; k<m_K; k++) {
                tmp = tmpVec(k) - max_value;
                res += ( tmp >= -100 ? std::exp(tmp) : 3e-44);
            }
            m_exp_ElogU_ElogV_k(i,j) = res;

            // Rcpp::Rcout << "### m_exp_ElogU_ElogV_k(" << i << ", " << j <<") = " << m_exp_ElogU_ElogV_k(i, j) << std::endl;
        }
    }

    // Rcpp::Rcout << "m_exp_ElogU_ElogV_k = " << std::endl;
    // Rcpp::Rcout << m_exp_ElogU_ElogV_k << std::endl;
}

// rule for variational Gamma parameter in variational framework
void gap_factor_model::update_variational_gamma_param() {
    // factor U
    m_a1cur = m_alpha1cur.array() + m_EZ_j.array();
    m_a2cur = m_alpha2cur.rowwise() + m_EV.colwise().sum();
    this->U_stats();

    // fctor V
    m_b1cur = m_beta1cur.array() + m_EZ_i.array();
    m_b2cur = m_beta2cur.rowwise() + m_EU.colwise().sum();
    this->V_stats();
}

// rule for prior Gamma parameter in variational framework
void gap_factor_model::update_prior_gamma_param() {
    // factor U
    m_alpha1cur = (m_alpha2cur.mlog().rowwise() + m_ElogU.colwise().mean()).mdigammaInv();
    m_alpha2cur = m_alpha1cur.array().rowwise() / m_EU.colwise().mean().array();
    // factor V
    m_beta1cur = (m_beta2cur.mlog().rowwise() + m_ElogV.colwise().mean()).mdigammaInv();
    m_beta2cur = m_beta1cur.array().rowwise() / m_EV.colwise().mean().array();
}

}
