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
 * \brief implementations of the ZI sparse Gamma Poisson Factor model
 * \author Ghislain Durif
 * \version 1.0
 * \date 10/04/2018
 */

#if defined(_OPENMP)
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#endif

#include <Rcpp.h>
#include <RcppEigen.h>

#include <stdio.h>

#include "zi_sparse_gap_factor_model.h"
#include "utils/internal.h"
#include "utils/likelihood.h"
#include "utils/matrix.h"
#include "utils/probability.h"
#include "utils/random.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::Map;                       // 'maps' rather than copies
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::MatrixXi;                  // variable size matrix, integer
using Eigen::PermutationMatrix;         // variable size permutation matrix
using Eigen::RowVectorXd;                  // variable size row vector, double precision
using Eigen::VectorXd;                  // variable size vector, double precision

#define mdigammaInv() unaryExpr(std::bind2nd(std::pointer_to_binary_function<double,int,double>(internal::digammaInv),6))
#define mexpit() unaryExpr(std::ptr_fun<double,double>(internal::expit))
#define mlgamma() unaryExpr(std::ptr_fun<double,double>(lgamma))
#define mlog() unaryExpr(std::ptr_fun<double,double>(std::log))

using internal::digammaInv;

namespace pCMF {

// Constructor for the class `sparse_gap_factor_model`
zi_sparse_gap_factor_model::zi_sparse_gap_factor_model(int n, int p, int K, const MatrixXd &X)
    : sparse_gap_factor_model(n, p, K, X) {

    // ZI compartment
    m_prob_D = MatrixXd::Ones(m_n, m_p);
    m_prior_prob_D = VectorXd::Ones(m_p);
    m_freq_D = VectorXd::Ones(m_p);

}

// destructor for the class `sparse_gap_factor_model`
zi_sparse_gap_factor_model::~zi_sparse_gap_factor_model() {}

// Initialize variational and hyper-parameters with given values
void zi_sparse_gap_factor_model::init_all_param(const MatrixXd &alpha1, const MatrixXd &alpha2,
                                         const MatrixXd &beta1, const MatrixXd &beta2,
                                         const MatrixXd &a1, const MatrixXd &a2,
                                         const MatrixXd &b1, const MatrixXd &b2) {
    this->gap_factor_model::init_all_param(alpha1, alpha2, beta1, beta2,
                                           a1, a2, b1, b2);

    // init ZI param
    this->init_zi_param();
}

// Initialize variational parameters with given values
void zi_sparse_gap_factor_model::init_variational_param(const MatrixXd &a1, const MatrixXd &a2,
                                                 const MatrixXd &b1, const MatrixXd &b2) {
    this->gap_factor_model::init_variational_param(a1, a2, b1, b2);

    // init ZI param
    this->init_zi_param();
}

// Initialize variational and hyper-parameters with given values
void zi_sparse_gap_factor_model::init_hyper_param(const MatrixXd &alpha1, const MatrixXd &alpha2,
                                           const MatrixXd &beta1, const MatrixXd &beta2) {
    this->gap_factor_model::init_hyper_param(alpha1, alpha2, beta1, beta2);

    // init ZI param
    this->init_zi_param();
}

// Initialize variational parameters with from given factor matrices U and V
void zi_sparse_gap_factor_model::init_from_factor(const MatrixXd &U, const MatrixXd &V) {
    this->gap_factor_model::init_from_factor(U, V);

    // init ZI param
    this->init_zi_param();
}

// Initialize variational and hyper-parameters with random values
void zi_sparse_gap_factor_model::random_init_model_param(myRandom::RNGType &rng) {
    this->gap_factor_model::random_init_model_param(rng);

    // init ZI param
    this->init_zi_param();
}

// initialize variational and hyper-parameter from ZI compartment
void zi_sparse_gap_factor_model::init_zi_param() {
    m_freq_D = (m_X.array() > 0).cast<double>().colwise().mean();
    m_prior_prob_D = m_freq_D;
    // Rcpp::Rcout << "m_freq_D = " << m_freq_D.transpose() << std::endl;
    // Rcpp::Rcout << "m_freq_D.size() = " << m_freq_D.size() << std::endl;
    // m_prob_D = m_freq_D.transpose().replicate(m_n, 1);
    m_prob_D = (m_X.array() > 0).cast<double>();
    // Rcpp::Rcout << "m_prob_D.dim() = " << m_prob_D.rows() << "," << m_prob_D.cols() << std::endl;
}

// randomly perturb parameters
void zi_sparse_gap_factor_model::perturb_param(myRandom::RNGType &rng, double noise_level) {
    this->gap_factor_model::perturb_param(rng, noise_level);

    // init ZI param
    this->init_zi_param();
}

// update variational parameters
void zi_sparse_gap_factor_model::update_variational_param() {
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
    // ZI compartment
#if defined(_DEV)
    Rcpp::Rcout << "sparse compartment" << std::endl;
#endif
    this->update_variational_sparse_param();
    // Poisson rate
#if defined(_DEV)
    Rcpp::Rcout << "Poisson rate" << std::endl;
#endif
    this->update_poisson_param();
    // ZI compartment
#if defined(_DEV)
    Rcpp::Rcout << "ZI compartment" << std::endl;
#endif
    this->update_variational_zi_param();
}

// update prior hyper-parameters
void zi_sparse_gap_factor_model::update_hyper_param() {
    // Gamma parameters
    this->update_prior_gamma_param();
    // sparse compartment
    this->update_prior_sparse_param();
    // ZI compartment
    this->update_prior_zi_param();
}

// compute the optimization criterion associated to the GaP factor model
// in the variational framework corresponding to the ELBO
double zi_sparse_gap_factor_model::optim_criterion() {
    return(this->elbo());
}

// compute the joint log-likelihood associated to the Gamma-Poisson factor model
double zi_sparse_gap_factor_model::loglikelihood() {
    double res1 = likelihood::zi_poisson_loglike(m_X, m_UVt, m_prob_D);
    double res2 = likelihood::gamma_loglike(m_EU, m_a1cur, m_a2cur);
    double res3 = likelihood::gamma_loglike(m_EV, m_b1cur, m_b2cur);
    double res4 = likelihood::bernoulli_loglike(m_prob_D, m_prob_D);
    double res5 = likelihood::bernoulli_loglike(m_prob_S, m_prob_S);
    return res1 + res2 + res3 + res4 + res5;

}

// compute the evidence lower bound for the model
double zi_sparse_gap_factor_model::elbo() {
    double res = 0;

    // Gamma compartment
    res += likelihood::gamma_loglike(m_EU, m_ElogU, m_alpha1cur, m_alpha2cur);
    res += probability::gamma_entropy(m_a1cur, m_a2cur);
    res += likelihood::gamma_loglike(m_EV, m_ElogV, m_beta1cur, m_beta2cur);
    res += probability::gamma_entropy(m_b1cur, m_b2cur);

    // Bernoulli compartment
    res += likelihood::bernoulli_loglike_vec(m_prob_S, m_prior_prob_S);
    res -= likelihood::bernoulli_loglike(m_prob_S, m_prob_S);

    res += likelihood::bernoulli_loglike_vec(m_prob_D, m_prior_prob_D);
    res -= likelihood::bernoulli_loglike(m_prob_D, m_prob_D);

    // Poisson/Mutinomial compartment (after simplification)
    // Rcpp::Rcout << "Poisson/Mutinomial compartment" << std::endl;
    this->update_poisson_param();
    res -= (m_prob_D.array() * m_UVt.array()).sum();
    // \sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})])
    // Rcpp::Rcout << "sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})]) " << std::endl;
    this->intermediate_update_variational_multinomial_param();

    // \sum_i S_{jk} * X_{ij} * r_{ijk} * log(sum_l r_{ijl})
    int i,j, k;
    VectorXd tmp_sum = VectorXd::Zero(m_p);

#if defined(_OPENMP)
#pragma omp parallel for private(i,k)
#endif
    for(j=0; j<m_p; j++) {

        MatrixXd r_ik = MatrixXd::Zero(m_n, m_K); // Multinomial param when j=1,...,p is fixed

        for(i=0; i<m_n; i++) {
            VectorXd tmpVec = m_ElogU.row(i) + m_ElogV.row(j);
            double max_value = tmpVec.maxCoeff();
            if(max_value < 100) max_value = 0;
            for(k=0; k<m_K; k++) {
                double tmp = tmpVec(k) - max_value;
                r_ik(i,k) = m_S(j,k) * (tmp >= -100 ? std::exp(tmp) : 3e-44) / (m_exp_ElogU_ElogV_k(i,j) > 0 ? m_exp_ElogU_ElogV_k(i,j) : 1);
            }
        }

        for(k=0; k<m_K; k++) {
            double res = 0;
            for(i=0; i<m_n; i++) {
                // tmp_sum(j) += m_prob_D(i,j) * m_prob_S(j,k) * m_X(i,j) * r_ik(i,k) * std::log(m_exp_ElogU_ElogV_k(i,j));
                tmp_sum(j) += m_prob_D(i,j) * m_prob_S(j,k) * m_X(i,j) * r_ik(i,k) * (m_ElogU(i,k) + m_ElogV(j,k));
            }
        }
    }
    res += tmp_sum.sum();

    // res -= (m_X.array() + 1).mlgamma().sum();
    tmp_sum = VectorXd::Zero(m_p);
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
    for(j=0; j<m_p; j++) {
        for(i=0; i<m_n; i++) {
            tmp_sum(j) += m_prob_D(i,j) * lgamma(m_X(i,j) + 1);
        }
    }
    res -= tmp_sum.sum();
    return(res);
}

// compute the deviance associated to the GaP factor model
double zi_sparse_gap_factor_model::deviance() {
    double res1 = likelihood::poisson_loglike(m_X, m_prob_D.array() * m_UVt.array());
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    // Rcpp::Rcout << "deviance 2 = " << -2 * (res1 - res2) << std::endl;

    return(-2 * (res1 - res2));
}

// compute the percentage of explained deviance associated to the GaP factor model
double zi_sparse_gap_factor_model::exp_deviance() {
    double res1 = likelihood::poisson_loglike(m_X, m_prob_D.array() * m_UVt.array());
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    double res3 = likelihood::poisson_loglike_vec(m_X, m_X.colwise().mean());

    return( (res1 - res3) / (res2 - res3) );
    // return(-1);
}

// create list of object to return
void zi_sparse_gap_factor_model::get_output(Rcpp::List &results) {
    this->sparse_gap_factor_model::get_output(results);

    Rcpp::List ZI = Rcpp::List::create(Rcpp::Named("prob_D") = m_prob_D,
                                       Rcpp::Named("freq_D") = m_freq_D,
                                       Rcpp::Named("prior_prob_D") = m_prior_prob_D);

    results.push_back(ZI, "ZI_param");
}


//--------------------------------------------//
// parameter updates for standard variational //
//--------------------------------------------//

// compute partial deviance using a sub-set of factors (among 1...K)
double zi_sparse_gap_factor_model::partial_deviance(const vector<int> &factor, int k) {

    // permutation based on 'factor'
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(m_K);
    for(int l=0; l<m_K; l++) {
        perm.indices()[l] = factor[l];
    }

    MatrixXd tmp_U = (m_EU * perm).leftCols(k);
    MatrixXd tmp_V = (m_EV * perm).leftCols(k);
    MatrixXd tmp_prob_S = (m_prob_S * perm).leftCols(k);

    // MatrixXd tmp_U(m_n, k);
    // MatrixXd tmp_V(m_p, k);
    // MatrixXd tmp_prob_S(m_p, k);
    // for(int l=0; l<k; l++) {
    //     tmp_U.col(l) = m_U.col(factor[l]);
    //     tmp_V.col(l) = m_V.col(factor[l]);
    //     tmp_prob_S.col(l) = m_prob_S.col(factor[l]);
    // }

    double res1 = likelihood::poisson_loglike(m_X,
                        m_prob_D.array() * (tmp_U * (tmp_prob_S.array() * tmp_V.array()).matrix().transpose()).array());
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    return(-2 * (res1 - res2));
}

// update rule for the multinomial parameters in variational framework
void zi_sparse_gap_factor_model::update_variational_multinomial_param() {
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
                res += m_prob_S(j,k) * m_prob_D(i,j) * m_S(j,k) * m_X(i,j) * (tmp >= -100 ? std::exp(tmp) : 3e-44) / (m_exp_ElogU_ElogV_k(i,j) > 0 ? m_exp_ElogU_ElogV_k(i,j) : 1);
            }
            m_EZ_j(i,k) = res;
        }
    }

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
                res += m_prob_S(j,k) * m_prob_D(i,j) * m_S(j,k) * m_X(i,j) * (tmp >= -100 ? std::exp(tmp) : 3e-44) / (m_exp_ElogU_ElogV_k(i,j) > 0 ? m_exp_ElogU_ElogV_k(i,j) : 1);
            }
            m_EZ_i(j,k) = res;
        }
    }
}

// rule for variational Gamma parameter in variational framework
void zi_sparse_gap_factor_model::update_variational_gamma_param() {
    // factor U
    m_a1cur = m_alpha1cur.array() + m_EZ_j.array();
    m_a2cur = m_alpha2cur.array() + (m_prob_D * (m_EV.array() * m_prob_S.array()).matrix()).array();
    this->U_stats();

    // fctor V
    m_b1cur = m_beta1cur.array() + m_EZ_i.array();

    MatrixXd tmpMat = m_prob_D.transpose() * m_EU;

    int j, k;
#if defined(_OPENMP)
#pragma omp parallel for private(k)
#endif
    for(j=0; j<m_p; j++) {
        double max_value;
        double res;
        double tmp;
        VectorXd tmpVec(m_K);
        for(k=0; k<m_K; k++) {
            m_b2cur(j,k) = m_beta2cur(j,k) + m_prob_S(j,k) * tmpMat(j,k);
        }
    }

    this->V_stats();
}

// update rule for variational parameter from ZI compartment
void zi_sparse_gap_factor_model::update_variational_zi_param() {
    // Rcpp::Rcout << "m_prob_D" << std::endl;
    int i, j;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
    for(j=0; j<m_p; j++) {
        if(m_prior_prob_D(j) == 1) {
            m_prob_D.col(j) = VectorXd::Ones(m_n);
        } else if(m_prior_prob_D(j) == 0) {
            m_prob_D.col(j) = VectorXd::Zero(m_n);
        } else {
            for(i= 0; i<m_n; i++) {
                if(m_X(i,j) != 0) {
                    m_prob_D(i,j) = 1;
                } else {
                    m_prob_D(i,j) = internal::expit( internal::logit(m_prior_prob_D(j)) - m_UVt(i,j));
                }
            }
        }
    }
}

// update rule for variational parameter from ZI compartment
void zi_sparse_gap_factor_model::update_variational_sparse_param() {

    this->intermediate_update_variational_multinomial_param();

    // Rcpp::Rcout << "m_prob_S" << std::endl;
    int i, j, k;
#if defined(_OPENMP)
#pragma omp parallel for private(i,k)
#endif
    for(j=0; j<m_p; j++) {
        if(m_prior_prob_S(j) == 1) {
            m_prob_S.row(j) = RowVectorXd::Ones(m_K);
        } else if(m_prior_prob_S(j) == 0) {
            m_prob_S.row(j) = RowVectorXd::Zero(m_K);
        } else {
            MatrixXd r_ik = MatrixXd::Zero(m_n, m_K); // Multinomial param when j=1,...,p is fixed

            for(i=0; i<m_n; i++) {
                VectorXd tmpVec = m_ElogU.row(i) + m_ElogV.row(j);
                double max_value = tmpVec.maxCoeff();
                if(max_value < 100) max_value = 0;
                for(k=0; k<m_K; k++) {
                    double tmp = tmpVec(k) - max_value;
                    r_ik(i,k) = m_S(j,k) * (tmp >= -100 ? std::exp(tmp) : 3e-44) / (m_exp_ElogU_ElogV_k(i,j) > 0 ? m_exp_ElogU_ElogV_k(i,j) : 1);
                }
            }

            for(k=0; k<m_K; k++) {
                double res = 0;
                for(i=0; i<m_n; i++) {
                    res -= m_prob_D(i,j) * m_EU(i,k) * m_EV(j,k);
                    // res += m_prob_D(i,j) * m_X(i,j) * r_ik(i,k) * std::log(m_exp_ElogU_ElogV_k(i,j));
                    res += m_prob_D(i,j) * m_X(i,j) * r_ik(i,k) * (m_ElogU(i,k) + m_ElogV(j,k));
                }
                m_prob_S(j,k) = internal::expit( internal::logit(m_prior_prob_S(j)) + res);
            }
        }
    }

    // infer indicator
    m_S = (m_prob_S.array() >= m_sel_bound).cast<int>();
}

// update rule for prior parameter from ZI compartment
void zi_sparse_gap_factor_model::update_prior_zi_param() {
    m_prior_prob_D = m_prob_D.colwise().mean();
}

// update rule for prior parameter from sparse compartment
void zi_sparse_gap_factor_model::update_prior_sparse_param() {
    m_prior_prob_S = m_prob_S.rowwise().mean();
}

}
