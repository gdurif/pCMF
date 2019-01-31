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
 * \brief implementations of the sparse Gamma Poisson Factor model
 * \author Ghislain Durif
 * \version 1.0
 * \date 10/04/2018
 */

#if defined(_OPENMP)
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#endif

#include <math.h>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <stdio.h>

#include "sparse_gap_factor_model.h"
#include "utils/likelihood.h"
#include "utils/matrix.h"
#include "utils/probability.h"
#include "utils/random.h"

#include "utils/macros.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::Map;                       // 'maps' rather than copies
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::MatrixXi;                  // variable size matrix, integer
using Eigen::PermutationMatrix;         // variable size permutation matrix
using Eigen::RowVectorXd;                  // variable size row vector, double precision
using Eigen::VectorXd;                  // variable size vector, double precision

namespace pCMF {

// Constructor for the class `sparse_gap_factor_model`
sparse_gap_factor_model::sparse_gap_factor_model(int n, int p, int K, const MatrixXd &X)
    : gap_factor_model(n, p, K, X) {

    // sparse compartment
    m_prob_S = MatrixXd::Zero(m_p, m_K);
    m_prior_prob_S = VectorXd::Zero(m_p);
    m_S = MatrixXi::Ones(m_p, m_K);

    m_sel_bound = 0.5;

}

// destructor for the class `sparse_gap_factor_model`
sparse_gap_factor_model::~sparse_gap_factor_model() {}

// initialize variational and hyper-parameter from ZI compartment
void sparse_gap_factor_model::init_sparse_param(double sel_bound, myRandom::RNGType &rng) {
    m_sel_bound = sel_bound;

    // m_prob_S = MatrixXd::Zero(m_p, m_K).array() + 0.9;
    // m_prior_prob_S = VectorXd::Zero(m_p).array() + 0.9;
    // m_S = MatrixXi::Ones(m_p, m_K);

    // m_prob_S = MatrixXd::Zero(m_p, m_K).array() + 1-m_sel_bound;
    // m_prior_prob_S = VectorXd::Zero(m_p).array() + 1-m_sel_bound;

    // indicators
    // MatrixXd tmp(m_p, m_K);
    // myRandom::rUnif(tmp, m_p, m_K, 0, 1, rng);
    // m_S = (tmp.array() >= 1-m_sel_bound).cast<int>();
    // Rcpp::Rcout << "m_prob_D.dim() = " << m_prob_D.rows() << "," << m_prob_D.cols() << std::endl;

    m_prob_S = MatrixXd::Zero(m_p, m_K).array();
    m_prior_prob_S = VectorXd::Zero(m_p).array();
    myRandom::rUnif(m_prob_S, m_p, m_K, 0, 1, rng);
    m_prior_prob_S = m_prob_S.rowwise().mean();
    m_S = (m_prob_S.array() >= m_sel_bound).cast<int>();
}

// initialize variational and hyper parameters from sparse compartment
// with given values
void sparse_gap_factor_model::init_sparse_param(double sel_bound,
                                                const MatrixXd &prob_S,
                                                const VectorXd &prior_S) {
    m_sel_bound = sel_bound;
    m_prob_S = prob_S;
    m_prior_prob_S = prior_S;

    m_S = (m_prob_S.array() >= m_sel_bound).cast<int>();
}

// randomly perturb parameters
void sparse_gap_factor_model::perturb_param(myRandom::RNGType &rng, double noise_level) {
    this->gap_factor_model::perturb_param(rng, noise_level);
}

// update variational parameters
void sparse_gap_factor_model::update_variational_param() {
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
}

// update prior hyper-parameters
void sparse_gap_factor_model::update_hyper_param() {
    // Gamma parameters
    this->update_prior_gamma_param();
    // sparse compartment
    this->update_prior_sparse_param();
}

// compute the optimization criterion associated to the GaP factor model
// in the variational framework corresponding to the ELBO
double sparse_gap_factor_model::optim_criterion() {
    return(this->elbo());
}

// compute the joint log-likelihood associated to the Gamma-Poisson factor model
double sparse_gap_factor_model::loglikelihood() {
    double res1 = likelihood::poisson_loglike(m_X, m_UVt);
    double res2 = likelihood::gamma_loglike(m_EU, m_a1cur, m_a2cur);
    double res3 = likelihood::gamma_loglike(m_EV, m_b1cur, m_b2cur);
    double res4 = likelihood::bernoulli_loglike(m_prob_S, m_prob_S);
    return res1 + res2 + res3 + res4;

}

// compute the evidence lower bound for the model
double sparse_gap_factor_model::elbo() {
    double res = 0;

    // Gamma compartment
    // Rcpp::Rcout << "Gamma compartment" << std::endl;
    res += likelihood::gamma_loglike(m_EU, m_ElogU, m_alpha1cur, m_alpha2cur);
    res += probability::gamma_entropy(m_a1cur, m_a2cur);
    res += likelihood::gamma_loglike(m_EV, m_ElogV, m_beta1cur, m_beta2cur);
    res += probability::gamma_entropy(m_b1cur, m_b2cur);

    // Bernoulli compartment
    // Rcpp::Rcout << "Benoulli compartment" << std::endl;
    res += likelihood::bernoulli_loglike_vec(m_prob_S, m_prior_prob_S);
    res -= likelihood::bernoulli_loglike(m_prob_S, m_prob_S);

    // Poisson/Mutinomial compartment (after simplification)
    // Rcpp::Rcout << "Poisson/Mutinomial compartment" << std::endl;
    this->update_poisson_param();
    res -= m_UVt.sum();
    // \sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})])
    // Rcpp::Rcout << "sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})]) " << std::endl;
    this->intermediate_update_variational_multinomial_param();

    // \sum_i S_{jk} * X_{ij} * r_{ijk} * log(sum_l r_{ijl})
    int i,j, k;
    VectorXd tmp_vec(m_K);
    double tmp = 0;
    double tmp_val = 0;
    double tmp_acc = 0;
    double max_value = 0;

#if defined(_OPENMP)
#pragma omp parallel for private(i,k,max_value,tmp_val,tmp_vec,tmp_acc) reduction(+:tmp)
#endif
    for(j=0; j<m_p; j++) {

        double r_ijk = 0; // Multinomial param when j=1,...,p is fixed

        for(i=0; i<m_n; i++) {
            tmp_vec = m_ElogU.row(i) + m_ElogV.row(j);
            max_value = tmp_vec.maxCoeff();
            if(max_value < 100) max_value = 0;
            tmp_vec = ((tmp_vec.array() - max_value).matrix().mcexp()).cwiseProduct(m_S.row(j).transpose().cast<double>());
            tmp_val = tmp_vec.sum();
            tmp_val = tmp_val > 0 ? tmp_val : 1;

            for(int k = 0; k<m_K; k++) {
                r_ijk = tmp_vec(k) / tmp_val;
                // tmp += m_prob_S(j,k) * m_X(i,j) * r_ijk * std::log(m_exp_ElogU_ElogV_k(i,j));
                tmp += m_prob_S(j,k) * m_X(i,j) * r_ijk * (m_ElogU(i,k) + m_ElogV(j,k));
            }
        }
    }
    res += tmp;

    // res -= (m_X.array() + 1).mlgamma().sum();
    tmp = 0;
#if defined(_OPENMP)
#pragma omp parallel for private(i) reduction(+:tmp)
#endif
    for(j=0; j<m_p; j++) {
        for(i=0; i<m_n; i++) {
            tmp += lgamma(m_X(i,j) + 1);
        }
    }
    res -= tmp;

    return(res);
}

// compute the deviance associated to the GaP factor model
double sparse_gap_factor_model::deviance() {
    double res1 = likelihood::poisson_loglike(m_X, m_UVt.array());
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    // Rcpp::Rcout << "deviance 2 = " << -2 * (res1 - res2) << std::endl;

    return(-2 * (res1 - res2));
}

// compute the percentage of explained deviance associated to the GaP factor model
double sparse_gap_factor_model::exp_deviance() {
    double res1 = likelihood::poisson_loglike(m_X, m_UVt.array());
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    double res3 = likelihood::poisson_loglike_vec(m_X, m_X.colwise().mean());

    return( (res1 - res3) / (res2 - res3) );
    // return(-1);
}

// reorder factor according to the 'm_factor_order' attribute
void sparse_gap_factor_model::reorder_factor() {

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

    m_S *= perm;
    m_prob_S *= perm;
}

// create list of object to return
void sparse_gap_factor_model::get_output(Rcpp::List &results) {
    this->gap_factor_model::get_output(results);

    Rcpp::List sparse = Rcpp::List::create(Rcpp::Named("prob_S") = m_prob_S,
                                           Rcpp::Named("prior_prob_S") = m_prior_prob_S,
                                           Rcpp::Named("S") = m_S);

    results.push_back(sparse, "sparse_param");
}


//--------------------------------------------//
// parameter updates for standard variational //
//--------------------------------------------//

// compute partial deviance using a sub-set of factors (among 1...K)
double sparse_gap_factor_model::partial_deviance(const vector<int> &factor, int k) {

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
    //     tmp_U.col(l) = m_EU.col(factor[l]);
    //     tmp_V.col(l) = m_EV.col(factor[l]);
    //     tmp_prob_S.col(l) = m_prob_S.col(factor[l]);
    // }

    double res1 = likelihood::poisson_loglike(m_X,
                                              tmp_U * (tmp_prob_S.array() * tmp_V.array()).matrix().transpose());
    double res2 = likelihood::poisson_loglike(m_X, m_X);
    return(-2 * (res1 - res2));
}

// update rule for Poisson intensity matrix
void sparse_gap_factor_model::update_poisson_param() {
    m_UVt = m_EU * (m_prob_S.array() * m_EV.array()).matrix().transpose();
}

// update rule for the multinomial parameters in variational framework
void sparse_gap_factor_model::update_variational_multinomial_param() {
    int i, j, k;

    // \sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})])
    // \sum_i X_{ij} r_{ijk} = \sum_i E_q[Z_{ijk}]
    // \sum_j X_{ij} r_{ijk} = \sum_j E_q[Z_{ijk}]
    // new version with reduce

    VectorXd tmp_vec(m_K);
    double tmp_val = 0;
    double tmp_acc = 0;
    double max_value = 0;

    MatrixXd tmp_EZ_i = MatrixXd::Zero(m_p,m_K);
    MatrixXd tmp_EZ_j = MatrixXd::Zero(m_n,m_K);

#if defined(_OPENMP)
#pragma omp declare reduction (+: Eigen::MatrixXd: omp_out=omp_out+omp_in)\
    initializer(omp_priv=MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))
#pragma omp parallel for private(i,k,max_value,tmp_acc,tmp_val,tmp_vec) reduction(+:tmp_EZ_i,tmp_EZ_j)
#endif
    for(j=0; j<m_p; j++) {
        for(i=0; i<m_n; i++) {
            tmp_vec = m_ElogU.row(i) + m_ElogV.row(j);
            max_value = tmp_vec.maxCoeff();
            if(max_value < 100) max_value = 0;
            tmp_vec = ((tmp_vec.array() - max_value).matrix().mcexp()).cwiseProduct(m_S.row(j).transpose().cast<double>());
            tmp_val = tmp_vec.sum();
            m_exp_ElogU_ElogV_k(i,j) = tmp_val;
            tmp_val = tmp_val > 0 ? tmp_val : 1;

            for(int k = 0; k<m_K; k++) {
                tmp_acc = m_prob_S(j,k) * m_X(i,j) * tmp_vec(k) / tmp_val;
                tmp_EZ_i(j,k) += tmp_acc;
                tmp_EZ_j(i,k) += tmp_acc;
            }
        }
    }

    m_EZ_i = tmp_EZ_i;
    m_EZ_j = tmp_EZ_j;
}

// intemrediate computation when updating the multinomial parameters in variational framework
void sparse_gap_factor_model::intermediate_update_variational_multinomial_param() {
    int i, j, k;

    VectorXd tmp_vec(m_K);
    double max_value = 0;

    // \sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})])
    // Rcpp::Rcout << "sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})]) " << std::endl;
#if defined(_OPENMP)
#pragma omp parallel for private(i,k,max_value,tmp_vec)
#endif
    for(j=0; j<m_p; j++) {
        for(i=0; i<m_n; i++) {
            tmp_vec = m_ElogU.row(i) + m_ElogV.row(j);
            max_value = tmp_vec.maxCoeff();
            if(max_value < 100) max_value = 0;
            tmp_vec = ((tmp_vec.array() - max_value).matrix().mcexp()).cwiseProduct(m_S.row(j).transpose().cast<double>());
            m_exp_ElogU_ElogV_k(i,j) = tmp_vec.sum();
        }
    }

    // Rcpp::Rcout << "m_exp_ElogU_ElogV_k = " << std::endl;
    // Rcpp::Rcout << m_exp_ElogU_ElogV_k << std::endl;
}

// rule for variational Gamma parameter in variational framework
void sparse_gap_factor_model::update_variational_gamma_param() {
    // factor U
    m_a1cur = m_alpha1cur.array() + m_EZ_j.array();
    m_a2cur = m_alpha2cur.array().rowwise() + (m_EV.array() * m_prob_S.array()).colwise().sum();
    this->U_stats();

    // fctor V
    m_b1cur = m_beta1cur.array() + m_EZ_i.array();

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
            m_b2cur(j,k) = m_beta2cur(j,k) + m_prob_S(j,k) * m_EU.col(k).sum();
        }
    }

    this->V_stats();
}

// update rule for variational parameter from ZI compartment
void sparse_gap_factor_model::update_variational_sparse_param() {

    int i,j, k;
    VectorXd tmp_vec(m_K);
    MatrixXd tmp_mat = MatrixXd::Zero(m_p,m_K);
    double tmp = 0;
    double tmp_val = 0;
    double tmp_acc = 0;
    double max_value = 0;

#if defined(_OPENMP)
#pragma omp parallel for private(i,k,max_value,tmp_val,tmp_vec,tmp_acc)
#endif
    for(j=0; j<m_p; j++) {

        if(m_prior_prob_S(j) == 1) {
            m_prob_S.row(j) = RowVectorXd::Ones(m_K);
        } else if(m_prior_prob_S(j) == 0) {
            m_prob_S.row(j) = RowVectorXd::Zero(m_K);
        } else {

            double r_ijk = 0; // Multinomial param when j=1,...,p is fixed

            for(i=0; i<m_n; i++) {
                tmp_vec = m_ElogU.row(i) + m_ElogV.row(j);
                max_value = tmp_vec.maxCoeff();
                if(max_value < 100) max_value = 0;
                tmp_vec = ((tmp_vec.array() - max_value).matrix().mcexp()).cwiseProduct(m_S.row(j).transpose().cast<double>());
                tmp_val = tmp_vec.sum();
                tmp_val = tmp_val > 0 ? tmp_val : 1;

                for(int k = 0; k<m_K; k++) {
                    r_ijk = tmp_vec(k) / tmp_val;

                    tmp_mat(j,k) -= m_EU(i,k) * m_EV(j,k);
                    tmp_mat(j,k) += m_X(i,j) * r_ijk * (m_ElogU(i,k) + m_ElogV(j,k));
                }
            }

            for(k=0; k<m_K; k++) {
                m_prob_S(j,k) = internal::expit( internal::logit(m_prior_prob_S(j)) + tmp_mat(j,k));
            }
        }
    }

    // infer indicator
    m_S = (m_prob_S.array() >= m_sel_bound).cast<int>();
}

// update rule for prior parameter from sparse compartment
void sparse_gap_factor_model::update_prior_sparse_param() {
    m_prior_prob_S = m_prob_S.rowwise().mean();
}

}
