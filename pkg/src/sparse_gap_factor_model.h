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
 * \brief definitions of the sparse Gamma Poisson Factor model
 * \author Ghislain Durif
 * \version 1.0
 * \date 10/04/2018
 */

#ifndef SPARSE_GAP_FACTOR_MODEL_H
#define SPARSE_GAP_FACTOR_MODEL_H

#include <RcppEigen.h>

#include "gap_factor_model.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::MatrixXi;                  // variable size matrix, integer
using Eigen::VectorXd;                  // variable size vector, double precision

namespace pCMF {

/*!
 * \brief definition of the Sparse Gamma Poisson Factor model
 *
 * The sparsity concerns the factor matrix V
 *
 * Model:
 *     * \f$ X_{ij} = \sum_k Z_{ijk} \f$
 *     * \f$ X | U,V \sim Poisson(U V^t) \f$
 *       equivalent to \f$ Z_{ijk} | U_{ik}, V_{jk} \sim Poisson(U_{ik} V_{jk}) \f$
 *     * \f$ U_{ik} \sim Gamma(\alpha_{ik}) \f$ with \f$ \alpha_{ik} = (\alpha_{ik,1}, \alpha_{ik,2}) \f$
 *     * \f$ V_{jk} \sim (1-p_{k}^s) delta_0(V_{jk}) + p_{k}^s Gamma(\beta_{jk}) \f$ with \f$ \beta_{jk} = (\beta_{jk,1}, \beta_{jk,2}) \f$
 *       which corresponds to a two-group prior in a spike-and-slab setting (for probabilistic variable selection).
 *       In practice, we introduce the variables \f$ V'_{jk} \sim Gamma(\beta_{jk}) \f$ and Bernoulli
 *       variables \f$ S_{jk} \sim Bernoulli(pi_j^s) \f$ indicating is \f$ V_{jk} \f$ is selected or not \f$.
 *       Note:
 *           - In our approach, all \f$ \alpha_{ik} \f$ are identical across i.
 *           - Similarly, all \f$ \beta_{jk} \f$ are identical across j.
 *     * Consequence 1: \f$ Z_{ijk} | U_{ik}, V'_{jk}, S_{jk} \sim Poisson(U_{ik} V'_{jk} S_{jk}) \f$
 *     * Consequence 2: \f$ (Z_{ijk})_k \sim Multinomial((\rho_{ijk})_k)
 *       where \f$ \rho_{ijk} = \frac{U_{ik} V'_{jk} S_{jk}}{\sum_l U_{il} V'_{jl} S_{jl}} \f$
 *
 * Variational distribution \f$ q \f$:
 *     * \f$ (Z_{ijk})_k \sim_q Multinomial((r_{ijk})_k)
 *     * \f$ U_{ik} \sim_q Gamma(a_{ik}) \f$ with \f$ a_{ik} = (a_{ik,1}, a_{ik,2}) \f$
 *     * \f$ V'_{jk} \sim_q Gamma(b_{jk}) \f$ with \f$ b_{jk} = (b_{jk,1}, b_{jk,2}) \f$
 *     * \f$ S_{jk} \sim_q Bernoulli(p_{jk}^s) \f$
 *
 * Sufficient statitics needed:
 *     * \f$ E_q[U_{ik}] \f$, \f$ E_q[log(U_{ik})] \f$
 *     * \f$ E_q[V'_{jk}] \f$, \f$ E_q[log(V'_{jk})] \f$
 *     * \f$ \sum_i E_q[D_{ij}] E_q[Z_{ijk}] = \sum_i p_{ij}^d * X_{ij} r_{ijk} \f$
 *     * \f$ \sum_j E_q[D_{ij}] E_q[Z_{ijk}] = \sum_j p_{ij}^d * X_{ij} r_{ijk} \f$
 *     * \f$ \sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})]) \f$
 *
 */
class sparse_gap_factor_model : public gap_factor_model {

protected:

    // sparse compartment
    MatrixXd m_prob_S;          /*!< matrix of probability for variational distribution over selection indicator S, dimension p x K */
    VectorXd m_prior_prob_S;    /*!< vector of probability for prior distribution over selection indicator S, length p */
    MatrixXi m_S;               /*!< matrix of selection indicator estimated value \f$ \hat{S} \fS, dimension p x K */

    double m_sel_bound;         /*!< real value s in (0,1) used to infer S values (\f$ S_{jk} = 1 \f$ if \f$ p_{jk}^s > s \f$) */

public:
    /*!
     * \brief Constructor for the class `sparse_gap_factor_model`
     */
    sparse_gap_factor_model(int n, int p, int K, const MatrixXd &X);

    /*!
     * \brief destructor for the class `sparse_gap_factor_model`
     */
    ~sparse_gap_factor_model();

    /*!
     * \brief initialize variational and hyper parameters from sparse compartment
     *
     * Set m_sel_bound attribute with sel_bound input parameter
     *
     * Prior probabilities over D i.e. \f$ (pi_j^s)_j \f$ are initialized with
     * the value 1-m_sel_bound
     *
     * Variational probabilities over S i.e. \f$ (p_{jk}^s_j \f$ are
     * initialized with the value 1-m_sel_bound
     *
     * Indicators \f$ S_{jk} \f$ are randomly generated from Bernoulli(1-m_sel_bound)
     *
     * \param[in] sel_bound real value in [0,1] used to threshold sparsity
     * probabilities for factor V
     * \param[in,out] rng Boost random number generator
     */
    virtual void init_sparse_param(double sel_bound, myRandom::RNGType &rng);

    /*!
     * \brief randomly perturb parameters
     *
     * \param[in,out] rng random number generator
     * \param[in] noise_level level of the perturbation, based on a uniform
     * distribution on [-noise_level, noise_level]
     */
    virtual void perturb_param(myRandom::RNGType &rng, double noise_level);

    /*!
     * \brief update variational parameters
     */
    virtual void update_variational_param();

    /*!
     * \brief update prior hyper-parameters
     */
    virtual void update_hyper_param();

    /*!
     * \brief compute the optimization criterion associated to the GaP factor model
     * in the variational framework corresponding to the ELBO
     */
    virtual double optim_criterion();

    /*!
     * \brief compute the joint (or complete) log-likelihood associated to the
     * Gamma-Poisson factor model
     *
     * \f[
     * \log p(X | \Lambda = UV^t) + \log p(U; \alpha) + \log p(V'; \beta)
     * + log p(S; (p_j^s)_j)
     * \f]
     *
     * recalling that \f$ V_{jk} = S_{jk} V'_{jk} \f$
     *
     * where \f$ \log p(X | \Lambda ) \f$ is the Poisson log-likelihood and
     * \f$ \Lambda \f$ the Poisson rate matrix, i.e
     *
     * \f[
     * \log p(X | \Lambda ) = \log p(x_{ij} | \lambda_{ij})
     * \f]
     *
     * \f$ \log p(U; \alpha) \f$ and \f$ \log p(V'; \beta) \f$ are
     * the Gamma log-likelihood for U and V respectively
     *
     * and \f$ \log p(S; (p_j^s)_j) \f$ the Bernoulli log-likelihood
     */
    virtual double loglikelihood();

    /*!
     * \brief compute the evidence lower bound for the model
     */
    virtual double elbo();

    /*!
     * \brief compute the deviance associated to the GaP factor model
     *
     * \f[
     * deviance = -2 \times [ \log p(X | \Lambda = U (S * V)^t) - \log p(X | \Lambda = X)]
     * \f]
     *
     * where \f$ \log p(X | \Lambda ) \f$ is the Poisson log-likelihood and
     * \f$ \Lambda \f$ the Poisson rate matrix, i.e
     *
     * \f[
     * \log p(X | \Lambda ) = \log p(x_{ij} | \lambda_{ij})
     * \f]
     *
     * and \f$ S * V \f$ is the element-wise product between \f$ S \f$ and \f$ V f$.
     *
     * This is equivalent to computing the Bregman divergence
     * between \f$ X \f$ and \f$ U (S * V)^t \f$ in the Poisson framework:
     *
     * \f[
     * d(X,Y) = \sum_{i,j} d(x_{i,j}, y_{i,j})
     * \f]
     *
     * with \f$ d(x,y) = x \log\frac{x}{y} - x + y \f$
     */
    virtual double deviance();

    /*!
     * \brief compute the percentage of explained deviance associated
     * to the Gap factor model
     *
     * \f[
     * %deviance = \frac{ \log p(X | \Lambda = U (S * V)^t) - \log p(X | \Lambda = \bar{X})}{ \log p(X | \Lambda = X) - \log p(X | \Lambda = \bar{X}) }
     * \f]
     *
     * (c.f. deviance doc for \f$ \log p(X | \Lambda = U (S * V)^t) \f$ definition)
     *
     * where \f$ \log p(X | \Lambda ) \f$ is the Poisson log-likelihood and
     * \f$ \Lambda \f$ the Poisson rate matrix, i.e
     *
     * \f[
     * \log p(X | \Lambda ) = \log p(x_{ij} | \lambda_{ij})
     * \f]
     *
     * and \f$ \bar{X} \f$ the column-wise empirical mean of \f$ X \f$
     *
     */
    virtual double exp_deviance();

    /*!
     * \brief reorder factor according to the 'm_factor_order' attribute
     *
     * Reordering U, V, prob_S and S
     */
    virtual void reorder_factor();

    /*!
     * \brief create list of object to return
     *
     * \param[out] results list of returned objects
     */
    virtual void get_output(Rcpp::List &results);

protected:

    //--------------------------------------------//
    // parameter updates for standard variational //
    //--------------------------------------------//


    /*!
     * \brief compute partial deviance using a sub-set of factors (among 1...K)
     *
     * \param[in] factor integer vector of size 'K' giving the sub-set of
     * 'k' factors to consider to compute the deviance in first 'k' positions.
     * \param[in] k integer, sub-dimension to consider (<='K').
     */
    virtual double partial_deviance(const vector<int> &factor, int k);

    /*!
     * \brief update rule for Poisson intensity matrix
     *
     * \f$ \Lambda = U (S * V)^t \f$
     *
     * \f$ S * V \f$ is element-wise product
     */
    virtual void update_poisson_param();

    /*!
     * \brief update rule for the multinomial parameters in variational framework
     *
     * Compute:
     *     * \f$ \sum_i E_q[Z_{ijk}] = \sum_i p_{jk}^s * X_{ij} r_{ijk} \f$
     *     * \f$ \sum_j E_q[Z_{ijk}] = \sum_j p_{jk}^s * X_{ij} r_{ijk} \f$
     * where $\f r_{ijk} = \frac{S_{jk} * exp(E_q[log(U_{ik}) + log(V_{jk})])}{\sum_l S_{kl} * exp(E_q[log(U_{il}) + log(V_{jl})])} \f$
     *
     * Resilient to underflowing and overflowing in exponential
     */
    virtual void update_variational_multinomial_param();

    /*!
     * \brief intemrediate computation when updating the multinomial parameters
     * in variational framework
     *
     * Compute: \f$ \sum_k S_{jk} exp(E_q[log(U_{ik})] + E_q[log(V_{jk})]) \f$
     *
     * Resilient to underflowing and overflowing in exponential
     */
    virtual void intermediate_update_variational_multinomial_param();

    /*!
     * \brief update rule for variational Gamma parameter in variational framework
     */
    virtual void update_variational_gamma_param();

    /*!
     * \brief update rule for variational parameter from sparsity compartment
     */
    virtual void update_variational_sparse_param();

    /*!
     * \brief update rule for prior parameter from sparsity compartment
     */
    virtual void update_prior_sparse_param();
};

}

#endif
