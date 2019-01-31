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
 * \brief definitions of the zero-inflated Gamma Poisson Factor model
 * \author Ghislain Durif
 * \version 1.0
 * \date 10/04/2018
 */

#ifndef ZI_GAP_FACTOR_MODEL_H
#define ZI_GAP_FACTOR_MODEL_H

#include <RcppEigen.h>

#include "gap_factor_model.h"
#include "utils/random.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::VectorXd;                  // variable size vector, double precision

namespace pCMF {

/*!
 * \brief definition of the Zero-Inflated Gamma Poisson Factor model
 *
 * Zero-inflation concerns the conditional distribution over the data X
 *
 * Model:
 *     * \f$ X_{ij} = \sum_k Z_{ijk} \f$
 *     * \f$ X_{ij} | (U_{ik},V_{jk})_k,D_{ij} \sim (1-D_{ij}) delta_0(X_{ij}) + D_{ij} * Poisson(U V^t) \f$
 *       where \f$ D_{ij} \sim Bernoulli(pi_j^d) \f$ is the drop-out indicator for \f$ X_{ij} \f$
 *       ( \f$ D_{ij} = 0 \f$ corresponds to a drop-out event)
 *       and \f$ pi_j^d \f$ the drop-out probability for gene \f$ j \f$,
 *       which corresponds to the zero-inflated distribution:
 *       \f[
 *       X_{ij} | (U_{ik},V_'jk})_k \sim (1-pi_{j}^d) delta_0(X_{ij}) + pi_{j}^d * Poisson(U V^t) \f$
 *       \f]
 *       thus \f$ Z_{ijk} | U_{ik}, V_{jk}, D_{ij} \sim (1-D_{ij}) delta_0(Z_{ijk}) + D_{ij} * Poisson(U_{ik} V_{jk}) \f$
 *     * \f$ U_{ik} \sim Gamma(\alpha_{ik}) \f$ with \f$ \alpha_{ik} = (\alpha_{ik,1}, \alpha_{ik,2}) \f$
 *     * \f$ V_{jk} \sim Gamma(\beta_{jk}) \f$ with \f$ \beta_{jk} = (\beta_{jk,1}, \beta_{jk,2}) \f$
 *       Note:
 *           - In our approach, all \f$ \alpha_{ik} \f$ are identical across i.
 *           - Similarly, all \f$ \beta_{jk} \f$ are identical across j.
 *     * Consequence: \f$ (Z_{ijk})_k \sim Multinomial((\rho_{ijk})_k)
 *       where \f$ \rho_{ijk} = \frac{U_{ik} V_{jk}}{\sum_l U_{il} V_{jl}} \f$
 *
 * Variational distribution \f$ q \f$:
 *     * \f$ (Z_{ijk})_k \sim_q Multinomial((r_{ijk})_k)
 *     * \f$ U_{ik} \sim_q Gamma(a_{ik}) \f$ with \f$ a_{ik} = (a_{ik,1}, a_{ik,2}) \f$
 *     * \f$ V_{jk} \sim_q Gamma(b_{jk}) \f$ with \f$ b_{jk} = (b_{jk,1}, b_{jk,2}) \f$
 *     * \f$ D_{ij} \sim_q Bernoulli(p_{ij}^d) \f$
 *
 * Sufficient statitics needed:
 *     * \f$ E_q[U_{ik}] \f$, \f$ E_q[log(U_{ik})] \f$
 *     * \f$ E_q[V_{jk}] \f$, \f$ E_q[log(V_{jk})] \f$
 *     * \f$ \sum_i E_q[D_{ij}] E_q[Z_{ijk}] = \sum_i p_{ij}^d * X_{ij} r_{ijk} \f$
 *     * \f$ \sum_j E_q[D_{ij}] E_q[Z_{ijk}] = \sum_j p_{ij}^d * X_{ij} r_{ijk} \f$
 *     * \f$ \sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})]) \f$
 *
 */
class zi_gap_factor_model : public gap_factor_model {

protected:

    // ZI compartment
    MatrixXd m_prob_D;          /*!< matrix of probability for variational distribution over drop-out indicator D, dimension n x p */
    VectorXd m_prior_prob_D;    /*!< vector of probability for prior distribution over drop-out indicator D, length p */
    VectorXd m_freq_D;          /*!< vector of frequence of non null values in each column of X, length p */

public:
    /*!
     * \brief Constructor for the class `zi_gap_factor_model`
     */
    zi_gap_factor_model(int n, int p, int K, const MatrixXd &X);

    /*!
     * \brief destructor for the class `zi_gap_factor_model`
     */
    ~zi_gap_factor_model();

    /*!
     * \brief Initialize variational and hyper-parameters from Gamma compartment
     * with given values
     *
     * \param[in] alpha1 matrix n x K, intial values for the first parameter (shape) of Gamma prior on U
     * \param[in] alpha2 matrix n x K, intial values for the second parameter (rate) of Gamma prior on U
     * \param[in] beta1 matrix p x K, intial values for the first parameter (shape) of Gamma prior on V
     * \param[in] beta2 matrix p x K, intial values for the second parameter (rate) of Gamma prior on V
     * \param[in] a1 matrix n x K, intial values for the first parameter (shape) of Gamma variational distribution on U
     * \param[in] a2 matrix n x K, intial values for the second parameter (rate) of Gamma variational distribution on U
     * \param[in] b1 matrix p x K, intial values for the first parameter (shape) of Gamma variational distribution on V
     * \param[in] b2 matrix p x K, intial values for the second parameter (rate) of Gamma variational distribution on V
     */
    void init_all_param(const MatrixXd &alpha1, const MatrixXd &alpha2,
                        const MatrixXd &beta1, const MatrixXd &beta2,
                        const MatrixXd &a1, const MatrixXd &a2,
                        const MatrixXd &b1, const MatrixXd &b2);

    /*!
     * \brief Initialize variational parameters from Gamma compartment with given values
     *
     * \param[in] a1 matrix n x K, intial values for the first parameter (shape) of Gamma variational distribution on U
     * \param[in] a2 matrix n x K, intial values for the second parameter (rate) of Gamma variational distribution on U
     * \param[in] b1 matrix p x K, intial values for the first parameter (shape) of Gamma variational distribution on V
     * \param[in] b2 matrix p x K, intial values for the second parameter (rate) of Gamma variational distribution on V
     */
    void init_variational_param(const MatrixXd &a1, const MatrixXd &a2,
                                const MatrixXd &b1, const MatrixXd &b2);

    /*!
     * \brief Initialize hyper-parameters from Gamma compartment with given values
     *
     * \param[in] alpha1 matrix n x K, intial values for the first parameter (shape) of Gamma prior on U
     * \param[in] alpha2 matrix n x K, intial values for the second parameter (rate) of Gamma prior on U
     * \param[in] beta1 matrix p x K, intial values for the first parameter (shape) of Gamma prior on V
     * \param[in] beta2 matrix p x K, intial values for the second parameter (rate) of Gamma prior on V
     */
    void init_hyper_param(const MatrixXd &alpha1, const MatrixXd &alpha2,
                          const MatrixXd &beta1, const MatrixXd &beta2);

    /*!
     * \brief Initialize variational parameters from given
     * factor matrices U and V for Gamma compartment
     *
     * U is used to initialize the a1 shape variational parameter. The
     * a2 rate variational parameter is set to 1, such that a1/a2 = U.
     * V is used to initialize the b1 shape variational parameter. The
     * b2 rate variational parameter is set to 1, such that b1/b2 = V.
     *
     * \param[in] U matrix n x K, initial values for U
     * \param[in] V matrix p x K, initial values for V
     */
    void init_from_factor(const MatrixXd &U, const MatrixXd &V);

    /*!
     * \brief Initialize variational and hyper-parameters with random values
     *
     * Each row \code{i} in \code{a1} is
     * randomly initialized from a Gamma distribution of parameters
     * \code{(1,sqrt(K/mean(X)_i))} where \code{mean(X)_i} is the rowwise mean of
     * the corresponding row in the input data matrix \code{X}.
     *
     * Each row \code{j} in \code{b1} is
     * randomly initialized from a Gamma distribution of parameters
     * \code{(1,sqrt(K/mean(X)_j))} where \code{mean(X)_j} is the colwise mean of
     * the corresponding column in the input data matrix \code{X}.
     *
     * \param[in,out] rng Boost random number generator
     */
    virtual void random_init_model_param(myRandom::RNGType &rng);

    /*!
     * \brief initialize variational and hyper-parameter from ZI compartment
     *
     * Prior probabilities over D i.e. \f$ (pi_j^d)_j \f$ are initialized with
     * frequences of non null values in the corresponding column of X
     *
     * Variational probabilities over D i.e. \f$ (p_{ij}^d)_j \f$ are
     * initialized with frequences of non null values in the corresponding
     * column of X
     */
    virtual void init_zi_param();

    /*!
     * \brief initialize variational and hyper-parameter from ZI compartment
     *
     * Prior probabilities over D i.e. \f$ (pi_j^d)_j \f$ and variational
     * probabilities over D i.e. \f$ (p_{ij}^d)_j \f$ are
     * initialized with given input parameter values.
     *
     * \param[in] prob_D matrix of dimension n x p to intialize attribute
     * m_prob_D (variational probabilities over D).
     * \param[in] prior_D vector of length p to intialize attribute
     * m_prior_prob_D (prior probabilities over D).
     */
    virtual void init_zi_param(const MatrixXd &prob_D,
                               const VectorXd &prior_D);

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
     * \log p(X | \Lambda = UV^t, (p_j^d)_j) + \log p(U; \alpha) + \log p(V; \beta)
     * + \log p(D; (p_j^d)_j)
     * \f]
     *
     * where \f$ \log p(X | \Lambda, (p_j^d)_j) \f$ is the zero-inflated Poisson
     * log-likelihood and \f$ \Lambda \f$ the Poisson rate matrix, i.e
     *
     * \f[
     * \log p(X | \Lambda, (p_j^d)_j ) = \log p(x_{ij} | \lambda_{ij}, p_j^d)
     * \f]
     *
     * \f$ \log p(U; \alpha) \f$ and \f$ \log p(V; \beta) \f$ are
     * the Gamma log-likelihood for U and V respectively
     *
     * and \f$ \log p(D; (p_j^d)_j) \f$ the Bernoulli log-likelihood
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
     * deviance = -2 \times [ \log p(X | \Lambda = D * UV^t) - \log p(X | \Lambda = X)]
     * \f]
     *
     * where \f$ \log p(X | \Lambda ) \f$ is the Poisson log-likelihood and
     * \f$ \Lambda \f$ the Poisson rate matrix, i.e
     *
     * \f[
     * \log p(X | \Lambda ) = \log p(x_{ij} | \lambda_{ij})
     * \f]
     *
     * and \f$ D * UV^t \f$ is the element-wise product between \f$ D \f$ and \f$ UV^t f$.
     *
     * This formulation is based on the reformulation of the zero-inflated
     * Poisson distribution as \f$ X \sim Poisson(D * UV^t) \f$ because Poisson(0) degenerates
     * into a Dirac mass at 0.
     *
     * This is equivalent to computing the Bregman divergence
     * between \f$ X \f$ and \f$ D * UV^t \f$ in the Poisson framework:
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
     * %deviance = \frac{ \log p(X | \Lambda = D * UV^t) - \log p(X | \Lambda = \bar{X})}{ \log p(X | \Lambda = X) - \log p(X | \Lambda = \bar{X}) }
     * \f]
     *
     * (c.f. deviance doc for \f$ \log p(X | \Lambda = D * UV^t) \f$ definition)
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
     * \brief update rule for the multinomial parameters in variational framework
     *
     * Compute:
     *     * \f$ \sum_i E_q[Z_{ijk}] = \sum_i p_{ij}^d * X_{ij} r_{ijk} \f$
     *     * \f$ \sum_j E_q[Z_{ijk}] = \sum_j p_{ij}^d * X_{ij} r_{ijk} \f$
     * where $\f r_{ijk} = \frac{exp(E_q[log(U_{ik}) + log(V_{jk})])}{\sum_l exp(E_q[log(U_{il}) + log(V_{jl})])} \f$
     *
     * Resilient to underflowing and overflowing in exponential
     */
    virtual void update_variational_multinomial_param();

    /*!
     * \brief update rule for variational Gamma parameter in variational framework
     */
    virtual void update_variational_gamma_param();

    /*!
     * \brief update rule for variational parameter from ZI compartment
     */
    virtual void update_variational_zi_param();

    /*!
     * \brief update rule for prior parameter from ZI compartment
     */
    virtual void update_prior_zi_param();
};

}

#endif
