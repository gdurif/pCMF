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
 * \brief definitions of the Gamma Poisson Factor model and derivatives
 * \author Ghislain Durif
 * \version 1.0
 * \date 07/02/2018
 */

#ifndef GAP_FACTOR_MODEL_H
#define GAP_FACTOR_MODEL_H

#include <RcppEigen.h>

#include "model.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision

namespace pCMF {

/*!
 * \brief definition of the Gamma Poisson Factor model
 *
 * Model:
 *     * \f$ X_{ij} = \sum_k Z_{ijk} \f$
 *     * \f$ X | U,V \sim Poisson(U V^t) \f$
 *       equivalent to \f$ Z_{ijk} | U_{ik}, V_{jk} \sim Poisson(U_{ik} V_{jk}) \f$
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
 *
 * Sufficient statitics needed:
 *     * \f$ E_q[U_{ik}] \f$, \f$ E_q[log(U_{ik})] \f$
 *     * \f$ E_q[V_{jk}] \f$, \f$ E_q[log(V_{jk})] \f$
 *     * \f$ \sum_i E_q[Z_{ijk}] = \sum_i X_{ij} r_{ijk} \f$
 *     * \f$ \sum_j E_q[Z_{ijk}] = \sum_j X_{ij} r_{ijk} \f$
 *     * \f$ \sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})]) \f$
 *
 */
class gap_factor_model : public variational_matrix_factor_model {

protected:

    // prior hyper-parameter
    MatrixXd m_alpha1cur;      /*!< current values of first parameter (shape) of Gamma prior on U, dimension n x K */
    MatrixXd m_alpha2cur;      /*!< current values of second parameter (rate) of prior Gamma prior on U, dimension n x K */
    MatrixXd m_beta1cur;       /*!< current values of first parameter (shape) of Gamma prior on V, dimension p x K */
    MatrixXd m_beta2cur;       /*!< current values of second parameter (rate) of prior Gamma prior on V, dimension p x K */

    MatrixXd m_alpha1old;      /*!< previous values of first parameter (shape) of Gamma prior on U, dimension n x K */
    MatrixXd m_alpha2old;      /*!< previous values of second parameter (rate) of prior Gamma prior on U, dimension n x K */
    MatrixXd m_beta1old;       /*!< previous values of first parameter (shape) of Gamma prior on V, dimension p x K */
    MatrixXd m_beta2old;       /*!< previous values of second parameter (rate) of prior Gamma prior on V, dimension p x K */

    // variational parameters
    MatrixXd m_a1cur;       /*!< current values of first parameter (shape) of Gamma variational distribution on U, dimension n x K */
    MatrixXd m_a2cur;       /*!< current values of second parameter (rate) of variational distribution Gamma variational distribution on U, dimension n x K */
    MatrixXd m_b1cur;       /*!< current values of first parameter (shape) of Gamma variational distribution on V, dimension p x K */
    MatrixXd m_b2cur;       /*!< current values of second parameter (rate) of variational distribution Gamma variational distribution on V, dimension p x K */

    MatrixXd m_a1old;       /*!< previous values of first parameter (shape) of Gamma variational distribution on U, dimension n x K */
    MatrixXd m_a2old;       /*!< previous values of second parameter (rate) of variational distribution Gamma variational distribution on U, dimension n x K */
    MatrixXd m_b1old;       /*!< previous values of first parameter (shape) of Gamma variational distribution on V, dimension p x K */
    MatrixXd m_b2old;       /*!< previous values of second parameter (rate) of variational distribution Gamma variational distribution on V, dimension p x K */


    // additional sufficient statistics
    MatrixXd m_EZ_i;            /*!< \f$ \sum_i X_{ij} r_{ijk} = \sum_i E_q[Z_{ijk}] \f$, dimension p x k */
    MatrixXd m_EZ_j;            /*!< \f$ \sum_j X_{ij} r_{ijk} = \sum_j E_q[Z_{ijk}] \f$, dimension n x k */

    MatrixXd m_exp_ElogU_ElogV_k;   /*!< \f$ \sum_k exp(E_q[log(U_{ik})]) * exp(E_q[log(V_{jk})]) \f$, dimension n x p */

public:
    /*!
     * \brief Constructor for the class `gap_factor_model`
     */
    gap_factor_model(int n, int p, int K, const MatrixXd &X);

    /*!
     * \brief destructor for the class `gap_factor_model`
     */
    ~gap_factor_model();

    /*!
     * \brief Initialize variational and hyper-parameters with given values
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
     * \brief Initialize variational parameters with given values
     *
     * \param[in] a1 matrix n x K, intial values for the first parameter (shape) of Gamma variational distribution on U
     * \param[in] a2 matrix n x K, intial values for the second parameter (rate) of Gamma variational distribution on U
     * \param[in] b1 matrix p x K, intial values for the first parameter (shape) of Gamma variational distribution on V
     * \param[in] b2 matrix p x K, intial values for the second parameter (rate) of Gamma variational distribution on V
     */
    void init_variational_param(const MatrixXd &a1, const MatrixXd &a2,
                                const MatrixXd &b1, const MatrixXd &b2);

    /*!
     * \brief Initialize variational and hyper-parameters with given values
     *
     * \param[in] alpha1 matrix n x K, intial values for the first parameter (shape) of Gamma prior on U
     * \param[in] alpha2 matrix n x K, intial values for the second parameter (rate) of Gamma prior on U
     * \param[in] beta1 matrix p x K, intial values for the first parameter (shape) of Gamma prior on V
     * \param[in] beta2 matrix p x K, intial values for the second parameter (rate) of Gamma prior on V
     */
    void init_hyper_param(const MatrixXd &alpha1, const MatrixXd &alpha2,
                          const MatrixXd &beta1, const MatrixXd &beta2);

    /*!
     * \brief Initialize variational parameters with from given
     * factor matrices U and V
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
     * \brief update variational parameters values between iterations
     *
     * current values of varitional parameters become old values of parameters
     */
    virtual void prepare_next_iterate_variational_param();

    /*!
     * \brief update prior hyper-parameters values between iterations
     *
     * current values of prior hyper-parameters become old values of parameters
     */
    virtual void prepare_next_iterate_hyper_param();

    /*!
     * \brief compute absolute and normalized gap of parameters between two iterates
     *
     * \param[out] abs_gap absolute gap.
     * \param[out] norm_gap normalized gap.
     */
    virtual void gap_between_iterates(double &abs_gap, double& norm_gap);

    /*!
     * \brief compute a convergence criterion to assess convergence based on
     * the RV coefficients
     *
     * The RV coefficients measures the closeness of the two set of points stored
     * in two matrices [1]. Here, we compute:
     * f\[
     * crit = min( RV_coeff(U_{new}, U_{old}), RV_coeff(V_{new}, V_{old}) )
     * f\]
     *
     * Important: The RV coefficient custom is transformed so that
     * it converges to zero when convergence occurs. Thus, this function
     * returns f\$ 1 - crit f\$.
     *
     * \return value of the criterion
     *
     * [1] Friguet, C., 2010. Impact de la dépendance dans les procédures de tests
     * multiples en grande dimension. Rennes, AGROCAMPUS-OUEST.
     */
    virtual double custom_conv_criterion();

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
     * \log p(X | \Lambda = UV^t) + \log p(U; \alpha) + \log p(V; \beta)
     * \f]
     *
     * where \f$ \log p(X | \Lambda ) \f$ is the Poisson log-likelihood and
     * \f$ \Lambda \f$ the Poisson rate matrix, i.e
     *
     * \f[
     * \log p(X | \Lambda ) = \log p(x_{ij} | \lambda_{ij})
     * \f]
     *
     * and \f$ \log p(U; \alpha) \f$ and \f$ \log p(V; \beta) \f$ are
     * the Gamma log-likelihood for U and V respectively
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
     * deviance = -2 \times [ \log p(X | \Lambda = UV^t) - \log p(X | \Lambda = X)]
     * \f]
     *
     * where \f$ \log p(X | \Lambda ) \f$ is the Poisson log-likelihood and
     * \f$ \Lambda \f$ the Poisson rate matrix, i.e
     *
     * \f[
     * \log p(X | \Lambda ) = \log p(x_{ij} | \lambda_{ij})
     * \f]
     *
     * This is equivalent to computing the Bregman divergence
     * between \f$ X \f$ and \f$ UV^t \f$ in the Poisson framework:
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
     * %deviance = \frac{ \log p(X | \Lambda = UV^t) - \log p(X | \Lambda = \bar{X})}{ \log p(X | \Lambda = X) - \log p(X | \Lambda = \bar{X}) }
     * \f]
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
     * Default behavior is to only reorder U and V
     */
    virtual void reorder_factor();

    /*!
     * \brief create list of object to return
     *
     * \param[out] results list of returned objects
     */
    virtual void get_output(Rcpp::List &results);

    // define getters
    /*!
     * \brief getter for U and V
     *
     * \param[out] U stores the current value of attribute U
     * \param[out] V stores the current value of attribute V
     */
    virtual void get_factor(MatrixXd &U, MatrixXd &V);

protected:

    //--------------------------------------------//
    // parameter updates for standard variational //
    //--------------------------------------------//

    /*!
     * \brief Compute sufficient statistics for U regarding
     * the variational distribution
     */
    virtual void U_stats();

    /*!
     * \brief Compute sufficient statistics for V regarding
     * the variational distribution
     */
    virtual void V_stats();

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
     * \f$ \Lambda = U V^t \f$
     */
    virtual void update_poisson_param();

    /*!
     * \brief update rule for the multinomial parameters in variational framework
     *
     * Compute:
     *     * \f$ \sum_i E_q[Z_{ijk}] = \sum_i X_{ij} r_{ijk} \f$
     *     * \f$ \sum_j E_q[Z_{ijk}] = \sum_j X_{ij} r_{ijk} \f$
     * where $\f r_{ijk} = \frac{exp(E_q[log(U_{ik}) + log(V_{jk})])}{\sum_l exp(E_q[log(U_{il}) + log(V_{jl})])} \f$
     *
     * Resilient to underflowing and overflowing in exponential
     */
    virtual void update_variational_multinomial_param();

    /*!
     * \brief intemrediate computation when updating the multinomial parameters
     * in variational framework
     *
     * Compute: \f$ \sum_k exp(E_q[log(U_{ik})] + E_q[log(V_{jk})]) \f$
     *
     * Resilient to underflowing and overflowing in exponential
     */
    virtual void intermediate_update_variational_multinomial_param();

    /*!
     * \update rule for variational Gamma parameter in variational framework
     */
    virtual void update_variational_gamma_param();

    /*!
     * \update rule for prior Gamma parameter in variational framework
     */
    virtual void update_prior_gamma_param();
};

}

#endif
