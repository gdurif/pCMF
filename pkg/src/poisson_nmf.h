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
* \brief definitions of the 'poisson_nmf' class (Non-negative Matrix Factorization in the Poisson setting)
* \author Ghislain Durif
* \version 1.0
* \date 21/02/2018
*/

#ifndef POISSON_NMF_H
#define POISSON_NMF_H

#include <RcppEigen.h>

#include "model.h"
#include "utils/random.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision

namespace pCMF {

/*!
 * \brief class definition for a non-negative matrix factorization model in the Poisson setting
 *
 * Defines a Non-negative Matrix Factorization for count matrices,
 * based on a Poisson model over the count, i.e.
 * f\[
 * X \sim P(UV^t)
 * f\]
 *
 * Optimization process follows what was proposed by [1]
 * and implemented by [2]: minimization of the Bregman divergence in the
 * Poisson setting between f\$ X f\$ and \f$ UV^t f\$, which happens to be
 * the Kullback-Leibler divergence between f\$ X f\$ and \f$ UV^t f\$,
 * which is equivalent to minimizing the deviance of the model previously defined.
 *
 * [1] Brunet, J.-P., Tamayo, P., Golub, T.R., Mesirov, J.P., 2004.
 * Metagenes and molecular pattern discovery using matrix factorization.
 * PNAS 101, 4164–4169.
 *
 * [2] Gaujoux, R., Seoighe, C., 2010.
 * A flexible R package for nonnegative matrix factorization.
 * BMC Bioinformatics 11, 367.
 *
 */
class poisson_nmf : public matrix_factor_model {

protected:
    MatrixXd m_oldU;       /*!< previous values for latent components, dim n x K (representation of individuals) */
    MatrixXd m_oldV;       /*!< previous values for latent loadings, dim p x K (contribution of recorded variables) */

public:

    /*!
     * \brief constructor for the class `poisson_nmf`
     */
    poisson_nmf(int n, int p, int K, const MatrixXd &X);

    /*!
     * \brief randomly perturb parameters
     *
     * \param[in,out] rng random number generator
     * \param[in] noise_level level of the perturbation, based on a uniform
     * distribution on [-noise_level, noise_level]
     */
    virtual void perturb_param(myRandom::RNGType &rng, double noise_level);

    /*!
     * \brief update rules for parameters in the optimization process
     *
     * Update U and V as proposed in [1]
     *
     * f\[
     * U_{i,k} \gets U_{i,k} \times \frac{\sum_j V_{j,k} X_{i,j} / (UV^t)_{i,j}}{\sum_j V_{j,k}}
     * f\]
     *
     * f\[
     * V_{jk} \gets V_{j,k} \times \frac{\sum_i U_{i,k} X_{i,j} / (UV^t)_{i,j}}{\sum_i U_{i,k}}
     * f\]
     *
     * [1] Brunet, J.-P., Tamayo, P., Golub, T.R., Mesirov, J.P., 2004.
     * Metagenes and molecular pattern discovery using matrix factorization.
     * PNAS 101, 4164–4169.
     *
     */
    virtual void update_param();

    /*!
     * \brief update parameters values between iterations
     *
     * current values of parameters become old values of parameters
     */
    virtual void prepare_next_iterate();

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
     * \brief compute the optimization criterion associated to the Poisson NMF model
     *
     * The loss is the Bregman divergence between \f$ X \f$ and \f$ UV^t \f$
     * in the Poisson framework:
     *
     * \f[
     * d(X,Y) = \sum_{i,j} d(x_{i,j}, y_{i,j})
     * \f]
     *
     * with \f$ d(x,y) = x \log\frac{x}{y} - x + y \f$
     *
     * This is equivalent to optimizing the deviance of the Poisson model, i.e.
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
     */
    virtual double optim_criterion();

    /*!
     * \brief compute the Poisson log-likelihood associated to the Poisson NMF model
     *
     * \f[
     * \log p(X | \Lambda = UV^t)
     * \f]
     *
     * where \f$ \log p(X | \Lambda ) \f$ is the Poisson log-likelihood and
     * \f$ \Lambda \f$ the Poisson rate matrix, i.e
     *
     * \f[
     * \log p(X | \Lambda ) = \log p(x_{ij} | \lambda_{ij})
     * \f]
     */
    virtual double loglikelihood();

    /*!
     * \brief compute the deviance associated to the Poisson NMF model
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
     * to the Poisson NMF model
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

protected:

    /*!
     * \brief compute partial deviance using a sub-set of factors (among 1...K)
     *
     * \param[in] factor integer vector of size 'K' giving the sub-set of
     * 'k' factors to consider to compute the deviance in first 'k' positions.
     * \param[in] k integer, sub-dimension to consider (<='K').
     */
    virtual double partial_deviance(const vector<int> &factor, int k);

};

}

#endif // POISSON_NMF_H
