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
* \brief definitions of the template 'simple_factor_algo' class
* \author Ghislain Durif
* \version 1.0
* \date 26/02/2018
*/

#ifndef VARIATIONAL_EM_H
#define VARIATIONAL_EM_H

#include <RcppEigen.h>

#include "algorithm.h"
#include "utils/random.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::VectorXd;                  // variable size vector, double precision

namespace pCMF {

/*!
* \brief template class definition for a generic variational EM inference algorithm
* \tparam model the statistical model considered for the inference framework
*/
template <typename model>
class variational_em_algo : public algorithm<model> {

public:

    /*!
     * \brief Constructor for the class `variational_em_algo`
     * \tparam model the statistical model considered for the inference framework
     *
     * \param[in] n number of observation in the data matrix (rows)
     * \param[in] p number of recorded variables in the data matrix (columns)
     * \param[in] K dimension of the latent subspace to consider
     * \param[in] X count data matrix of dimension n x p
     *
     * \param[in] verbose boolean indicating verbosity in the output
     * \param[in] monitor boolean indicating if log-likelihood, elbo, deviance, etc. should be monitored during optimization
     * \param[in] iter_max maximum number of iterations
     * \param[in] iter_min minimum number of iterations
     * \param[in] epsilon precision value to assess convergence
     * \param[in] additional_iter number of iterations where model parameter values are stable to confirm convergence
     * \param[in] conv_mode how to assess convergence: 0 for absolute gap and 1 for normalized gap between two iterates
     */
    variational_em_algo(int n, int p, int K, const MatrixXd &X,
                        bool verbose, bool monitor, int iter_max, int iter_min,
                        double epsilon, int additional_iter, int conv_mode)
        : algorithm<model>(n, p, K, X, verbose, monitor, iter_max, iter_min,
                           epsilon, additional_iter, conv_mode) {};

    /*!
    * \brief Destructor for the class `variational_em_algo`
    */
    ~variational_em_algo() {};

    /*!
     * \brief Initialization of the model parameters with given values
     *
     * \param[in] U initial values for the factor matrix U
     * \param[in] V initial values for the factor matrix V
     */
    virtual void init(const MatrixXd &U, const MatrixXd &V) {
        this->m_model.init_from_factor(U, V);
    };

    /*!
     * \brief Initialize variational parameters in Gamma Poisson
     * factor model with given values
     *
     * \param[in] a1 matrix n x K, intial values for the first parameter (shape) of Gamma variational distribution on U
     * \param[in] a2 matrix n x K, intial values for the second parameter (rate) of Gamma variational distribution on U
     * \param[in] b1 matrix p x K, intial values for the first parameter (shape) of Gamma variational distribution on V
     * \param[in] b2 matrix p x K, intial values for the second parameter (rate) of Gamma variational distribution on V
     */
    void init_variational_param_gap(const MatrixXd &a1, const MatrixXd &a2,
                                    const MatrixXd &b1, const MatrixXd &b2) {
        this->m_model.init_variational_param(a1, a2, b1, b2);
    }

    /*!
     * \brief Initialize hyper parameters in Gamma Poisson
     * factor model with given values
     *
     * \param[in] alpha1 matrix n x K, intial values for the first parameter (shape) of Gamma prior distribution on U
     * \param[in] alpha2 matrix n x K, intial values for the second parameter (rate) of Gamma prior distribution on U
     * \param[in] beta1 matrix p x K, intial values for the first parameter (shape) of Gamma prior distribution on V
     * \param[in] beta2 matrix p x K, intial values for the second parameter (rate) of Gamma prior distribution on V
     */
    void init_hyper_param_gap(const MatrixXd &alpha1, const MatrixXd &alpha2,
                              const MatrixXd &beta1, const MatrixXd &beta2) {
        this->m_model.init_hyper_param(alpha1, alpha2, beta1, beta2);
    }

    /*!
     * \brief Initialize variational and hyper-parameters in Gamma Poisson
     * factor model with given values
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
    void init_all_param_gap(const MatrixXd &alpha1, const MatrixXd &alpha2,
                        const MatrixXd &beta1, const MatrixXd &beta2,
                        const MatrixXd &a1, const MatrixXd &a2,
                        const MatrixXd &b1, const MatrixXd &b2) {
        this->m_model.init_all_param(alpha1, alpha2, beta1, beta2,
                                     a1, a2, b1, b2);
    };

    /*!
     * \brief Initialize sparsity-related variational and hyper-parameters in
     * sparse Gamma Poisson factor model
     *
     * \param[in] sel_bound real value in [0,1] used to threshold sparsity
     * probabilities for factor V
     * \param[in,out] rng Boost random number generator
     */
    virtual void init_sparse_param_gap(double sel_bound, myRandom::RNGType &rng) {
        this->m_model.init_sparse_param(sel_bound, rng);
    };

    /*!
     * \brief Initialize sparsity-related variational and hyper-parameters in
     * sparse Gamma Poisson factor model with given values
     *
     * \param[in] sel_bound real value in [0,1] used to threshold sparsity
     * probabilities for factor V
     * \param[in] prob_S matrix of dimension p x K to intialize attribute
     * m_prob_S (variational probabilities over S).
     * \param[in] prior_S vector of length p to intialize attribute
     * m_prior_prob_S (prior probabilities over S).
     */
    virtual void init_sparse_param_gap(double sel_bound,
                                       const MatrixXd &prob_S,
                                       const VectorXd &prior_S) {
        this->m_model.init_sparse_param(sel_bound, prob_S, prior_S);
    };

    /*!
     * \brief Initialize zero-inflation related variational and hyper-parameters
     * in Gamma Poisson factor model
     */
    virtual void init_zi_param_gap() {
        this->m_model.init_zi_param();
    };

    /*!
     * \brief Initialize zero-inflation related variational and hyper-parameters
     * in Gamma Poisson factor model with given values
     *
     * \param[in] prob_D matrix of dimension n x p to intialize attribute
     * m_prob_D (variational probabilities over D).
     * \param[in] prior_D vector of length p to intialize attribute
     * m_prior_prob_D (prior probabilities over D).
     */
    virtual void init_zi_param_gap(const MatrixXd &prob_D,
                                   const VectorXd &prior_D) {
        this->m_model.init_zi_param(prob_D, prior_D);
    };

    /*!
     * \brief order factor according to the deviance criterion
     */
    virtual void order_factor() {
        this->m_model.order_factor();
        this->m_model.reorder_factor();
    };

    // define getters
    /*!
     * \brief getter for U and V
     *
     * \param[out] U stores the current value of attribute U
     * \param[out] V stores the current value of attribute V
     */
    void get_factor(MatrixXd &U, MatrixXd &V) {
        this->m_model.get_factor(U, V);
    };

};

}

#endif // VARIATIONAL_EM_H
