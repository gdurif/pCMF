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
 * \brief definitions of the 'model' and 'matrix_factorization' classes
 * \author Ghislain Durif
 * \version 1.0
 * \date 07/02/2018
 */

#ifndef MODEL_H
#define MODEL_H

#include <RcppEigen.h>
#include <vector>

#include "utils/random.h"

using std::vector;

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::Map;                       // 'maps' rather than copies
using Eigen::VectorXi;                  // variable size vector, integer

/*!
 * \namespace pCMF
 *
 * A specific namespace for all classes and functions related to the pCMF package
 */
namespace pCMF {

/*!
 * \brief class definition for a generic statistical model in the variational framework
 */
class model {

public:
    /*!
     * \brief Initialize variational and hyper-parameters with random values
     *
     * \param[in,out] rng Boost random number generator
     */
    virtual void init_model_param(myRandom::RNGType &rng);

    /*!
     * \brief update rules for parameters in the optimization process
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
     * \brief compute a custom (model dependent) convergence criterion to assess convergence
     *
     * Important: This custom criterion should be transformed if necessary
     * so that it converges to zero when convergence occurs.
     *
     * Note: by default, this method raises an error if the user asks that
     * convergence is assessed based on this custom criterion when using
     * a method/model for which there is no custom criterion defined.
     *
     * \return value of the criterion
     */
    virtual double custom_conv_criterion();

    /*!
     * \brief compute the joint (or complete) log-likelihood of the model
     */
    virtual double loglikelihood() = 0;

    /*!
     * \brief compute the deviance associated to the model
     */
    virtual double deviance() = 0;

    /*!
     * \brief compute the percentage of explained deviance associated to the model
     */
    virtual double exp_deviance() = 0;

    /*!
     * \brief compute the evidence lower bound (ELBO) for the model (if defined)
     *
     * The default behavior is to return 0 for cases when ELBO is not defined.
     */
    virtual double elbo();

    /*!
     * \brief create list of object to return
     *
     * \param[out] results list of returned objects
     */
    virtual void get_output(Rcpp::List &results) = 0;

};


/*!
 * \brief definition of shared attributes and member functions for a matrix factorization procedure
 *
 * Data:
 *     * X: count data matrix of dimension n x p
 *     * n: number of observations in the sample
 *     * p: number of recorded variables
 *
 * Matrix factorization:
 *     * K: dimension of the latent subspace
 *     * U: representation of the individuals in the latent subspace, called components, dimension n x K
 *     * V: contribution of the variables to the representation in the latent subspace, called loadings, dimension p x K
 *
 */
class matrix_factorization {

protected:
    MatrixXd m_X;       /*!< count data matrix, dim n x p */
    int m_n;            /*!< number of observations (rows) */
    int m_p;            /*!< number of recorded variables (columns) */

    int m_K;            /*!< dimension of the latent subspace */
    MatrixXd m_U;       /*!< latent components, dim n x K (representation of individuals) */
    MatrixXd m_V;       /*!< latent loadings, dim p x K (contribution of recorded variables) */
    MatrixXd m_UVt;     /*!< store the product \f$ U V^t \f$, dim n x p (reconstruction of X) */

public:
    /*!
     * \brief Constructor for the class `matrix_factorization`
     *
     * \param[in] n number of observation in the data matrix (rows)
     * \param[in] p number of recorded variables in the data matrix (columns)
     * \param[in] K dimension of the latent subspace to consider
     * \param[in] X count data matrix of dimension n x p
     *
     */
    matrix_factorization(int n, int p, int K, const MatrixXd &X);

    /*!
     * \brief Destructor for the class `matrix_factorization`
     */
    ~matrix_factorization();

    /*!
     * \brief compute the value of the loss associated to the model
     */
    virtual double optim_criterion() = 0;

    // define getters
    /*!
     * \brief getter for U and V
     *
     * \param[out] U stores the current value of attribute U
     * \param[out] V stores the current value of attribute V
     */
    virtual void get_factor(MatrixXd &U, MatrixXd &V);

};


/*!
 * \brief class definition for a generic matrix factorization model
 */
class matrix_factor_model : public matrix_factorization, public model {

protected:
    VectorXi m_factor_order;      /*!< order of the factor according to the deviance criterion */

public:

    /*!
    * \brief constructor for the class `matrix_factor_model`
    */
    matrix_factor_model(int n, int p, int K, const MatrixXd &X);

    /*!
     * \brief Initialize variational and hyper-parameters with random values
     *
     * \param[in,out] rng Boost random number generator
     */
    virtual void random_init_model_param(myRandom::RNGType &rng);

    /*!
     * \brief Initialize model parameter with given values
     *
     * \param[in] U matrix n x K, initial values for U
     * \param[in] V matrix p x K, initial values for V
     */
    virtual void init_model_param(const MatrixXd &U, const MatrixXd &V);

    /*!
     * \brief randomly perturb parameters
     *
     * \param[in,out] rng random number generator
     * \param[in] noise_level level of the perturbation, based on a uniform
     * distribution on [-noise_level, noise_level]
     */
    virtual void perturb_param(myRandom::RNGType &rng, double noise_level) = 0;

    /*!
    * \brief update rules for parameters in the optimization process
    */
    virtual void update_param();

    /*!
    * \brief update parameters values between iterations
    *
    * current values of parameters become old values of parameters
    */
    virtual void prepare_next_iterate();

    /*!
     * \brief order factor according to the deviance criterion
     *
     * Compute best factor order and update the 'm_factor_order' attribute
     *
     * Note: this method does not modify the factor order, it is necessary to
     * call the 'reorder_factor' method to effectively reorder factors
     * according to the 'm_factor_order' attribute
     */
    virtual void order_factor();

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
    void get_output(Rcpp::List &results);

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



/*!
 * \brief class definition for a generic matrix factorization model in the variational framework
 */
class variational_matrix_factor_model : public matrix_factor_model {

protected:
    // sufficient statistics regarding the variational distribution q
    MatrixXd m_EU;              /*!< Expectation of U regarding the variational distribution q, dimension n x K */
    MatrixXd m_ElogU;           /*!< Expectation of log U regarding the variational distribution q, dimension n x K */

    MatrixXd m_EV;              /*!< Expectation of V regarding the variational distribution q, dimension p x K */
    MatrixXd m_ElogV;           /*!< Expectation of log V regarding the variational distribution q, dimension p x K */

public:

    /*!
    * \brief constructor for the class `variational_matrix_factor_model`
    */
    variational_matrix_factor_model(int n, int p, int K, const MatrixXd &X);

    /*!
     * \brief initialize variational and hyper parameters in sparse model
     *
     * \param[in] sel_bound real value in [0,1] used to threshold sparsity
     * probabilities for factor V
     * \param[in,out] rng Boost random number generator
     */
    virtual void init_sparse_param(double sel_bound, myRandom::RNGType &rng);

    /*!
    * \brief update rules for parameters in the optimization process
    */
    virtual void update_param();

    /*!
    * \brief update parameters values between iterations
    *
    * current values of parameters become old values of parameters
    */
    virtual void prepare_next_iterate();

protected:

    /*!
     * \brief update rule for variational parameters in the optimization process
     */
    virtual void update_variational_param();

    /*!
     * \brief update rules for prior hyper-parameters in the optimization process
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

};


}

#endif // MODEL_H
