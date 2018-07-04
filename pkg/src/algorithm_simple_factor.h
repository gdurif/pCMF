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

#ifndef SIMPLE_ALGORITHM_H
#define SIMPLE_ALGORITHM_H

#include <RcppEigen.h>

#include "algorithm.h"
#include "utils/random.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::VectorXd;                  // variable size vector, double precision

namespace pCMF {

/*!
* \brief template class definition for a generic optimization/inference algorithm
* \tparam model the statistical model considered for the inference framework
*/
template <typename model>
class simple_factor_algo : public algorithm<model> {

public:

    /*!
    * \brief Constructor for the class `simple_factor_algo`
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
    simple_factor_algo(int n, int p, int K, const MatrixXd &X,
                       bool verbose, bool monitor, int iter_max, int iter_min,
                       double epsilon, int additional_iter, int conv_mode)
        : algorithm<model>(n, p, K, X, verbose, monitor, iter_max, iter_min,
                           epsilon, additional_iter, conv_mode) {};

    /*!
    * \brief Destructor for the class `algorithm`
    */
    ~simple_factor_algo() {};

    /*!
     * \brief Initialization of the model parameters with given values
     *
     * \param[in] U initial values for the factor matrix U
     * \param[in] V initial values for the factor matrix V
     */
    virtual void init(const MatrixXd &U, const MatrixXd &V) {
        this->m_model.init_model_param(U, V);
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

#endif // SIMPLE_ALGORITHM_H
