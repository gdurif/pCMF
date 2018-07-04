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
 * \brief implementation of the 'model' and 'matrix_factorization' classes
 * \author Ghislain Durif
 * \version 1.0
 * \date 07/02/2018
 */

#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>

#include <stdio.h>

#include "model.h"
#include "utils/random.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::Map;                       // 'maps' rather than copies
using Eigen::PermutationMatrix;         // variable size permutation matrix
using Eigen::VectorXi;                  // variable size vector, integer

using std::vector;

namespace pCMF {

// #############################################################################
// `model` class

// Initialize variational and hyper-parameters with given values
void model::init_model_param(myRandom::RNGType &rng) {}

// update rules for parameters in the optimization process
void model::update_param() {}

// update parameters values between iterations
void model::prepare_next_iterate() {}

// compute absolute and normalized gap of parameters between two iterates
void model::gap_between_iterates(double &abs_gap, double& norm_gap) {}

// compute a custom (model dependent) convergence criterion to assess convergence
double model::custom_conv_criterion() {
    Rcpp::stop("You cannot use a custom convergence criterion with this method/model, i.e. the 'conv_mode' option should be 0 or 1 and NOT 2");
    return(0);
}

// compute the evidence lower bound (ELBO) for the model (if defined)
double model::elbo() {
    return(0);
}


// #############################################################################
// `matrix_factorization` class

// Constructor for the class `matrix_factorization`
matrix_factorization::matrix_factorization(int n, int p, int K, const MatrixXd &X) {
    // dimension
    m_n = n;
    m_p = p;
    m_K = K;

    // data
    m_X = MatrixXd(X);

    // latent subspace
    m_U = MatrixXd::Ones(n,K);
    m_V = MatrixXd::Ones(p,K);

    // reconstruction
    m_UVt = MatrixXd::Ones(n,p);
}

// Destructor for the class `matrix_factorization`
matrix_factorization::~matrix_factorization() {}

// define getters
// getter for U and V
void matrix_factorization::get_factor(MatrixXd &U, MatrixXd &V) {
    U = m_U;
    V = m_V;
}



// #############################################################################
// `matrix_factor_model` class

// constructor for the class `matrix_factor_model`
matrix_factor_model::matrix_factor_model(int n, int p, int K, const MatrixXd &X)
    : matrix_factorization(n, p, K, X) {
    m_factor_order = VectorXi::LinSpaced(K,0,K-1);
}

// Initialize model parameter with random values
void matrix_factor_model::random_init_model_param(myRandom::RNGType &rng) {
    myRandom::rUnif(m_U, m_n, m_K, 0, 1, rng);
    myRandom::rUnif(m_V, m_p, m_K, 0, 1, rng);
    // compute UV^t
    m_UVt = m_U * m_V.transpose();
}

// Initialize model parameter with given values
void matrix_factor_model::init_model_param(const MatrixXd &U, const MatrixXd &V) {
    m_U = U;
    m_V = V;
    // compute UV^t
    m_UVt = m_U * m_V.transpose();
}

// update rules for parameters in the optimization process
void matrix_factor_model::update_param() {}

// update parameters values between iterations
void matrix_factor_model::prepare_next_iterate() {}

// order factor according to the deviance criterion
void matrix_factor_model::order_factor() {
    vector<int> left_index;
    vector<int> chosen_index;
    left_index.reserve(m_K);
    chosen_index.reserve(m_K);

    // init
    for(int k=0; k<m_K; k++) {
        left_index.push_back(k);
    }

    // ordering
    for(int k=0; k<m_K; k++) {
        double val_min = -1;
        int ind_min = 0;
        int ind_left_index = 0;
        // check all factor among the ones that are left
        for(int ind=0; ind<left_index.size(); ind++) {

            vector<int> tmp_index(chosen_index);
            vector<int> tmp_left(left_index);
            tmp_index.push_back(left_index[ind]);
            tmp_left.erase(tmp_left.begin() + ind);
            tmp_index.insert(tmp_index.end(), tmp_left.begin(), tmp_left.end());

            double res = this->partial_deviance(tmp_index, k+1);

            // init values with first left index
            if(ind==0) {
                val_min = res;
                ind_min = left_index[ind];
                ind_left_index = ind;
            }

            // check for the min
            if(res < val_min) {
                val_min = res;
                ind_min = left_index[ind];
                ind_left_index = ind;
            }
        }
        // min found
        chosen_index.push_back(ind_min);
        left_index.erase(left_index.begin() + ind_left_index);
    }

    // return
    for(int k=0; k<m_K; k++) {
        m_factor_order(k) = chosen_index[k];
    }
}

// reorder factor according to the 'm_factor_order' attribute
void matrix_factor_model::reorder_factor() {

    // permutation based on 'm_factor_order'
    PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(m_K);
    for(int k=0; k<m_K; k++) {
        perm.indices()[k] = m_factor_order[k];
    }
    m_U = m_U * perm;
    m_V = m_V * perm;
}

// create list of object to return
void matrix_factor_model::get_output(Rcpp::List &results) {
    Rcpp::List output = Rcpp::List::create(Rcpp::Named("U") = m_U,
                                           Rcpp::Named("V") = m_V);
    results.push_back(output, "factor");
}

// compute partial deviance using a sub-set of factors (among 1...K)
double matrix_factor_model::partial_deviance(const vector<int> &factor, int k) {
    return(0);
}


// #############################################################################
// `variational_matrix_factor_model` class

// constructor for the class `variational_matrix_factor_model`
variational_matrix_factor_model::variational_matrix_factor_model(int n, int p, int K, const MatrixXd &X)
    : matrix_factor_model(n, p, K, X) {
    m_EU = MatrixXd::Ones(n,K);
    m_EV = MatrixXd::Ones(p,K);
    m_ElogU = MatrixXd::Ones(n,K);
    m_ElogV = MatrixXd::Ones(p,K);
}

// initialize variational and hyper parameters in sparse model
void variational_matrix_factor_model::init_sparse_param(double sel_bound, myRandom::RNGType &rng) {}

// update rules for parameters in the optimization process
void variational_matrix_factor_model::update_param() {
    this->update_variational_param();
    this->update_hyper_param();
}

// update parameters values between iterations
void variational_matrix_factor_model::prepare_next_iterate() {
    this->prepare_next_iterate_variational_param();
    this->prepare_next_iterate_hyper_param();
}

// update rule for variational parameters in the optimization process
void variational_matrix_factor_model::update_variational_param() {}

// update rules for prior hyper-parameters in the optimization process
void variational_matrix_factor_model::update_hyper_param() {}

// update variational parameters values between iterations
void variational_matrix_factor_model::prepare_next_iterate_variational_param() {}

// update prior hyper-parameters values between iterations
void variational_matrix_factor_model::prepare_next_iterate_hyper_param() {}

}
