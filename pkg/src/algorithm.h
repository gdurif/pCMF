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
 * \brief definitions of the template 'algorithm' class
 * \author Ghislain Durif
 * \version 1.0
 * \date 07/02/2018
 */

#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <Rcpp.h>
#include <RcppEigen.h>

#include "utils/random.h"

// [[Rcpp::depends(RcppEigen)]]
using Eigen::VectorXd;                  // variable size vector, double precision

namespace pCMF {

/*!
 * \brief template class definition for a generic optimization/inference algorithm
 * \tparam model the statistical model considered for the inference framework
 */
template <typename model>
class algorithm {

protected:
    model m_model;      /*!< the statistical model on which the template is built */

    bool m_verbose;     /*!< boolean indicating verbosity in the output */
    bool m_monitor;     /*!< boolean indicating if log-likelihood, elbo, deviance, etc. should be monitored during optimization */

    // iterations
    int m_iter_max;      /*!< maximum number of iterations */
    int m_iter;         /*!< current iteration */
    int m_iter_min;      /*!< minimum number of iterations */

    // convergence
    double m_epsilon;           /*!< precision value to assess convergence */
    int m_additional_iter;      /*!< number of iterations where model parameter values are stable to confirm convergence */
    bool m_converged;           /*!< status of convergence */
    int m_conv_mode;            /*!< how to assess convergence: 0 for absolute gap, 1 for normalized gap between two iterates, 2 for a custom criterion */

    VectorXd m_conv_criterion;  /*!< convergence criterion used to assess convergence. Its value depends on 'conv_mode' */

    VectorXd m_abs_gap;         /*!< absolute gap between two iterates */
    VectorXd m_norm_gap;        /*!< normalized gap between two iterates */

    // criterion
    double m_loss;              /*!< value of the loss after optimization */
    VectorXd m_optim_criterion; /*!< values of the loss across iterations */
    VectorXd m_loglikelihood;   /*!< values of the log-likelihood of the model over iterations */
    VectorXd m_deviance;        /*!< values of the deviance associated to the model over iterations */

public:

    /*!
     * \brief Constructor for the class `algorithm`
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
    algorithm(int n, int p, int K, const MatrixXd &X,
              bool verbose, bool monitor, int iter_max, int iter_min,
              double epsilon, int additional_iter, int conv_mode)
        : m_model(n, p, K, X) {

        m_verbose = verbose;
        m_monitor = monitor;
        m_iter_max = iter_max;
        m_iter_min = iter_min;
        m_epsilon = epsilon;
        m_additional_iter = additional_iter;
        m_conv_mode = conv_mode;

        m_converged = false;

        // initialize other attributes
        m_iter = 0;
        m_conv_criterion = VectorXd::Zero(m_iter_max);

        m_loss = 0;

        if(monitor) {
            m_optim_criterion = VectorXd::Zero(m_iter_max);
            m_abs_gap = VectorXd::Zero(m_iter_max);
            m_norm_gap = VectorXd::Zero(m_iter_max);
            m_loglikelihood = VectorXd::Zero(m_iter_max);
            m_deviance = VectorXd::Zero(m_iter_max);
        } else {
            m_optim_criterion = VectorXd::Zero(1);
            m_abs_gap = VectorXd::Zero(1);
            m_norm_gap = VectorXd::Zero(1);
            m_loglikelihood = VectorXd::Zero(1);
            m_deviance = VectorXd::Zero(1);
        }

    };

    /*!
     * \brief Destructor for the class `algorithm`
     */
    ~algorithm() {};

    /*!
     * \brief Initialization of the model parameters with random values
     *
     * \param[in] rng Random Number generator
     */
    void init_random(myRandom::RNGType &rng) {
        m_model.random_init_model_param(rng);
    };

    /*!
     * \brief randomly perturb parameters of the model
     *
     * \param[in,out] rng random number generator
     * \param[in] noise_level level of the perturbation, based on a uniform
     * distribution on [-noise_level, noise_level]
     */
    void perturb_param(myRandom::RNGType &rng, double noise_level) {
        m_model.perturb_param(rng, noise_level);
    };

    /*!
     * \brief run the optimization process
     */
    virtual void optimize() {

        // Iteration
        int nb_iter_stab = 0; // number of successive iteration where the normalized gap betwwen two iteration is close to zero (convergence when nstab > rstab)

        while( (m_iter < m_iter_max) && (m_converged==false) ) {

#if defined(_DEV)
            Rcpp::Rcout << "iter " << m_iter << std::endl;
            // check for interrupt
            Rcpp::checkUserInterrupt();
#else
            if((m_verbose==true) && (m_iter % 10 == 0) ) {
                Rcpp::Rcout << "iter " << m_iter << std::endl;
                // check for interrupt
                Rcpp::checkUserInterrupt();
            }
#endif

            // update parameters
#if defined(_DEV)
            Rcpp::Rcout << "update_param" << std::endl;
#endif
            this->update_param();

            // monitor criterion during optimization (if required)
#if defined(_DEV)
            Rcpp::Rcout << "monitor" << std::endl;
#endif
            this->monitor();

            // convergence
#if defined(_DEV)
            Rcpp::Rcout << "check_convergence" << std::endl;
#endif
            this->check_convergence(nb_iter_stab);

            // prepare for next iterate
#if defined(_DEV)
            Rcpp::Rcout << "prepare_next_iterate" << std::endl;
#endif
            this->prepare_next_iterate();

            // increment iteration
            m_iter++;
        }

        // compute the optimization criterion (current value of the loss)
#if defined(_DEV)
        Rcpp::Rcout << "optim_criterion" << std::endl;
#endif
        this->optim_criterion();
    };

    // getter
    /*!
     * \brief getter to check convergence
     *
     * \param[out] nb_iter returns the current number of iterations
     * \param[out] converged returns the convergence status
     */
    void get_conv(int &nb_iter, bool& converged) {
        nb_iter = m_iter;
        converged = m_converged;
    };

    /*!
     * \brief getter for monitored critera
     *   * m_conv_criterion
     *   * m_abs_gap
     *   * m_norm_gap
     *   * m_optim_criterion
     *   * m_loglikelihood
     *   * m_elbo
     *   * m_deviance
     *
     * \param[out] conv_criterion value of the convergence criterion
     * \param[out] abs_gap absolute gap between iterates
     * \param[out] norm_gap normalized gap between iterates
     * \param[out] optim_criterion values of the loss across iterations
     * \param[out] loglikelihood values of the joint log-likelihood across
     * iterations
     * \param[out] deviance values of the deviance across iterations
     */
    void get_criterion(VectorXd &conv_criterion,
                       VectorXd &abs_gap, VectorXd &norm_gap,
                       VectorXd &optim_criterion, VectorXd &loglikelihood,
                       VectorXd &deviance) {
        conv_criterion = m_conv_criterion.head(m_iter);
        if(m_monitor) {
            abs_gap = m_abs_gap.head(m_iter);
            norm_gap = m_norm_gap.head(m_iter);
            loglikelihood = m_loglikelihood.head(m_iter);
            optim_criterion = m_optim_criterion.head(m_iter);
            deviance = m_deviance.head(m_iter);
        } else {
            abs_gap = m_abs_gap;
            norm_gap = m_norm_gap;
            loglikelihood = m_loglikelihood;
            optim_criterion = m_optim_criterion;
            deviance = m_deviance;
        }
    };

    /*!
     * \brief get current value of the loss
     *
     * \return current value of the loss
     */
    double get_loss() {
        return(m_loss);
    }

    // setter
    /*!
     * \brief setter for monitored critera
     *   * m_conv_criterion
     *   * m_abs_gap
     *   * m_norm_gap
     *   * m_optim_criterion
     *   * m_loglikelihood
     *   * m_deviance
     *
     * \param[in] conv_criterion value of the convergence criterion
     * \param[in] abs_gap absolute gap between iterates
     * \param[in] norm_gap normalized gap between iterates
     * \param[in] optim_criterion values of the loss across iterations
     * \param[in] loglikelihood values of the joint log-likelihood across
     * iterations
     * \param[in] deviance values of the deviance across iterations
     */
    void set_criterion(const VectorXd &conv_criterion,
                       const VectorXd &abs_gap, const VectorXd &norm_gap,
                       const VectorXd &optim_criterion, const VectorXd &loglikelihood,
                       const VectorXd &deviance) {
        m_conv_criterion.head(conv_criterion.size()) = conv_criterion;
        if(m_monitor) {
            m_abs_gap.head(abs_gap.size()) = abs_gap;
            m_norm_gap.head(norm_gap.size()) = norm_gap;
            m_loglikelihood.head(loglikelihood.size()) = loglikelihood;
            m_optim_criterion.head(optim_criterion.size()) = optim_criterion;
            m_deviance.head(deviance.size());
        } else {
            m_abs_gap = abs_gap;
            m_norm_gap = norm_gap;
            m_loglikelihood = loglikelihood;
            m_optim_criterion = optim_criterion;
            m_deviance = deviance;
        }
    };

    /*!
     * \brief setter for current number of iterations
     * (for instance to restart an optimization run)
     *
     * \param[in] nb_iter iteration step
     */
    void set_current_iter(int nb_iter) {
        m_iter = nb_iter;
    };

    /*!
     * \brief setter for iteration parameter (iter_min, iter_max, epsilon)
     * (for instance to restart an optimization run)
     *
     * \param[in] iter_min integer, minimum number of iterations
     * \param[in] iter_max integer, maximum number of iterations
     * \param[in] epsilon double, positive value, precision value used
     * to assess convergence
     */
    void set_iter(const int iter_min, const int iter_max,
                  const double epsilon) {
        m_iter_min = iter_min;
        m_iter_max = iter_max;
        m_epsilon = epsilon;
    };

    /*!
     * \brief set verbosity
     */
    void set_verbosity(bool verbose) {
        m_verbose = verbose;
    }

    /*!
     * \brief set monitoring
     */
    void set_monitor(bool monitor) {
        m_monitor = monitor;
    }

    // output
    /*!
     * \brief create list of object to return
     *
     * \param[out] results list of returned objects
     */
    void get_output(Rcpp::List &results) {

        m_model.get_output(results);

        Rcpp::List conv = Rcpp::List::create(Rcpp::Named("converged") = m_converged,
                                             Rcpp::Named("nb_iter") = m_iter,
                                             Rcpp::Named("conv_crit") = m_conv_criterion.head(m_iter),
                                             Rcpp::Named("conv_mode") = m_conv_mode);

        results.push_back(conv, "convergence");
        results.push_back(m_loss, "loss");

        results.push_back(m_model.exp_deviance(), "exp_dev");

        if(m_monitor) {
            Rcpp::List monitor = Rcpp::List::create(Rcpp::Named("norm_gap") = m_norm_gap.head(m_iter),
                                                    Rcpp::Named("abs_gap") = m_abs_gap.head(m_iter),
                                                    Rcpp::Named("loglikelihood") = m_loglikelihood.head(m_iter),
                                                    Rcpp::Named("deviance") = m_deviance.head(m_iter),
                                                    Rcpp::Named("optim_criterion") = m_optim_criterion.head(m_iter));
            results.push_back(monitor, "monitor");
        }
    };

protected:

    /*!
     * \brief update parameters of the model according to iterative optimization rule
     */
    virtual void update_param() {
        m_model.update_param();
    };

    /*!
     * \brief monitor log-likelihood, elbo, deviance and parameter norms of the model (if required)
     */
    virtual void monitor() {
        if(m_monitor==true) {
            m_loglikelihood(m_iter) = m_model.loglikelihood();
            m_optim_criterion(m_iter) = m_model.optim_criterion();
            m_deviance(m_iter) = m_model.deviance();
        }
    };

    /*!
     * \brief compute the current value of the optimization criterion (=loss)
     */
    virtual void optim_criterion() {
        m_loss = m_model.optim_criterion();
    }

    /*!
     * \brief check convergence
     *
     * \param[in,out] nb_iter_stab current number of successive iterations for which
     * the convergence criterion is verified
     */
    virtual void check_convergence(int &nb_iter_stab) {
        double abs_gap;
        double norm_gap;
        double crit;

        // convergence check mode
        if(m_conv_mode == 0) {
            m_model.gap_between_iterates(abs_gap, norm_gap);
            m_conv_criterion(m_iter) = abs_gap;
        } else if(m_conv_mode == 1) {
            m_model.gap_between_iterates(abs_gap, norm_gap);
            m_conv_criterion(m_iter) = norm_gap;
        } else if(m_conv_mode == 2) {
            m_conv_criterion(m_iter) = m_model.custom_conv_criterion();
        } else {
            Rcpp::stop("`conv_mode` parameter should take value in {0,1,2}");
        }

        // in case gap between iterates should be monitored
        // (check to avoid recomputing it)
        if(m_monitor) {
            if(m_conv_mode == 2) {
                m_model.gap_between_iterates(abs_gap, norm_gap);
            }
            m_abs_gap(m_iter) = abs_gap;
            m_norm_gap(m_iter) = norm_gap;
        }

        // convergence check
        if(std::abs(m_conv_criterion(m_iter)) < m_epsilon) {
            nb_iter_stab++;
        } else {
            nb_iter_stab = 0;
        }

        if((nb_iter_stab > m_additional_iter) && (m_iter > m_iter_min)) {
            m_converged=true;
        }
    };

    /*!
     * \brief update parameter values between iterations
     */
    virtual void prepare_next_iterate() {
        m_model.prepare_next_iterate();
    };

};

}

#endif // ALGORITHM_H
