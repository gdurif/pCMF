// Copyright 2017-06 Ghislain Durif
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
 * \brief header file for random number generation functions
 * \author Ghislain Durif
 * \version 1.0
 * \date 08/02/2018
 */

#ifndef myRANDOM_H
#define myRANDOM_H

#include <boost/random/gamma_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <ctime>
#include <RcppEigen.h>

// [[Rcpp::depends(BH)]]
using boost::random::gamma_distribution;
using boost::random::mt19937;
using boost::random::poisson_distribution;
using boost::random::uniform_int_distribution;
using boost::random::uniform_real_distribution;
using boost::random::variate_generator;

// [[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;
using Eigen::VectorXd;


/*!
* \namespace myRandom
*
* A specific namespace for random related functions
*/
namespace myRandom {

/*!
 * \typedef RNG type (Mersenne-Twister)
 */
typedef mt19937 RNGType;

/*!
 * \fn initialize a Mersenne-Twister Random Number Generator based on local time
 *
 * \return a Boost Mersenne-Twister random number generator
 */
inline RNGType rngInit() {
    RNGType rng(static_cast<unsigned int>(std::time(0)));
    return rng;
};

/*!
 * \fn initialize a Mersenne-Twister Random Number Generator with a user specified seed
 *
 * \param[in] seed uint32_t integer (between 0 and 2^32-1)
 *
 * \return a Boost Mersenne-Twister random number generator
 */
inline RNGType rngInit(uint32_t seed) {
    RNGType rng(seed);
    return rng;
};

/*!
 * \fn generate random sample from Gamma distribution
 *
 * Simulate n repetitions of Gamma(param1, param2)
 *
 * Note: param1 = shape, param2 = rate
 *
 * \param[out] vec vector to store the simulated values
 * \param[in] n number of values to generate
 * \param[in] param1 shape Gamma parameter
 * \param[in] param1 rate Gamma parameter
 * \param[in] rng Random Number Generator from boost
 */
inline void rGamma(VectorXd &vec, int n, double param1, double param2, RNGType &rng) {

    // Gamma generator
    gamma_distribution<> myGamma(param1, 1/param2);
    variate_generator< RNGType &, gamma_distribution<> >generator(rng, myGamma);

    // n is assumed to be the length of vector vec
    for(int ind=0; ind<n; ind++) {
        vec(ind) = generator();
    }
};


/*!
 * \fn generate random sample of unsigned int32 integers
 *
 * Simulate n repetitions of Uniform distribution on unsigned 32bit integer
 *
 * \param[out] vec vector to store the simulated values
 * \param[in] n number of values to generate
 * \param[in] rng Random Number Generator from boost
 */
inline void rInt32(uint32_t* vec, int n, RNGType &rng) {

    // integer generator
    uniform_int_distribution<> candidate(0, std::numeric_limits<uint32_t>::max());
    variate_generator< RNGType &, uniform_int_distribution<> >generator(rng, candidate);

    // n is assumed to be the length of vector vec
    for(int ind=0; ind<n; ind++) {
        vec[ind] = generator();
    }
};

/*!
 * \fn generate random sample from Poisson distribution
 *
 * Simulate n repetitions of Poisson(rate)
 *
 * Note: rate is the Poisson intensity
 *
 * \param[out] vec vector to store the simulated values
 * \param[in] n number of values to generate
 * \param[in] rate Poisson intensity
 * \param[in] rng Random Number Generator from boost
 */
inline void rPoisson(VectorXd &vec, int n, double rate, RNGType &rng) {

    // Uniform generator
    poisson_distribution<int> myPois(rate);
    variate_generator< RNGType &, poisson_distribution<int> >generator(rng, myPois);

    // n is assumed to be the length of vector vec
    for(int ind=0; ind<n; ind++) {
        vec(ind) = generator();
    }
};

/*!
 * \fn generate a matrix of random sample from Poisson distribution with different parameters
 *
 * Simulate nrow x ncol repetition of Poisson distributions
 *
 * Note: param1 = min, param2 = max with a different
 * parameter pair (param1, param2) for each drawning
 *
 * \param[out] mat matrix to store the simulated values
 * \param[in] nrow number of rows in the matrix mat
 * \param[in] ncol number of cols in the matrix mat
 * \param[in] rate matrix of Poisson intensity
 * \param[in] rng Random Number Generator from boost
 */
inline void rPoisson(MatrixXd &mat, int nrow, int ncol,
                     const MatrixXd &rate, RNGType &rng) {

    // nrow and ncol are assumed to be consistent with the dimension of the different input matrices
    for(int rowInd=0; rowInd<nrow; rowInd++) {
        for(int colInd=0; colInd<ncol; colInd++) {
            // Uniform generator
            poisson_distribution<int> myPois(rate(rowInd, colInd));
            variate_generator< RNGType &, poisson_distribution<int> >generator(rng, myPois);
            mat(rowInd, colInd) = generator();
        }
    }
};

/*!
 * \fn generate random sample from uniform distribution
 *
 * Simulate n repetitions of Uniform(param1, param2)
 *
 * Note: param1 = min, param2 = max
 *
 * \param[out] vec vector to store the simulated values
 * \param[in] n number of values to generate
 * \param[in] param1 min uniform parameter
 * \param[in] param2 max uniform parameter
 * \param[in] rng Random Number Generator from boost
 */
inline void rUnif(VectorXd &vec, int n, double param1, double param2, RNGType &rng) {

    // Uniform generator
    uniform_real_distribution<> myUnif(param1, param2);
    variate_generator< RNGType &, uniform_real_distribution<> >generator(rng, myUnif);

    // n is assumed to be the length of vector vec
    for(int ind=0; ind<n; ind++) {
        vec(ind) = generator();
    }
};

/*!
 * \fn generate a matrix of random sample from uniform distribution
 *
 * Simulate nrow x ncol repetition of Uniform(param1, param2)
 *
 * Note: param1 = min, param2 = max with a different
 * parameter pair (param1, param2) for each drawning
 *
 * \param[out] mat matrix to store the simulated values
 * \param[in] nrow number of rows in the matrix mat
 * \param[in] ncol number of cols in the matrix mat
 * \param[in] param1 min uniform parameter
 * \param[in] param2 max uniform parameter
 * \param[in] rng Random Number Generator from boost
 */
inline void rUnif(MatrixXd &mat, int nrow, int ncol,
                  double param1, double param2, RNGType &rng) {

    uniform_real_distribution<> myUnif(param1, param2);
    variate_generator< RNGType &, uniform_real_distribution<> >generator(rng, myUnif);

    // nrow and ncol are assumed to be consistent with the dimension of the different input matrices
    for(int rowInd=0; rowInd<nrow; rowInd++) {
        for(int colInd=0; colInd<ncol; colInd++) {
            // Uniform generator
            mat(rowInd, colInd) = generator();
        }
    }
};

/*!
 * \fn generate a matrix of random sample from uniform distribution with different parameters
 *
 * Simulate nrow x ncol repetition of Uniform distributions
 *
 * Note: param1 = min, param2 = max with a different
 * parameter pair (param1, param2) for each drawning
 *
 * \param[out] mat matrix to store the simulated values
 * \param[in] nrow number of rows in the matrix mat
 * \param[in] ncol number of cols in the matrix mat
 * \param[in] param1 matrix of min uniform parameter
 * \param[in] param2 matrix of max uniform parameter
 * \param[in] rng Random Number Generator from boost
 */
inline void rUnif(MatrixXd &mat, int nrow, int ncol,
                  const MatrixXd &param1, const MatrixXd &param2, RNGType &rng) {

    // nrow and ncol are assumed to be consistent with the dimension of the different input matrices
    for(int rowInd=0; rowInd<nrow; rowInd++) {
        for(int colInd=0; colInd<ncol; colInd++) {
            // Uniform generator
            uniform_real_distribution<> myUnif(param1(rowInd, colInd),
                                               param2(rowInd, colInd));
            variate_generator< RNGType &, uniform_real_distribution<> >generator(rng, myUnif);
            mat(rowInd, colInd) = generator();
        }
    }
};

}

#endif // myRANDOM_H
