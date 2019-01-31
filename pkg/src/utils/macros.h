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
* \brief definitions of macros
* \author Ghislain Durif
* \version 1.0
* \date 18/12/2018
*/

#ifndef MACROS_H
#define MACROS_H

#include <boost/math/special_functions/digamma.hpp>
#include <math.h>
#include "internal.h"

// [[Rcpp::depends(BH)]]
using boost::math::digamma;

// require the following line i Makevars: CXX_STD = CXX11
// #define mclog() unaryExpr([](double d) -> double { return internal::custom_log(d); })
// #define mdigamma() unaryExpr([](double d) -> double { return digamma(d); })
// #define mdigammaInv() unaryExpr([](double d) -> double { return internal::digammaInv(d,6); })
// #define mexpit() unaryExpr([](double d) -> double { return internal::expit(d); })
// #define mlgamma() unaryExpr([](double d) -> double { return lgamma(d); })
// #define mlog() unaryExpr([](double d) -> double { return std::log(d); })

#define mcexp() unaryExpr(std::ptr_fun<double,double>(internal::custom_exp))
#define mclog() unaryExpr(std::ptr_fun<double,double>(internal::custom_log))
#define mdigamma() unaryExpr(std::ptr_fun<double,double>(digamma))
#define mdigammaInv() unaryExpr(std::bind2nd(std::pointer_to_binary_function<double,int,double>(internal::digammaInv),6))
#define mexpit() unaryExpr(std::ptr_fun<double,double>(internal::expit))
#define mlgamma() unaryExpr(std::ptr_fun<double,double>(lgamma))
#define mlog() unaryExpr(std::ptr_fun<double,double>(std::log))

#endif // MACROS_H
