### Copyright 2013-2017 Hadley Wickham; RStudio
###
### This file is part of the `pCMF' library for R and related languages.
### It is made available under the terms of the GNU General Public
### License, version 2, or at your option, any later version,
### incorporated herein by reference.
###
### This program is distributed in the hope that it will be
### useful, but WITHOUT ANY WARRANTY; without even the implied
### warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
### PURPOSE.  See the GNU General Public License for more
### details.
###
### You should have received a copy of the GNU General Public
### License along with this program; if not, write to the Free
### Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
### MA 02111-1307, USA

#' @title run_testthat_tests
#' @keywords internal
#'
#' @name run_testthat_tests
#'
#' @description
#' This dummy function definition is included with the package to ensure that
#' `tools::package_native_routine_registration_skeleton()` generates the
#' required registration info for the 'run_testthat_tests' symbol.
#'
#' @details
#' See \url{https://github.com/r-lib/testthat} or
#' \url{https://CRAN.R-project.org/package=testthat}
#'
#' @author Hadley Wickham; RStudio
#'
(function() {
    .Call("run_testthat_tests", PACKAGE = "pCMF")
})
