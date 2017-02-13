// This file is part of BayeSIFSG - Bayesian Statistical Inference Framework with Sparse Grid
// Copyright (C) 2015-today Ao Mo-Hellenbrand.
//
// SIPFSG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// SIPFSG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License.
// If not, see <http://www.gnu.org/licenses/>.

#ifndef TOOLS_MATRIX_HPP_
#define TOOLS_MATRIX_HPP_

#include <cstddef>
#include <iostream>
#include <iomanip>

namespace tools {

/**
 * Allocate 2D matrix:
 * 		For row-major    storage : M[index of rows][index of columns]
 * 		For column-major storage : M[index of columns][index of rows]
 */
template<typename T>
T** alloc_matrix(
		std::size_t nrows,
		std::size_t ncols,
		bool is_row_major)
{
	// size check
	if ((nrows <= 0) || (ncols <= 0)) return nullptr;

	T** m;
	T* elems = new T[nrows * ncols];

	if (is_row_major) {
		m = new T*[nrows];
		for (std::size_t i=0; i < nrows; i++) {
			m[i] = &elems[i*ncols];
		}
	} else {
		m = new T*[ncols];
		for (std::size_t i=0; i < ncols; i++) {
			m[i] = &elems[i*nrows];
		}
	}
	return m;
}

/**
 * Free 2D matrix : in the same manner it was created
 */
template<typename T>
void free_matrix(T** m)
{
	if (m) { // free non-null pointer only
		delete[] m[0];
		delete[] m;
	}
	return;
}

/**
 * Initialize the entire 2D matrix with a value
 */
template<typename T>
void init_matrix(
		T** m,
		std::size_t nrows,
		std::size_t ncols,
		bool is_row_major,
		T value)
{
	if (is_row_major) {
		for (std::size_t r=0; r < nrows; r++)
			for (std::size_t c=0; c < ncols; c++)
				m[r][c] = value;
	} else {
		for (std::size_t c=0; c < ncols; c++)
			for (std::size_t r=0; r < nrows; r++)
				m[c][r] = value;
	}
	return;
}

/**
 * Initialize a square area of a 2D matrix
 * NOTE: Indices are NOT checked. Make sure the indices are within bound!
 */
template<typename T>
void init_matrix(
		T** m,
		std::size_t nrl,
		std::size_t nrh,
		std::size_t ncl,
		std::size_t nch,
		bool is_row_major,
		T value)
{
	if (is_row_major) {
		for (std::size_t r=nrl; r <= nrh; r++)
			for (std::size_t c=ncl; c <= nch; c++)
				m[r][c] = value;
	} else {
		for (std::size_t c=ncl; c <= nch; c++)
			for (std::size_t r=nrl; r <= nrh; r++)
				m[c][r] = value;
	}
	return;
}

/**
 * Print the entire 2D matrix
 */
template<typename T>
void print_matrix(
		T** m,
		std::size_t nrows,
		std::size_t ncols,
		bool is_row_major,
		std::size_t elem_width)
{
	if (is_row_major) {
		for (std::size_t r=0; r < nrows; r++) {
			for (std::size_t c=0; c < ncols; c++) {
				std::cout << std::setw(elem_width) << m[nrows-1-r][c] << " ";
			}
			std::cout << std::endl;
		}
	} else {
		for (std::size_t r=0; r < nrows; r++) {
			for (std::size_t c=0; c < ncols; c++) {
				std::cout << std::setw(elem_width) << m[c][nrows-1-r] << " ";
			}
			std::cout << std::endl;
		}
	}
	return;
}

/**
 * Print a square area of a 2D matrix.
 * NOTE: Indices are NOT checked. Make sure the indices are within bound!
 */
template<typename T>
void print_matrix(
		T** m,
		std::size_t nrl,
		std::size_t nrh,
		std::size_t ncl,
		std::size_t nch,
		bool is_row_major,
		std::size_t elem_width)
{
	if (is_row_major) {
		for (std::size_t r=nrl; r<=nrh; r++) {
			for (std::size_t c=ncl; c<=nch; c++) {
				std::cout << std::setw(elem_width) << m[nrh-r][c] << " ";
			}
			std::cout << std::endl;
		}
	} else {
		for (std::size_t c=ncl; c<=nch; c++) {
			for (std::size_t r=nrl; r<=nrh; r++) {
				std::cout << std::setw(elem_width) << m[c][nrh-r] << " ";
			}
			std::cout << std::endl;
		}
	}
	return;
}

/**
 * Compare element by element of the same area in two 2D matrices
 */
template<typename T>
bool compare_matrix(
		T** m1,
		T** m2,
		std::size_t nrl,
		std::size_t nrh,
		std::size_t ncl,
		std::size_t nch,
		bool is_row_major)
{
	bool result = true;
	if (is_row_major) {
		for (std::size_t r=nrl; r<=nrh; r++) {
			for (std::size_t c=ncl; c<=nch; c++) {
				if (m1[r][c] == m2[r][c]) {
					continue;
				} else {
					std::cout << "m1[" << r << "][" << c << "] = " << m1[r][c]
	                        << "; m2[" << r << "][" << c << "] = " << m2[r][c]
						    << std::endl;
					result = false;
				}
			}
		}
	} else {
		for (std::size_t c=ncl; c<=nch; c++) {
			for (std::size_t r=nrl; r<=nrh; r++) {
				if (m1[c][r] == m2[c][r]) {
					continue;
				} else {
					std::cout << "m1[" << c << "][" << r << "] = " << m1[c][r]
	                        << "; m2[" << c << "][" << r << "] = " << m2[c][r]
					        << std::endl;
					result = false;
				}
			}
		}
	}
	if (result)
		std::cout << "m1 == m2" << std::endl;
	else
		std::cout << "m1 != m2" << std::endl;
	return result;
}

/**
 * Compare element by element of the same area in two 2D matrices
 */
template<typename T>
void compare_print_matrix(
		T** m1,
		T** m2,
		std::size_t nrl,
		std::size_t nrh,
		std::size_t ncl,
		std::size_t nch,
		bool is_row_major,
		std::size_t elem_width)
{
	if (is_row_major) {
		for (std::size_t r=nrl; r<=nrh; r++) {
			std::cout << "\n[m1] row " << r << ": ";
			for (std::size_t c=ncl; c<=nch; c++) {
				std::cout << std::setw(elem_width) << m1[r][c] << "  ";
			}
			std::cout << std::endl;

			std::cout << "[m2] row " << r << ": ";
			for (std::size_t c=ncl; c<=nch; c++) {
				std::cout << std::setw(elem_width) << m2[r][c] << "  ";
			}
			std::cout << std::endl;
		}
	} else {
		for (std::size_t r=nrl; r<=nrh; r++) {
			std::cout << "\n[m1] row " << r << ": ";
			for (std::size_t c=ncl; c<=nch; c++) {
				std::cout << std::setw(elem_width) << m1[c][r] << "  ";
			}
			std::cout << std::endl;

			std::cout << "[m2] row " << r << ": ";
			for (std::size_t c=ncl; c<=nch; c++) {
				std::cout << std::setw(elem_width) << m2[c][r] << "  ";
			}
			std::cout << std::endl;
		}
	}
	return;
}

}
#endif /* TOOLS_MATRIX_HPP_ */
