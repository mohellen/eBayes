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

#ifndef TOOLS_IO_HPP_
#define TOOLS_IO_HPP_

#include <config.h>
#include <string>
#include <sstream>
#include <fstream>


template <typename T>
void write_array_to_column(
		std::size_t len,	 /// Input: data vector size
		T* arr,			     /// Input: data vector to be written to file
		std::string outfile) /// Input: name of the output file
{
	std::ofstream fout;
	fout.open(outfile.c_str(), std::ofstream::out);
	for (int i=0; i < len; i++) {
		fout << arr[i] << std::endl;
	}
	fout.close();
	return;
}

std::string arr_to_string(std::size_t len, double* m);



#endif /* TOOLS_IO_HPP_ */
