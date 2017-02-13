// iBayes - Elastic Bayesian Inference Framework with Sparse Grid
// Copyright (C) 2015-today Ao Mo-Hellenbrand
//
// All copyrights remain with the respective authors.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include<tools/io.hpp>

std::string arr_to_string(std::size_t len, double* m)
{
	std::ostringstream stream;
	stream << "[";
	for (std::size_t i=0; i < len-1; i++)
		stream << m[i] << ", ";
	stream << m[len-1] << "]";
	return stream.str();
}





