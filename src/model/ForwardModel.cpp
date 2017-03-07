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

#include "model/ForwardModel.hpp"

using namespace std;

double ForwardModel::compute_posterior_sigma(
		const double* observed_data, 
		std::size_t data_size, 
		double noise_in_data)
{
	double mean = 0.0;
	for (std::size_t j=0; j < data_size; j++)
		mean += observed_data[j];
	mean /= (double)data_size;
	return noise_in_data * mean;
}

double ForwardModel::compute_posterior(
		const double* observed_data, 
		const double* d,
		std::size_t data_size,
		double sigma)
{
	double sum = 0.0;
	for (std::size_t j=0; j < data_size; j++)
		sum += (d[j] - observed_data[j])*(d[j] - observed_data[j]);
	return exp(-0.5 * sum / (sigma*sigma));
}

double ForwardModel::compute_2norm(
		const double* d1, 
		const double* d2,
		std::size_t data_size)
{
	double tmp = 0.0;
	for (std::size_t j=0; j < data_size; j++)
		tmp += (d1[j] - d2[j])*(d1[j] - d2[j]);
	return sqrt(tmp);
}
