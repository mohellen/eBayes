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

#include <model/ForwardModel.hpp>

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

double ForwardModel::compute_l2norm(
		const double* d1, 
		const double* d2,
		std::size_t data_size)
{
	double tmp = 0.0;
	for (std::size_t j=0; j < data_size; j++)
		tmp += (d1[j] - d2[j])*(d1[j] - d2[j]);
	return sqrt(tmp);
}

double* ForwardModel::get_observed_data(
		const std::string & input_file,
		std::size_t data_size,
		double& noise_in_data)
{
	double* d = new double[data_size];

	ifstream infile(input_file);
	string s;
	while (std::getline(infile, s)) {
		istringstream iss(s);
		vector<string> tokens {istream_iterator<string>{iss}, istream_iterator<string>{}};

		// Ignore empty line
		if (tokens.size() <= 0) continue;

		// Ignore comment line
		tokens[0] = trim_white_space(tokens[0]);
		if (tokens[0].substr(0,2) == "//") continue;

		// Find parameters
		transform(tokens[0].begin(), tokens[0].end(), tokens[0].begin(), ::tolower);
		if (tokens[0] == "observed_data") {
			for (std::size_t j=0; j < data_size; j++) {
				d[j] = stod(tokens[j+1]);
			}
			continue;
		}
		if (tokens[0] == "noise_in_data") {
			noise_in_data = stod(tokens[1]);
			continue;
		}
	}//end while
	infile.close();
	return d;
}

string ForwardModel::trim_white_space(const string& str)
{
	std::string whitespace=" \t";
	const auto strBegin = str.find_first_not_of(whitespace);
	if (strBegin == std::string::npos) return "";
	const auto strEnd = str.find_last_not_of(whitespace);
	const auto strRange = strEnd - strBegin + 1;
	return str.substr(strBegin, strRange);
}

string ForwardModel::arr_to_string(const double* m, std::size_t len)
{
	std::ostringstream oss;
	oss << "[" << std::fixed << std::setprecision(4);
	for (std::size_t i=0; i < len-1; i++)
		oss << m[i] << ", ";
	oss << m[len-1] << "]";
	return oss.str();
}
