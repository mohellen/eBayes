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

#include <mcmc/MCMC.hpp>

using namespace std;

MCMC::MCMC(
		ForwardModel* forwardmodel,
		const string& observed_data_file,
		double rand_walk_size_domain_percent)
{
	this->model = forwardmodel;
	this->input_size = model->get_input_size();
	this->output_size = model->get_output_size();

	this->rand_walk_size.reset(new double[input_size]);
	double dmin, dmax;
	for (size_t i=0; i < input_size; i++) {
		model->get_input_space(i, dmin, dmax);
		rand_walk_size[i] = (dmax-dmin) * rand_walk_size_domain_percent;
	}

	// Get observed data and noise
	this->observed_data.reset(
			ForwardModel::get_observed_data(observed_data_file, output_size,
					this->observed_data_noise));

	this->pos_sigma;



}



double* MCMC::get_last_sample(const string& input_file)
{
	double* last_sample = nullptr;

	// Open file and place cursor at EOF
	ifstream fin (input_file, std::ios_base::ate);
	if (!fin.is_open()) {
		printf("MCMC get last samle open file fail. Abort!");
		return last_sample;
	}
	size_t len = fin.tellg(); // Get length of file

	// Loop backwards
	char c;
	string line;
	for (size_t i=len-2; i > 0; i--) {
		fin.seekg(i);
		c = fin.get();
		if (c == '\r' || c == '\n') {
			std::getline(fin, line);
			istringstream iss(line);
			vector<string> tokens {istream_iterator<string>{iss}, istream_iterator<string>{}};
			size_t ntokens = tokens.size();

			if (ntokens <= 0) {// skip empty line
				continue;
			} else {
				last_sample = new double[ntokens];
				for (int k=0; k < ntokens; k++)
					last_sample[k] = stod(tokens[k]);
				break;
			} //end if ntokens
		}
	}//end for
	fin.close();
	return last_sample;
}

double* gen_random_sample()
{



}
