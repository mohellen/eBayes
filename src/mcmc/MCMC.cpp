// eBayes - Elastic Bayesian Inference Framework with iMPI
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

#include <mcmc/MCMC.hpp>

using namespace std;

MCMC::MCMC(
		Parallel& para,
		ForwardModel& forwardmodel,
		const string& observed_data_file,
		double rand_walk_size_domain_percent)
		: par(para), model(forwardmodel)
{
	// Determin how many chains: min(ranks, MAX_CHAINS)
	this->num_chains = (par.mpisize < MCMC_MAX_CHAINS) ? par.mpisize : MCMC_MAX_CHAINS;
	this->input_size = model.get_input_size();
	this->output_size = model.get_output_size();

	this->rand_walk_size.reset(new double[input_size]);
	double dmin, dmax;
	for (size_t i=0; i < input_size; i++) {
		model.get_input_space(i, dmin, dmax);
		rand_walk_size[i] = (dmax-dmin) * rand_walk_size_domain_percent;
	}
	// Get observed data and noise
	this->observed_data.reset(
			ForwardModel::get_observed_data(observed_data_file, output_size,
					this->observed_data_noise));
	// Compute posterior sigma
	this->pos_sigma = ForwardModel::compute_posterior_sigma(
			this->observed_data.get(), output_size, observed_data_noise);
}


double* MCMC::gen_random_sample()
{
	double* sample = new double[input_size];
	mt19937 gen (chrono::system_clock::now().time_since_epoch().count());
	double dmin, dmax;
	for (size_t i=0; i < input_size; i++) {
		model.get_input_space(i, dmin, dmax);
		uniform_real_distribution<double> udist(dmin,dmax);
		sample[i] = udist(gen);
	}
	return sample;
}


bool MCMC::read_maxpos_sample(
		fstream& fin,
		double* point,
		double& pos)
{
	fin.seekg(0, fin.end);
	size_t len = fin.tellg(); // Get length of file

	// NOTE: Considering the mcmc output file would be large and the MAXPOS
	//    sample is stored in the last line, we loop from the bottom of file backwards.

	// Loop backwards
	char c;
	string line;
	for (long int i=len-2; i > 0; i--) {
		fin.seekg(i);
		c = fin.get();
		if (c == '\r' || c == '\n') {
			std::getline(fin, line);
			istringstream iss(line);
			vector<string> tokens {istream_iterator<string>{iss}, istream_iterator<string>{}};
			size_t num = tokens.size();

			if ((num > 0) && (tokens[0] == "MAXPOS")) { //This is the line we are looking for
				// Read in data
				for (int k=0; k < input_size; k++) {
					point[k] = stod(tokens[k+1]);
				}
				pos = stod(tokens[num-1]); // last token is pos
				fin.seekg(0, fin.end);
				return true;
			}
		}
	} //end for
	fin.seekg(0, fin.end);
	return false;
}


void MCMC::write_sample_pos(
		fstream& fout,
		const double* point,
		double pos)
{
	for (size_t i=0; i < input_size; i++) {
		fout << point[i] << " ";
	}
	fout << pos << endl;
	return;
}


void MCMC::one_step_single_dim(
		int dim,
		double& pos,
		double* p,
		double* d)
{
	double pos_tmp = 0.0;
	double dmin, dmax;
	double p_dim_old = p[dim]; /// When reject a proposal, we need ot restore p[dim];

	// Initialize random generators
	mt19937 gen(chrono::system_clock::now().time_since_epoch().count());
	normal_distribution<double> ndist(p[dim], rand_walk_size[dim]);
	uniform_real_distribution<double> udist(0.0,1.0);

	// 1. Draw a proposal (we only update p[dim])
	p[dim] = ndist(gen);
	model.get_input_space(dim, dmin, dmax);
	while ((p[dim] < dmin) || (p[dim] > dmax)) /// ensure p[dim] is in range
		p[dim] = ndist(gen);

	// 2. Compute acceptance rate
	model.run(p, d);
	pos_tmp = ForwardModel::compute_posterior(observed_data.get(), d, output_size, pos_sigma);
	double acc = fmin(1.0, pos_tmp/pos);

	// 3. Accept or reject proposal
	if (udist(gen) <= acc) {
		// Case accept:
		// p is already updated, only update pos
		pos = pos_tmp;
	} else {
		// Case reject:
		// restore p[dim], pos stays the same
		p[dim] = p_dim_old;
	}
	return;
}


double* MCMC::gen_init_sample(
		const string& rank_output_file, 	/// Input
		const vector<vector<double> >& init_sample_pos) /// Optional input
{
	// Initialize rank specific initial sample and posterior...
	// Use the provided initial samples if there's any
	double* init = new double[input_size+1];
	int started_chains = 0;
	int num_inits = init_sample_pos.size();
	int num = (num_inits < num_chains) ? num_inits : num_chains;
	if (par.mpirank < num) {
		for (size_t i=0; i < input_size; i++)
			init[i] = init_sample_pos[par.mpirank][i];
		init[input_size] = init_sample_pos[par.mpirank][input_size];
		return init;
	}
	// If no samples or not enought samples supplied, 2 options:
	// 	 (1) try to read maxpos from input file, if none then
	//	 (2) generate a random sample
	fstream fin (rank_output_file, fstream::in);
	if (fin && read_maxpos_sample(fin, init, init[input_size])) { // (1)
		fin.close();
	} else { // (2)
		unique_ptr<double[]> rand (gen_random_sample());
		unique_ptr<double[]> initd (new double[output_size]);
		for (std::size_t i=0; i < input_size; i++) {
			init[i] = rand[i];
		}
		model.run(rand.get(), initd.get());
		init[input_size] = ForwardModel::compute_posterior(
				observed_data.get(), initd.get(), output_size, pos_sigma);
	}
	return init;
}


