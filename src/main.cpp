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

#include <model/ForwardModel.hpp>
#include <model/NS.hpp>
#include <surrogate/SGI.hpp>
#include <mcmc/MCMC.hpp>
#include <mcmc/MetropolisHastings.hpp>
#include <mcmc/ParallelTempering.hpp>
#include <tools/ErrorAnalysis.hpp>
#include <tools/Parallel.hpp>

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <vector>
#include <memory>
#include <cstdlib>


using namespace std;

double true_input[] = {1.0, 0.8, 3.0, 1.5, 5.5, 0.2, 8.2, 1.0};

Parallel par;

void test_ns_mpi() {
#if(1==0)
	string inputfile = "./input/obstacles_in_flow.dat";

	NS* fm = new NS(inputfile, 1, 1);
	std::size_t input_size = fm->get_input_size();
	std::size_t output_size = fm->get_output_size();

	double noise;
	double* od = ForwardModel::get_observed_data(inputfile, output_size, noise);

	for (int p=0; p < par.mpisize; p++) {
		if (par.mpirank == p) {
			printf("\nRank %d: Observed data\n", par.mpirank);
			for (size_t j=0; j < output_size; j++)
				printf("%.6f ", od[j]);
			printf("\n");
		}
	}

	double* m = new double[input_size];
	m[0] = 1.0;
	m[1] = 0.8;
	m[2] = 3.0;
	m[3] = 1.5;
	m[4] = 5.5;
	m[5] = 0.2;
	m[6] = 8.2;
	m[7] = 1.0;

	double* d = new double[output_size];
	fm->run(m, d);

	for (int p=0; p < par.mpisize; p++) {
		if (par.mpirank == p) {
			printf("\nRank %d: Output data\n", par.mpirank);
			for (size_t j=0; j < output_size; j++)
				printf("%.6f ", d[j]);
			printf("\n");
		}
	}

	double sigma = ForwardModel::compute_posterior_sigma(od, output_size, noise);
	double pos = ForwardModel::compute_posterior(od, d, output_size, sigma);

	for (int p=0; p < par.mpisize; p++) {
		if (par.mpirank == p) {
			printf("\nRank %d: sigma = %.6f, posterior = %.6f\n", par.mpirank, sigma, pos);
		}
	}
#endif
}

void run_asgi() {
#if (1==1)
	int init_level = 2;
	int nsamples = 50000;
	bool is_dup = false;
	string inputfile = "./input/ns_obs4.dat";
	string outpath = "./output/obs4_asgi" + to_string(init_level) +"/";
	string cmd = "mkdir -p " + outpath;
	system(cmd.c_str());

	// Create forward models
	unique_ptr<NS>  full (new NS(inputfile, 1, 1));	 /// full model
	unique_ptr<SGI> sgi (new SGI(par, inputfile, outpath, 1, 1)); /// adaptive sgi

	// Get problem dimensions
	std::size_t input_size = full->get_input_size();
	std::size_t output_size = full->get_output_size();

	// Initialize the true input vector
	unique_ptr<double[]> m (new double[input_size]);
	for (std::size_t i=0; i < input_size; i++)
		m[i] = true_input[i];

	// ASGI
	// 1. build surrogate: duplicate or build from scratch
	if (is_dup) {
		sgi->duplicate("","","");
	} else {
		unique_ptr<ErrorAnalysis> ea (new ErrorAnalysis(full.get(), sgi.get()));
		ea->add_test_points(20);
		double err = 0.0, err_old = -1.0, tol = 0.1;
		int count = 0;
		while (true) {
			sgi->build(init_level, 0.1, false);
			err = ea->compute_model_error();
			count += 1;
			if(par.is_master()) {
				printf("\nRefinement # %d\n", count);
				printf("Adaptive Surrogate model error: %.6f\n", err);
			}
			if ((err - err_old < 0) && (fabs(err - err_old) < tol)) break;
			err_old = err;
		}
	}
	// 2. run MCMC
	vector<vector<double> > inits = sgi->get_top_maxpos(20, "");
	unique_ptr<MCMC> mcmc (new ParallelTempering(par, *sgi, inputfile, 0.2));
	mcmc->run(outpath, nsamples, inits);
#endif
}

void run_ssgi() {
#if (1==0)
	int init_level = 2;
	int nsamples = 50000;
	bool is_dup = false;
	string inputfile = "./input/ns_obs4.dat";
	string outpath = "./output/obs4_ssgi" + to_string(init_level) +"/";
	string cmd = "mkdir -p " + outpath;
	system(cmd.c_str());

	// Create forward models
	unique_ptr<NS>  full (new NS(inputfile, 1, 1));	 /// full model
	unique_ptr<SGI> ssgi (new SGI(inputfile, outpath, 1, 1)); /// adaptive sgi

	// Get problem dimensions
	std::size_t input_size = full->get_input_size();
	std::size_t output_size = full->get_output_size();

	// Initialize the true input vector
	unique_ptr<double[]> m (new double[input_size]);
	for (std::size_t i=0; i < input_size; i++)
		m[i] = true_input[i];

	// SSGI
	// 1. build surrogate: duplicate or build from scratch
	if (is_dup) {
		ssgi->duplicate("","","");
	} else {
		unique_ptr<ErrorAnalysis> ea (new ErrorAnalysis(full.get(), ssgi.get()));
		ea->add_test_points(20);
		double err = 0.0, err_old = -1.0, tol = 0.1;
		int count = 0;
		ssgi->build(init_level, 0.1, false);
	}
	// 2. run MCMC
	unique_ptr<MCMC> mcmc (new ParallelTempering(ssgi.get(), inputfile, 0.2));
	mcmc->run(outpath, nsamples);
#endif
}



int main(int argc, char* argv[]) {

	par.mpi_init(argc, argv);

	test_ns_mpi();
	run_asgi();
	run_ssgi();

#if defined(IMPI)
	printf("\n~~~~~~This is awesome!!~~~~~\n");
#endif

	par.mpi_final();
	return 0;
}

