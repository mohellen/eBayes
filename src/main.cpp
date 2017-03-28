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
#include <model/NS.hpp>
#include <surrogate/SGI.hpp>
#include <mcmc/MCMC.hpp>
#include <mcmc/MetropolisHastings.hpp>
#include <mcmc/ParallelTempering.hpp>
#include <analysis/ErrorAnalysis.hpp>

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <vector>
#include <memory>
#include <cstdlib>


using namespace std;

double true_input[] = {1.0, 0.8, 3.0, 1.5, 5.5, 0.2, 8.2, 1.0};

void test_ns_mpi()
{
#if(1==0)
	int mpisize, mpirank, mpistatus;
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

	string inputfile = "./input/obstacles_in_flow.dat";

	NS* fm = new NS(inputfile, 1, 1);
	std::size_t input_size = fm->get_input_size();
	std::size_t output_size = fm->get_output_size();

	double noise;
	double* od = ForwardModel::get_observed_data(inputfile, output_size, noise);

	for (int p=0; p < mpisize; p++) {
		if (mpirank == p) {
			printf("\nRank %d: Observed data\n", mpirank);
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

	for (int p=0; p < mpisize; p++) {
		if (mpirank == p) {
			printf("\nRank %d: Output data\n", mpirank);
			for (size_t j=0; j < output_size; j++)
				printf("%.6f ", d[j]);
			printf("\n");
		}
	}

	double sigma = ForwardModel::compute_posterior_sigma(od, output_size, noise);
	double pos = ForwardModel::compute_posterior(od, d, output_size, sigma);

	for (int p=0; p < mpisize; p++) {
		if (mpirank == p) {
			printf("\nRank %d: sigma = %.6f, posterior = %.6f\n", mpirank, sigma, pos);
		}
	}
#endif
}

void test_sgi_mpi() {
#if (1==0)
	int mpisize, mpirank, mpistatus;
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

	string inputfile = "./input/ns_obs1.dat";

	// Create forward models
	unique_ptr<NS>  full (new NS(inputfile, 1, 1));	 /// full model
	unique_ptr<SGI> ssgi (new SGI(inputfile, "./output/ssgi_", 1, 1)); /// static sgi
	unique_ptr<SGI> asgi (new SGI(inputfile, "./output/asgi_", 1, 1)); /// adaptive sgi

	// Get problem dimensions
	std::size_t input_size = full->get_input_size();
	std::size_t output_size = full->get_output_size();

	// Initialize the true input vector
	unique_ptr<double[]> m (new double[input_size]);
	for (std::size_t i=0; i < input_size; i++)
		m[i] = true_input[i];

	// Error analysis objects
	unique_ptr<ErrorAnalysis> eas (new ErrorAnalysis(full.get(), ssgi.get()));
	eas->add_test_points(20);

	unique_ptr<ErrorAnalysis> eaa (new ErrorAnalysis(full.get(), asgi.get()));
	eaa->copy_test_points(eas.get());

	// Construct Static SGI
	ssgi->build(0.1, 2, false);
	double errs = eas->compute_model_error();

	// Construct Adaptive SGI
	double erra = 0.0, erra_old = -1.0, tol = 0.1;
	int count = 0;
	while (true) {
		asgi->build(0.1, 2, false);
		erra = eaa->compute_model_error();
		count += 1;
		if(mpirank == MASTER) {
			printf("\nRefinement # %d\n", count);
			printf("Surrogate model error: %.6f\n", erra);
		}
		if ((erra - erra_old < 0) && (fabs(erra - erra_old) < tol)) break;
		erra_old = erra;
	}
#endif
}

void test_mcmc_mh() {
#if (1==0)
	int mpisize, mpirank, mpistatus;
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

	string inputfile = "./input/ns_obs1.dat";

	// Create forward models
	unique_ptr<ForwardModel> full (new NS(inputfile, 1, 1));
	std::size_t input_size = full->get_input_size();
	std::size_t output_size = full->get_output_size();

	unique_ptr<MCMC> mcmc (new MetropolisHastings(full.get(), inputfile));

	vector<double> i1 = {9.08556, 0.0286664, 5.41189e-05};
	vector<vector<double> > inits;
	inits.push_back(i1);

	mcmc->run("./output/mcmc.dat", 20, inits);
#endif
}

void test_mcmc_pt() {
#if (1==0)
	int mpisize, mpirank, mpistatus;
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

	string inputfile = "./input/ns_obs1.dat";

	// Create forward models
	unique_ptr<ForwardModel> full (new NS(inputfile, 1, 1));
	std::size_t input_size = full->get_input_size();
	std::size_t output_size = full->get_output_size();

	// Initial samples
	vector<vector<double> > inits;
	for (int i=0; i < mpisize; i++)
		inits.push_back(vector<double> {0.394162, 0.883078, 0.11071});

	unique_ptr<MCMC> mcmc (new ParallelTempering(full.get(), inputfile, 0.5));

	mcmc->run("./output/mcmc", 20, inits);
#endif
}

void run_asgi() {
#if (1==1)
	int mpisize, mpirank, mpistatus;
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

//	string inputfile = "./input/ns_obs1.dat";
	string inputfile = "./input/ns_obs4.dat";
	string outpath = "./output/obs4_asgi2/";
	string cmd = "mkdir -p " + outpath;
	system(cmd.c_str());

	int init_level = 2;
	int nsamples = 50000;

	// Create forward models
	unique_ptr<NS>  full (new NS(inputfile, 1, 1));	 /// full model
	unique_ptr<SGI> asgi (new SGI(inputfile, outpath, 1, 1)); /// adaptive sgi

	// Get problem dimensions
	std::size_t input_size = full->get_input_size();
	std::size_t output_size = full->get_output_size();

	// Initialize the true input vector
	unique_ptr<double[]> m (new double[input_size]);
	for (std::size_t i=0; i < input_size; i++)
		m[i] = true_input[i];

	unique_ptr<ErrorAnalysis> ea (new ErrorAnalysis(full.get(), asgi.get()));
	ea->add_test_points(20);

	// ASGI
	// 1. build surrogate
//	double aerr = 0.0, aerr_old = -1.0, tol = 0.1;
//	int count = 0;
//	while (true) {
//		asgi->build(0.1, init_level, false);
//		aerr = ea->compute_model_error();
//		count += 1;
//		if(mpirank == MASTER) {
//			printf("\nRefinement # %d\n", count);
//			printf("Adaptive Surrogate model error: %.6f\n", aerr);
//		}
//		if ((aerr - aerr_old < 0) && (fabs(aerr - aerr_old) < tol)) break;
//		aerr_old = aerr;
//	}
	// 1. Or read grid
	asgi->duplicate("","","");

	// 2. run MCMC
	auto inits = asgi->get_top_maxpos(20, outpath+"grid.mpibin");
	unique_ptr<MCMC> amcmc (new ParallelTempering(asgi.get(), inputfile, 0.2));
	amcmc->run(outpath, nsamples, inits);
#endif
}

void run_ssgi() {
#if (1==0)
	int mpisize, mpirank, mpistatus;
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

	string inputfile = "./input/ns_obs1.dat";
//	string inputfile = "./input/ns_obs4.dat";
	int nsamples = 50000;

	// Create forward models
	unique_ptr<NS>  full (new NS(inputfile, 1, 1));	 /// full model
	unique_ptr<SGI> ssgi (new SGI(inputfile, "./output/ssgi_", 1, 1)); /// static sgi
	unique_ptr<SGI> asgi (new SGI(inputfile, "./output/asgi_", 1, 1)); /// adaptive sgi

	// Get problem dimensions
	std::size_t input_size = full->get_input_size();
	std::size_t output_size = full->get_output_size();

	// Initialize the true input vector
	unique_ptr<double[]> m (new double[input_size]);
	for (std::size_t i=0; i < input_size; i++)
		m[i] = true_input[i];

	// Error analysis objects
	unique_ptr<ErrorAnalysis> sea (new ErrorAnalysis(full.get(), ssgi.get()));
	sea->add_test_points(20);

	unique_ptr<ErrorAnalysis> aea (new ErrorAnalysis(full.get(), asgi.get()));
	aea->copy_test_points(sea.get());

	// SSGI
	ssgi->build(0.1, 2, false);
	double serr = sea->compute_model_error();
	if(mpirank == MASTER) {
		printf("\nStatic Surrogate model error: %.6f\n", serr);
	}
	unique_ptr<MCMC> smcmc (new ParallelTempering(ssgi.get(), inputfile, 0.5));
	smcmc->run("./output/ssgi_mcmc", nsamples);

	// ASGI
	double aerr = 0.0, aerr_old = -1.0, tol = 0.1;
	int count = 0;
	while (true) {
		asgi->build(0.1, 2, false);
		aerr = aea->compute_model_error();
		count += 1;
		if(mpirank == MASTER) {
			printf("\nRefinement # %d\n", count);
			printf("Adaptive Surrogate model error: %.6f\n", aerr);
		}
		if ((aerr - aerr_old < 0) && (fabs(aerr - aerr_old) < tol)) break;
		aerr_old = aerr;
	}
	unique_ptr<MCMC> amcmc (new ParallelTempering(asgi.get(), inputfile, 0.5));
	amcmc->run("./output/asgi_mcmc", nsamples);

#endif
}



int main(int argc, char* argv[]) {
	int mpistatus;

#if (ENABLE_IMPI==1)
	MPI_Init_adapt(&argc, &argv, &mpistatus);
#else
	MPI_Init(&argc, &argv);
#endif

	test_ns_mpi();
	test_sgi_mpi();
	test_mcmc_mh();
	test_mcmc_pt();
	run_asgi();
	run_ssgi();

#if (ENABLE_IMPI==1)
	printf("\n~~~~~~This is awesome!!~~~~~\n");
#endif

	MPI_Finalize();
	return 0;
}
