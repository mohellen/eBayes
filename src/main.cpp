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


#include "config.h"
#include "ForwardModel.hpp"
#include "mpi/MPIObject.hpp"
#include "model/NS.hpp"
#include "surrogate/SGIDist.hpp"
#include "mcmc/MHMCMC.hpp"
#include "mcmc/PTMCMCDist.hpp"
#include "tools/io.hpp"
#include "tools/matrix.hpp"
#include "sim/NSSim.hpp"
#include "sim/NSSimDist.hpp"
#include "sim/DomainDecomposer2D.hpp"

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <sys/time.h>


using namespace std;


/**
 * Shared objects across all methods
 **/
#if (NS_NUM_OBS == 1)
	double m_true[] = {1.0, 0.8};
#elif (NS_NUM_OBS == 2)
	double m_true[] = {1.0, 0.8, 3.0, 1.5};
#elif (NS_NUM_OBS == 3)
	double m_true[] = {1.0, 0.8, 3.0, 1.5, 8.2, 1.0};
#elif (NS_NUM_OBS == 4)
	double m_true[] = {1.0, 0.2, 3.0, 1.5, 5.5, 0.2, 8.2, 1.0};
#endif

MPIObject* mpi;
FullModel* fm;
SGIDist* sm;
size_t param_size;
size_t data_size;
double sigma;

// Simulation resolution (full/sgi model)
size_t ncx = NS_NCX;
size_t ncy = NS_NCY;

/*****************************************/
double double_gettimeofday(){
  struct timeval start_tv;
  gettimeofday(&start_tv, NULL);
  return (double)(start_tv.tv_sec) +
    ((double)(start_tv.tv_usec)/1000000.0);
}

double data_euclidean_distance(
		size_t len, const double* d1, const double* d2)
{
	double tmp = 0;
	for (std::size_t i=0; i < len; i++)
		tmp += (d1[i] - d2[i])*(d1[i] - d2[i]);
	return sqrt(tmp);
}

void test_sim()
{
	cout << "Test full model simulation with vtk output..."
			<< "This should be run serially." << endl;

	unique_ptr<double[]> d (new double[data_size]);

	double tic = double_gettimeofday();
	fm->run(m_true, d.get());
	double toc = double_gettimeofday() - tic;

	cout << "Single simulation "<< ncx << "x" << ncy << " run time = "
			<< toc << endl;
}

void test_sgi_mpi()
{
	double pos_fm, pos_sm;
	unique_ptr<double[]> d_fm (new double[data_size]);
	unique_ptr<double[]> d_sm (new double[data_size]);

	fm->run(m_true, d_fm.get());
	pos_fm = fm->compute_posterior(sigma, d_fm.get());

	sm->initialize(4, "mm");
	sm->run(m_true, d_sm.get());

	pos_sm = sm->compute_posterior(sigma, d_sm.get());

	if (mpi->rank == MASTER) {

		cout << arr_to_string(data_size, d_sm.get()) << endl;

		cout << "\nREAL point test" << endl;
		cout << "Full Model posterior:\t" << pos_fm << endl;
		cout << "Initial sgi posterior:\t" << pos_sm << endl;
		cout << "Initial sgi result quality:\t" <<
				data_euclidean_distance(data_size, d_fm.get(), d_sm.get()) << endl;
	}

	sm->refine(0.1, "mm");
	sm->run(m_true, d_sm.get());
	pos_sm = sm->compute_posterior(sigma, d_sm.get());

	if (mpi->rank == MASTER) {
		cout << "\nRank " << mpi->rank << ": REAL point test" << endl;
		cout << "Full Model posterior:\t" << pos_fm << endl;
		cout << "1st refined sgi posterior:\t" << pos_sm << endl;
		cout << "1st refined sgi result quality:\t" <<
				data_euclidean_distance(data_size, d_fm.get(), d_sm.get()) << endl;
	}

	sm->refine(0.1, "mm");
	sm->run(m_true, d_sm.get());
	pos_sm = sm->compute_posterior(sigma, d_sm.get());

	if (mpi->rank == MASTER) {
		cout << "\nRank " << mpi->rank << ": REAL point test" << endl;
		cout << "Full Model posterior:\t" << pos_fm << endl;
		cout << "2nd refined sgi posterior:\t" << pos_sm << endl;
		cout << "2nd refined sgi result quality:\t" <<
				data_euclidean_distance(data_size, d_fm.get(), d_sm.get()) << endl;
	}

	sm->refine(0.1, "mm");
	sm->run(m_true, d_sm.get());
	pos_sm = sm->compute_posterior(sigma, d_sm.get());

	if (mpi->rank == MASTER) {
		cout << "\nRank " << mpi->rank << ": REAL point test" << endl;
		cout << "Full Model posterior:\t" << pos_fm << endl;
		cout << "3rd refined sgi posterior:\t" << pos_sm << endl;
		cout << "3rd refined sgi result quality:\t" <<
				data_euclidean_distance(data_size, d_fm.get(), d_sm.get()) << endl;
	}
}

void test_ptmcmcdist()
{
	PTMCMCDist* mcmc = new PTMCMCDist(mpi, fm, 0.5);

	unique_ptr<double[]> m_init (new double[param_size]);
	unique_ptr<double[]> m_maxpos (new double[param_size]);

	mcmc->sample(20, m_init.get(), m_maxpos.get());
}

void test_mhmcmc()
{
	MHMCMC* mcmc = new MHMCMC(fm);

	unique_ptr<double[]> m_maxpos (new double[param_size]);

	mcmc->sample(20, m_true, m_maxpos.get());
}

void test_nssim()
{
	NSSim* sim = new NSSim(135,75);
	sim->run();
}

void test_nssimdist()
{
	NSSimDist* sim = new NSSimDist(135,75,mpi);
//	sim->run();
}

void test_genblocks()
{
	int nblocks = 16;
	for (int i=0; i < nblocks; i++) {
		unique_ptr<DomainBlock> b (DomainDecomposer2D::gen_block(nblocks, 1, 27, 1, 15, i));

		cout << b->toString() << endl;
	}
}

void static_sgi_run()
{
	/********* Offline phase: construct surrogate model *********/
	int level = 8;
	sm->initialize(level, "na");

	if (mpi->rank == MASTER) {
		double pos_fm, pos_sm;
		unique_ptr<double[]> d_fm (new double[data_size]);
		unique_ptr<double[]> d_sm (new double[data_size]);

		fm->run(m_true, d_fm.get());
		pos_fm = fm->compute_posterior(sigma, d_fm.get());

		sm->run(m_true, d_sm.get());
		pos_sm = sm->compute_posterior(sigma, d_sm.get());

		cout << "\nStatic SGI surrogate model with level=" << level << endl;
		cout << "Full Model posterior of true point:\t" << pos_fm << endl;
		cout << "Static Sgi posterior of true point:\t" << pos_sm << endl;
		cout << "Static Sgi model quality:\t" <<
				data_euclidean_distance(data_size, d_fm.get(), d_sm.get()) << endl;
	}
	MPI_Barrier(mpi->comm);

	/********* Online phase: run MCMC sampler *********/
	// Create an MPI object with 10 ranks only
	if ((mpi->size < 10) && (mpi->rank == MASTER)) {
		cout << "Minimum ranks requirement not met. Program abort!" << endl;
		exit(EXIT_FAILURE);
	}
	MPIObject* mpi10 = new MPIObject();
	int color;
	if (mpi->rank < 10)
		color = 10;
	else
		color = 20;
	MPI_Comm_split(MPI_COMM_WORLD, color, mpi->rank, &(mpi10->comm));
	mpi10->update(mpi10->comm);

	// Only first 10 ranks run MCMC
	if (mpi->rank < 10) {
		PTMCMCDist* mcmc = new PTMCMCDist(mpi10, sm, 0.5);

		// Get the current maxpos point from the Sgi model
		unique_ptr<double[]> m_init (new double[param_size]);
		unique_ptr<double[]> m_maxpos (new double[param_size]);
		sm->get_maxpos_point(m_init.get());

		mcmc->sample(20000, m_init.get(), m_maxpos.get());
	}
}

void nssimdist_impi()
{
#if (ENABLE_IMPI == YES)
	NSSimDist* sim = new NSSimDist(135,75,mpi);

	if (mpi->status == MPI_ADAPT_STATUS_JOINING) {
		sim->impi_adapt();
	}

	sim->run();
	return;
#endif
}

void sgi_impi()
{
#if (ENABLE_IMPI == YES)
	/*** Initialization for ALL ***/
	double pos_fm, pos_sm;
	unique_ptr<double[]> d_fm (new double[data_size]);
	unique_ptr<double[]> d_sm (new double[data_size]);

	fm->run(m_true, d_fm.get());
	pos_fm = fm->compute_posterior(sigma, d_fm.get());

	/*** New PEs vs. Joining PEs ***/
	// From MPI_Init_adapt(), there can be only NEW or JOINING status
	if (mpi->status == MPI_ADAPT_STATUS_JOINING)
		sm->impi_adapt();
	else
		sm->CarryOver.phase = 0;

	int num_phases = 4;
	while(true) {

		if (sm->CarryOver.phase == 0)
			sm->initialize(NS_SGI_INIT_LEVEL, "mm");
		else
			sm->refine(0.1, "mm");

		sm->run(m_true, d_sm.get());
		pos_sm = sm->compute_posterior(sigma, d_sm.get());

		// output
		if (mpi->rank == MASTER) {
			cout << "***** PHASE " << sm->CarryOver.phase << " *****" << endl;
			cout << "Full Model posterior:\t" << pos_fm << endl;
			cout << "SGI Model posterior:\t" << pos_sm << endl;
			cout << "SGI quality:\t"
				<< data_euclidean_distance(data_size, d_fm.get(), d_sm.get()) << endl;
		}
		sm->CarryOver.phase ++ ;
		if (sm->CarryOver.phase >= num_phases) break;
	}
	return;
#endif
}

int main(int argc, char* argv[]) {

/************** MUST-HAVEs **************/
	mpi = new MPIObject();
#if (ENABLE_IMPI == YES)
	double tic = double_gettimeofday();
	MPI_Init_adapt(&argc, &argv, &(mpi->status));
	double toc = double_gettimeofday() - tic;
	cout << "Rank " << mpi->rank << " [STATUS " << mpi->status << "]: "
			<< "MPI_Init_adapt " << toc << " seconds" << endl;
#else
	MPI_Init(&argc, &argv);
#endif
	mpi->update(MPI_COMM_WORLD);

	fm = new NS(ncx,ncy);
	sm = new SGIDist(fm, mpi);
	param_size = fm->get_param_size();
	data_size = fm->get_data_size();
	sigma = fm->compute_posterior_sigma();
/****************************************/

	test_sim();
//	test_sgi_mpi();
//	test_ptmcmcdist();
//	test_mhmcmc();
//	test_nssim();
//	test_nssimdist();
//	test_genblocks();
//	static_sgi_run();

//	sgi_impi();

//	nssimdist_impi();

	MPI_Finalize();
	return 0;
}
