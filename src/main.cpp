#include <tools/Config.hpp>
#include <tools/Parallel.hpp>
#include <tools/ErrorAnalysis.hpp>
#include <model/NS.hpp>
#include <mcmc/MCMC.hpp>
#include <mcmc/MetropolisHastings.hpp>
#include <mcmc/ParallelTempering.hpp>
#include <surrogate/SGI.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace std;

const int ITERMAX = 10; // Cap the # of grid refinement (to prevent infinite loop)

int main(int argc, char** argv)
{
	Config cfg (argc, argv);
	Parallel par;
	par.mpi_init(argc, argv);

	double tic = MPI_Wtime();
	if (par.is_master()) {
		printf("\nMain: BEGIN wall.time(sec) %.6f\n", tic);
	}

	// Forward model
	NS ns (cfg);
	// Surrogate modez
	SGI sgi (cfg, par, ns);
	// Error analysis object
	ErrorAnalysis ea (cfg, par, ns, sgi);
	// Only Master need test points
	if (par.is_master()) {
		ea.add_test_points(cfg.get_param_sizet("sgi_masterworker_jobsize"));
		//ea.read_test_points(cfg.get_param_string("sgi_resume_path") + "/test_points_r" + std::to_string(par.rank) + ".txt");
		ea.write_test_points(cfg.get_param_string("global_output_path") + "/test_points_r" + std::to_string(par.rank) + ".txt");
		ea.print_test_points();
	}

	double tol = cfg.get_param_double("sgi_tol");
	for (int iter = 0; iter < ITERMAX; ++iter) {

		if (par.is_master()) {
			printf("\nMain: SGI phase %d | wall.time(sec) %.6f\n", iter, MPI_Wtime()-tic);
		}

		sgi.build();
		if (ea.eval_model_master(tol)) break;
	}

	if (par.is_master()) {
		printf("Main: MCMC phase wall.time(sec) %.6f\n", MPI_Wtime()-tic);
	}
	// MCMC
	MCMC* mcmc = new ParallelTempering(cfg, par, sgi);
	mcmc->run(cfg.get_param_sizet("mcmc_num_samples"), sgi.get_maxpos() );

	if (par.is_master()) {
		printf("Main: END wall.time(sec) %.6f\n", MPI_Wtime()-tic);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	par.mpi_final();
	return 0;
}
