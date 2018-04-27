#include <tools/Config.hpp>
#include <tools/Parallel.hpp>
#include <tools/ErrorAnalysis.hpp>
#include <model/NS.hpp>
#include <mcmc/MCMC.hpp>
#include <mcmc/MetropolisHastings.hpp>
#include <mcmc/ParallelTempering.hpp>
#include <surrogate/SGI.hpp>

#include <iostream>
#include <vector>
#include <map>

using namespace std;

int main(int argc, char** argv)
{
	Config cfg (argc, argv);
	Parallel par;
	par.mpi_init(argc, argv);
	
	cout << tools::yellow << "Rank " << par.rank << ": status " << par.status << tools::reset << endl;

	// Forward model
	NS ns (cfg);
	// Surrogate model
	SGI sgi (cfg, par, ns);
	// Error analysis object
	ErrorAnalysis ea (cfg, par, ns, sgi);
	ea.add_test_points(5);

	double tol = 0.1;
	while(true) {
		sgi.build();
		if (ea.mpi_is_model_accurate(tol)) break;
	}

	// Test full model
	//ns.print_info();
	//std::vector<double> m {1.0, 0.8, 3.0, 1.5, 5.5, 0.2, 8.2, 1.0};
	//std::vector<double> m {1.0, 0.8};
	//std::vector<double> d = ns.run(m);
	//m.push_back( cfg.compute_posterior(d) );
	//for (double v: d)
	//	std::cout << v << "  ";
	//std::cout << std::endl;

	// MCMC
//	MCMC* mcmc = new ParallelTempering(cfg, par, sgi);
//	mcmc->run(3, sgi.get_nth_maxpos(par.rank) );

	par.mpi_final();
	return 0;
}
