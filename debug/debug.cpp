#include <tools/Config.hpp>
#include <tools/Parallel.hpp>
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
	
	cout << tools::yellow << "Rank " << par.mpirank << ": status " << par.mpistatus << tools::reset << endl;

	// Forward model
	NS ns (cfg);

	// Surrogate model
	SGI sgi (cfg, par, ns);
	sgi.build();
	auto tops = sgi.get_top_maxpos();

	cout << "Rank " << par.mpirank << ": has " << tops.size() << " top maxpos points..." << endl;

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
	MCMC* mcmc = new ParallelTempering(cfg, par, sgi);
	mcmc->run(3, sgi.get_top_maxpos_point() );

	par.mpi_final();
	return 0;
}
