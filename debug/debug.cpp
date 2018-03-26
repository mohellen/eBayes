#include <tools/Config.hpp>
#include <tools/Parallel.hpp>
#include <model/NS.hpp>
#include <mcmc/MCMC.hpp>
#include <mcmc/MetropolisHastings.hpp>

#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char** argv)
{
	Config cfg (argc, argv);
	Parallel par;
	par.mpi_init(argc, argv);

	// Objects
	NS ns (cfg);

	// MCMC
	MCMC* mcmc = new MetropolisHastings(cfg, par, ns);

	// Test full model
	ns.print_info();
	std::vector<double> m {1.0, 0.8, 3.0, 1.5, 5.5, 0.2, 8.2, 1.0};
	std::vector<double> d = ns.run(m);
	m.push_back( cfg.compute_posterior(d) );
	//for (double v: d)
	//	std::cout << v << "  ";
	//std::cout << std::endl;

	// Test MCMC
	size_t num_samples = size_t(stoul(cfg.get_param("mcmc_num_samples")));
	cout << "num_samples = " << num_samples << endl;

	mcmc->run(4, m);

	par.mpi_final();
	return 0;
}
