#include <tools/Config.hpp>
#include <tools/Parallel.hpp>
#include <model/NS.hpp>
#include <mcmc/MCMC.hpp>
#include <mcmc/MetropolisHastings.hpp>

#include <iostream>
#include <vector>


int main(int argc, char** argv)
{
	Config cfg (argc, argv);
	Parallel par;
	par.mpi_init(argc, argv);

	// Objects
	NS ns (cfg);

	// MCMC
	MCMC* mcmc = new MetropolisHastings(cfg, par, ns);

#if (1==0)
	// Test full model
	ns.print_info();
	std::vector<double> m {1.0, 0.8, 3.0, 1.5, 5.5, 0.2, 8.2, 1.0};
	std::vector<double> d = ns.run(m);
	for (double v: d)
		std::cout << v << "  ";
	std::cout << std::endl;
#endif

#if (1==1)
	// Test MCMC
	mcmc->run(10);
#endif

	par.mpi_final();
	return 0;
}
