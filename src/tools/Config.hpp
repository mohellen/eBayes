#ifndef TOOLS_CONFIG_HPP_
#define TOOLS_CONFIG_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <iterator>
#include <algorithm>
#include <iomanip> //for std::setprecision

// Global enum type, so that everybody knows what are the available scenarios
enum Scenario {
	NS_OBS = 1,
	HEAT_SRC = 2
};

// Global constants defined at compile time (in SConstruct)
//--- GLOBAL_SCENARIO
//--- IMPI
//--- IMPI_PRINT_NODES
//--- IMPI_MASTER_RANK
//--- SGI_OUTPUT_FREQ_IN_SEC
//--- SGI_OUTPUT_RANK_PROGRESS
//--- SGI_OUTPUT_GRIDPOINT_PROGRESS

class Config
{
public:
	~Config() {}
	Config(int argc, char** argv);
	// Disable other constructors
	//(deleted functions should be made public for better error message)
	Config() = delete;
	Config(Config const&) = delete;
	void operator=(Config const&) = delete;

	// Getters
	std::size_t get_input_size() const {return insize;}
	std::size_t get_output_size() const {return observation.size();}
	std::vector<double> get_observation() const {return observation;}
	double get_observation_noise() const {return observation_noise;}

	// Retrieve parameter value
	std::string get_param(std::string var) const {return params.at(var).val;}

private:
	std::string config_file = "";

	// Sorted list of parameters: <parameter_name> (key), <description, value> (value)
	struct Param {
		std::string des; // parameter description
		std::string val; // parameter values
	};
	std::map<std::string, Param> params;

	// Inverse problem y = f(x) dimensions: insize -> x, outsize -> y
	std::size_t insize;
	std::vector<double> observation;
	double observation_noise;

	void add_params();	// Add/define all parameters
	void parse_file();	// Parse from input file
	void parse_args(int argc, char** argv);	// Parse from command line arguments
	void print_help(int argc, char** argv);	// Print all parameters and descriptions
};


// Global helper methods that everyone can use
namespace tools
{
	//Some colors for console output using std::cout
	const std::string red("\033[0;31m"); // color for ERROR and WARNING
	const std::string green("\033[1;32m");
	const std::string yellow("\033[1;33m");
	const std::string blue("\033[1;34m");
	const std::string magenta("\033[0;35m");
	const std::string cyan("\033[0;36m");
	const std::string reset("\033[0m"); // don't forget to reset to normal after coloration
	//const std::string& colorwarn = red;
	//const std::string& colorerr = red;
	
	// trim white space from a string
	std::string trim_white_space(const std::string& str);

//	// Convert array to string
//	std::string arr_to_string(const double* m, std::size_t len);

	// Compute the l2norm of two vectors
	double compute_l2norm(
			std::vector<double> const& d1,
			std::vector<double> const& d2);

	double compute_posterior_sigma(
			std::vector<double> const& observation,
			double observation_noise);

	double compute_posterior(
			std::vector<double> const& observation,
			std::vector<double> const& data,
			double sigma);
}

#endif
