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
#include <cstdlib> //for system()

// Global enum type, so that everybody knows what are the available scenarios
enum Scenario {
	NS_OBS = 1,
	HEAT_SRC = 2
};


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
	// Retrieve parameter value
	std::string get_param_string(std::string var) const {return params.at(var).val;}
	double get_param_double(std::string var) const {return stod(params.at(var).val);}
	std::size_t get_param_sizet(std::string var) const {return std::size_t(stoul(params.at(var).val));}
	bool get_param_bool(std::string var) const {
		return (params.at(var).val == "yes") ? true : false; }
	
	// Compute the posteria for a given simulation data
	double compute_posterior(std::vector<double> const& data) const;

private:
	std::string config_file = ""; // Config_file is optional

	// Sorted list of parameters: <parameter_name> (key), <description, value> (value)
	struct Param {
		std::string des; // parameter description
		std::string val; // parameter values
	};
	std::map<std::string, Param> params;

	// Inverse problem y = f(x) dimensions: insize -> x, outsize -> y
	std::size_t insize;
	std::vector<double> observation;
	double observation_sigma; // sigma := noise * mean(observation)

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
	std::string trim_white_space(std::string const& str);

	// for easy printing sample (input vector) and samplepos (input vector + posterior)
	std::string sample_to_string(std::vector<double> const& v);
	std::string samplepos_to_string(std::vector<double> const& v);

	// Compute the l2norm of two vectors
	double compute_l2norm(
			std::vector<double> const& d1,
			std::vector<double> const& d2);
}

#endif
