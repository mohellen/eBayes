#ifndef TOOLS_CONFIG_HPP_
#define TOOLS_CONFIG_HPP_

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>


class Config {
public:
	~Config() {}
	Config(int argc, char** argv);
	void add_params();	// Add/define all parameters
	void parse_file();	// Parse from input file
	void parse_args(int argc, char** argv);	// Parse from command line arguments
	std::string get_param(std::string);	// Retrieve parameter value
	void print_help();	// Print all parameters and descriptions

private:
	std::string input_file = "./input/test.dat";

	struct Param {
		std::string des; // parameter description
		std::string val; // parameter values
	};
	// Sorted list of parameters: <parameter_name> (key), <description, value> (value)
	std::map<std::string, Param> params;

public:
	// Disable other constructors
	//(deleted functions should be made public for better error message)
	Config() = delete;
	Config(Config const&) = delete;
	void operator=(Config const&) = delete;
};


// Global helper methods that everyone can use
namespace tools {
	//Some colors for console output using std::cout
	const std::string red("\033[0;31m"); // color for ERROR and WARNING
	const std::string green("\033[1;32m");
	const std::string yellow("\033[1;33m");
	const std::string blue("\033[1;34m");
	const std::string magenta("\033[0;35m");
	const std::string cyan("\033[0;36m");
	const std::string reset("\033[0m"); // don't forget to reset to normal after coloration
	const std::string& colorwarn = red;
	const std::string& colorerr = red;
	
	// trim white space from a string
	std::string trim_white_space(const std::string& str);
};

#endif
