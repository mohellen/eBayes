#ifndef TOOLS_CONFIG_HPP_
#define TOOLS_CONFIG_HPP_

#include <unordered_map>
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

	void parse_args();	// Parse from command line arguments

	std::string get_param(std::string);	// Retrieve parameter value

	void print_help();	// Print all parameters and descriptions

private:
	std::string input_file = "./input/test.dat";

	struct Param {
		std::string des; // parameter description
		std::string val; // parameter values
	};
	// List of parameters: <parameter_name> (key), <description, value> (value)
	std::unordered_map<std::string, Param> params;

public:
	// Disable other constructors
	//(deleted functions should be made public for better error message)
	Config() = delete;
	Config(Config const&) = delete;
	void operator=(Config const&) = delete;
};
#endif
