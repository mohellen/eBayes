#include "Config.hpp"

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

using namespace std;

Config::Config(int argc, char** argv)
{
	add_params();
}

void Config::add_params()
{
	string var;
	Param p;
	
	var = "input_file";
	p.des = "Input file that blah blah";
	p.val = "./input/ns.dat";
	params[var] = p;

	var = "output_mesg";
	p.des = "Output message to display";
	p.val = "It's doing something...";
	params[var] = p;
}


void Config::print_help()
{
	cout << "Available config options:\n" << endl;

	for (auto it=params.begin(); it!=params.end(); ++it) {
		cout << "\t" << it->first << " : " << it->second.des << "\n" << endl;
	}
}

string Config::get_param_value(string param){
	return params[param].val;
}

int main(int argc, char* argv[])
{
	Config cfg (argc, argv);
	//Config& cfg = Config::getInstance(argc, argv);
	
	string v = cfg.get_param_value("input_file");
	cout << v << endl;

	cfg.print_help();

	return 0;
}
