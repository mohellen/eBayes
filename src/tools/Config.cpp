#include "Config.hpp"


using namespace std;

Config::Config(int argc, char** argv)
{
	// Parameter value take in order: commond line argument > input file > default value
	// ">" means orderride.
	add_params();
	parse_file();
	parse_args(argc, argv);
}

void Config::add_params()
{
	string var;
	Param p;
	
	// Global setting
	var = "global_is_resume_job";
	p.des = "Set to create grid from grid files in specified path. (Default: no) (Options: yes|true|no|false)";
	p.val = "no";
	params[var] = p;

	var = "global_resume_from_path";
	p.des = "Directory containing the grid files to create grid for a resumed job. (Default: ) (Type: string)";
	p.val = "/my/resumed/path";
	params[var] = p;

	var = "global_output_path";
	p.des = "Path for resulting grid files and other outputs. (Default: ./output) (Type: string)";
	p.val = "./output";
	params[var] = p;

	var = "global_noise_in_data";
	p.des = "Assumed noise level in observed data [0.0, 1.0], 0 for no noise, 1 for 100\% noise. (Default: 0.2) (Type: double)";
	p.val = "0.2";
	params[var] = p;

	var = "global_observed_data";
	p.des = "Observed data. (Default: <obs4_data_set>) (Note: provide vector in one line separated by space)";
	p.val = "1.434041 1.375464 1.402000 0.234050 1.387931 1.006520 1.850871 1.545131 1.563303 0.973778 1.512808 1.387468 1.608557 0.141381 1.313631 0.990608 1.741001 1.551365 1.789867 1.170761  1.597586 1.509048 1.549320 0.135403 1.191323 1.015913 1.682937 1.592488 1.743632 1.296677 1.535493 1.341702 1.541945 0.137985 1.272473 1.041918 1.824279 1.690430 1.810520 1.358992";
	params[var] = p;

	// iMPI setting
	var = "impi_adapt_freq_sec";
	p.des = "iMPI adapt frequency in seconds: call probe_adapt every N seconds. (Default: 60) (Type: positive integer)";
	p.val = "60";
	params[var] = p;

	// Surrogate SGI setting
	var = "sgi_is_masterworker";
	p.des = "SGI construction style, enable to use Master-Worker (iMPI or MPI), disable to use SIMD (MPI only). (Default: yes) (Options: yes|true|no|false)";
	p.val = "yes";
	params[var] = p;

	var = "sgi_masterworker_jobsize";
	p.des = "For SGI construction Master-Worker style: # of grid points to compute in a job. (Default: 10) (Type: positive integer)";
	p.val = "10";
	params[var] = p;

	// MCMC setting
	var = "mcmc_is_progress";
	p.des = "Enable to output MCMC progress. (Default: no) (Options: yes|true|no|false)";
	p.val = "no";
	params[var] = p;

	var = "mcmc_progress_freq_step";
	p.des = "MCMC Progress output frequency: print progress every N MCMC steps. (Default: 2000) (Type: positive integer)";
	p.val = "2000";
	params[var] = p;

	var = "mcmc_num_chains";
	p.des = "Number of MCMC chains to be used for Parallel Tempering. (Default: 20) (Type: positive integer)";
	p.val = "20";
	params[var] = p;

	// Model NS setting
	var = "ns_domain_size_x";
	p.des = "Domain size in meters x-direction. (Default: 10.0) (Type: double)";
	p.val = "10.0";
	params[var] = p;

	var = "ns_domain_size_y";
	p.des = "Domain size in meters y-direction. (Default: 2.0) (Type: double)";
	p.val = "2.0";
	params[var] = p;

	var = "ns_min_ncx";
	p.des = "Miminum number of cells in x-direction. (Default: 100) (Type: positive integer)";
	p.val = "100";
	params[var] = p;

	var = "ns_min_ncy";
	p.des = "Miminum number of cells in y-direction. (Default: 20) (Type: positive integer)";
	p.val = "20";
	params[var] = p;

	var = "ns_initial_velocity_x";
	p.des = "Initial fluid velocity in x-direction. (Default: 1.0) (Type: double)";
	p.val = "1.0";
	params[var] = p;

	var = "ns_initial_velocity_y";
	p.des = "Initial fluid velocity in y-direction. (Default: 0.0) (Type: double)";
	p.val = "0.0";
	params[var] = p;

	var = "ns_initial_pressure";
	p.des = "Initial fluid pressure. (Default: 0.0) (Type: double)";
	p.val = "0.0";
	params[var] = p;

	var = "ns_inlet_velocity_x";
	p.des = "Fluid inlet velocity in x-direction. (Default: 1.0) (Type: double)";
	p.val = "1.0";
	params[var] = p;

	var = "ns_inlet_velocity_y";
	p.des = "Fluid inlet velocity in x-direction. (Default: 0.0) (Type: double)";
	p.val = "0.0";
	params[var] = p;

	var = "ns_external_force_x";
	p.des = "External force applied to fluid x-direction. (Default: 0.0) (Type: double)";
	p.val = "0.0";
	params[var] = p;

	var = "ns_external_force_y";
	p.des = "External force applied to fluid y-direction. (Default: 0.0) (Type: double)";
	p.val = "0.0";
	params[var] = p;

	var = "ns_re";
	p.des = "Reynolds number. (Default: 100.0) (Type: double)";
	p.val = "100.0";
	params[var] = p;

	var = "ns_tau";
	p.des = "Time step stability factor. (Default: 0.5) (Type: double)";
	p.val = "0.5";
	params[var] = p;

	var = "ns_alpha";
	p.des = "Upwind differencing factor. (Default: 0.9) (Type: double)";
	p.val = "0.9";
	params[var] = p;

	var = "ns_omega";
	p.des = "Pressure related. (Default: 1.0) (Type: double)";
	p.val = "1.0";
	params[var] = p;

	var = "ns_boundary_north";
	p.des = "North boundary type. (Default: noslip) (Options: inlet|outlet|noslip|freeslip)";
	p.val = "noslip";
	params[var] = p;

	var = "ns_boundary_south";
	p.des = "South boundary type. (Default: noslip) (Options: inlet|outlet|noslip|freeslip)";
	p.val = "noslip";
	params[var] = p;

	var = "ns_boundary_east";
	p.des = "East boundary type. (Default: outlet) (Options: inlet|outlet|noslip|freeslip)";
	p.val = "outlet";
	params[var] = p;

	var = "ns_boundary_west";
	p.des = "West boundary type. (Default: inlet) (Options: inlet|outlet|noslip|freeslip)";
	p.val = "inlet";
	params[var] = p;

	var = "ns_obstacle_list";
	p.des = "List of obstacles: obs1_locx, obs1_locy, obs1_sizex, obs1_sizey, obs2_locx, ... (Default: <list of 4 obstacles>) (Note: provide vector in one line separated by space)";
	p.val = "1.0 0.8 0.4 0.4	3.0 1.5 0.4 0.4		5.5 0.2 0.4 0.4		8.2 1.0 0.4 0.4";
	params[var] = p;

	var = "ns_output_time_list";
	p.des = "Simulation output sampiling time instances. (Default: <4 time instances>) (Note: provide vector in one line separated by space)";
	p.val = "2.5 5.0 7.5 10.0";
	params[var] = p;

	var = "ns_output_location_list";
	p.des = "Simulation output sampling locations: loc1x, loc1y, loc2x, loc2y, .... (Default: <10 locations>) (Note: provide vector in one line separated by space)";
	p.val = "1.5 0.6	 3.1 0.6	4.7 0.6		6.3 0.6		7.9 0.6		1.5 1.3		3.1 1.3		4.7 1.3		6.3 1.3		7.9 1.3";
	params[var] = p;
}

void Config::parse_file()
{
	ifstream f(input_file);
	if (!f) cout << tools::colorwarn << "\nWARNING: cannot open input file.\n" << tools::reset << endl;

	string s;
	while (std::getline(f, s)) {
		// Read line into string stream
		istringstream iss(s);
		vector<string> tokens {istream_iterator<string>{iss}, istream_iterator<string>{}};
		// Ignore empty lines, or lines with empty value (only parameter, no value)
		if (tokens.size() < 2) continue;
		// Ignore comment lines (start with // or #)
		tokens[0] = tools::trim_white_space(tokens[0]);
		if ((tokens[0].substr(0,2) == "//") || (tokens[0].substr(0,1) == "#")) continue;

		// For a valid line:
		// Transform parameter name to lower case string
		transform(tokens[0].begin(), tokens[0].end(), tokens[0].begin(), ::tolower);
		// Ignore line if invalid parameters (key not found in params)
		if (params.find(tokens[0]) == params.end()) continue;
	
		// For vector type parameters
		if (tokens.size() > 2) {
			params[tokens[0]].val = "";
			for (auto it=tokens.begin()+1; it != tokens.end(); ++it) {
				params[tokens[0]].val += *it;
				params[tokens[0]].val += " ";
			}
			continue;
		// For single-value type parameters
		} else {
			params[tokens[0]].val = tokens[1];
			continue;
		}
	}//end while
	f.close();
}

void Config::parse_args(int argc, char** argv)
{
}

void Config::print_help()
{
	cout << "Available config options:\n" << endl;
	for (auto it=params.begin(); it!=params.end(); ++it) {
		cout << "   " << it->first;
		cout << "\n   \t>> Description: " << it->second.des;
		cout << "\n   \t>> Current value: " << it->second.val << "\n" << endl;
	}
}

string Config::get_param(string var)
{
	return params[var].val;
}

string tools::trim_white_space(const string& str)
{
	std::string whitespace=" \t";
	const auto strBegin = str.find_first_not_of(whitespace);
	if (strBegin == std::string::npos) return "";
	const auto strEnd = str.find_last_not_of(whitespace);
	const auto strRange = strEnd - strBegin + 1;
	return str.substr(strBegin, strRange);
}


int main(int argc, char* argv[])
{
	Config cfg (argc, argv);

	cfg.print_help();

	string v = cfg.get_param("observed_data");
	cout << "\n\n" << v << endl;

	v = cfg.get_param("obstacle_list");
	cout << "\n\n" << v << endl;

	return 0;
}
