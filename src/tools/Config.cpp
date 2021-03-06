#include <tools/Config.hpp>


using namespace std;

Config::Config(int argc, char** argv)
{
	// Parameter value take in order: command line argument > input file > default value
	// ">" means orderride.
	// Define parameters
	add_params();
	// Parse config file (if provided)
	for (int i=0; i < argc; i++) {
		if ((string(argv[i]) == "configfile") && (i+1 < argc)) {
			config_file = string(argv[i+1]);
			break;
		}
	}
	if (config_file.length() > 0) parse_file();
	// Parse command line arguments
	parse_args(argc, argv);
	// Print help if specified
	print_help(argc, argv);

	// Determine input size
	insize = get_param_sizet("global_input_size");

	// Get observation
	istringstream iss(get_param_string("global_observation"));
	vector<string> tokens {istream_iterator<string>{iss}, istream_iterator<string>{}};
	observation.reserve(tokens.size());
	for (auto it=tokens.begin(); it != tokens.end(); ++it) {
		observation.push_back(stod(*it));
	}
	// Compute observation sigma := noise * mean(observation)
	double noise = get_param_double("global_observation_noise");
	double mean = 0.0;
	for (double d: observation)
		mean += d;
	mean /= observation.size();
	observation_sigma = noise * mean;

	// Make output dir if not exists
	string f, cmd;
	cmd = "mkdir -p " + get_param_string("global_output_path");
	system(cmd.c_str());

	// Check file/path and issue warnings
	f = get_param_string("ea_test_point_file");
	if (f != "") {
		cmd = "if [ -f " + f + " ]; then echo yes; else echo no; fi";
		if (!tools::exec(cmd.c_str()).compare("yes")) {
			cout << "WARNING: ea_test_point_file is not found! ErrorAnalysis test points will be generated instead." << endl;
			params.at("ea_test_point_file").val = "";
		}
	}
	f = get_param_string("sgi_resume_path");
	if (f != "") {
		cmd = "if [ -f " + get_grid_resume_fname() + " ]; then echo yes; else echo no; fi";
		if (!tools::exec(cmd.c_str()).compare("yes")) {
			cout << "WARNING: grid file cannot be found in resume path! SGI surrogate will be built from scratch instead." << endl;
			params.at("sgi_resume_path").val = "";
		}
	}
	return;
}

void Config::print_config() const
{
	cout << "\n==============================" << endl;
	cout << "Current configurations:" << endl;
	for (auto it=params.begin(); it!=params.end(); ++it) {
		cout << "   " << it->first;
		cout << "\n   \t>> Description: " << it->second.des;
		cout << "\n   \t>> Current value: " << it->second.val << "\n" << endl;
	}
	cout << "==============================\n" << endl;
}

double Config::compute_posterior(std::vector<double> const& data) const
{
	if (observation.size() != data.size()) {
		fflush(NULL);
		printf("ERROR: vectors size mismatch. Program abort!\n");
		exit(EXIT_FAILURE);
	}
	double sum = 0.0;
	for (int i=0; i < data.size(); ++i)
		sum += (data[i] - observation[i])*(data[i] - observation[i]);
	return exp(-0.5 * sum / (observation_sigma*observation_sigma));
}

void Config::add_params()
{
	string var;
	Param p;
	
	// Global setting
	var = "global_output_path";
	p.des = "Path for resulting grid files and other outputs. (Default: ./output) (Type: string)";
	p.val = "./output";
	params[var] = p;

	var = "global_input_size";
	p.des = "Input size for the forward model. (Default: 8) (Type: size_t)";
	p.val = "8";
	params[var] = p;

	var = "global_observation";
	p.des = "Observed data. (Default: <obs4_data_set>) (Note: provide vector in one line separated by space)";
	p.val = "1.434041 1.375464 1.402000 0.234050 1.387931 1.006520 1.850871 1.545131 1.563303 0.973778 1.512808 1.387468 1.608557 0.141381 1.313631 0.990608 1.741001 1.551365 1.789867 1.170761  1.597586 1.509048 1.549320 0.135403 1.191323 1.015913 1.682937 1.592488 1.743632 1.296677 1.535493 1.341702 1.541945 0.137985 1.272473 1.041918 1.824279 1.690430 1.810520 1.358992";
	params[var] = p;

	var = "global_observation_noise";
	p.des = "Assumed noise level in observed data [0.0, 1.0], 0 for no noise, 1 for 100% noise. (Default: 0.2) (Type: double)";
	p.val = "0.2";
	params[var] = p;

	// iMPI setting
	var = "impi_adapt_freq_sec";
	p.des = "iMPI adapt frequency in seconds: call probe_adapt every N seconds. (Default: 60) (Type: size_t)";
	p.val = "60";
	params[var] = p;

	// ErrorAnalysis setting
	var = "ea_test_point_file";
	p.des = "If this file is provided, ErrorAnalysis test points will be loaded from file instead of being generated randomly. (Default: "") (Type: string)";
	p.val = "";
	params[var] = p;
	
	var = "ea_num_test_points";
	p.des = "Number of test points to be generated, only applicable when test point file is not provided. (Default: 50) (Type: size_t)";
	p.val = "50";
	params[var] = p;
	
	// Surrogate SGI setting
	var = "sgi_build_itermin";
	p.des = "Minimum SGI build/refine iterations. (Default: 1) (Type: size_t)";
	p.val = "1";
	params[var] = p;
	
	var = "sgi_build_itermax";
	p.des = "Maximum SGI build/refine iterations. (Default: 4) (Type: size_t)";
	p.val = "4";
	params[var] = p;
	
	var = "sgi_tol";
	p.des = "SGI surrogate model error tolerance in [0, 1.0]. (Default: 0.08) (Type: double)";
	p.val = "0.08";
	params[var] = p;
	
	var = "sgi_resume_path";
	p.des = "If resume path is provided and exist, SGI model will be built from the grid/data files in the path. Otherwise, SGI is built from scratch. (Default: "") (Type: string)";
	p.val = "";
	params[var] = p;

	var = "sgi_init_level";
	p.des = "Grid construction initial level (before any grid refinement). (Default: 4) (Type: size_t)";
	p.val = "4";
	params[var] = p;

	var = "sgi_refine_portion";
	p.des = "Grid refinment portion (how many % grid points should be refined). (Default: 0.1) (Type: double in [0.0, 1.0])";
	p.val = "0.1";
	params[var] = p;

	var = "sgi_is_masterworker";
	p.des = "SGI construction style, enable to use Master-Worker (iMPI or MPI), disable to use SIMD (MPI only). (Default: yes) (Options: yes|no)";
	p.val = "yes";
	params[var] = p;

	var = "sgi_masterworker_jobsize";
	p.des = "For SGI construction Master-Worker style: # of grid points to compute in a job. (Default: 10) (Type: size_t)";
	p.val = "10";
	params[var] = p;

	// MCMC setting
	var = "mcmc_num_samples";
	p.des = "Number of samples to draw using the MCMC solver. (Default: 20000) (Type: size_t)";
	p.val = "20000";
	params[var] = p;

	var = "mcmc_randwalk_step";
	p.des = "MCMC use a random walk step = X * domain size (X in [0.0, 1.0]). (Default: 0.1) (Type: double in [0.0, 1.0])";
	p.val = "0.1";
	params[var] = p;

	var = "mcmc_is_progress";
	p.des = "Enable to output detail MCMC progress including chain exchange. (Default: no) (Options: yes|no)";
	p.val = "no";
	params[var] = p;

	var = "mcmc_progress_freq_step";
	p.des = "MCMC Progress output frequency: print progress every N MCMC steps. (Default: 1000) (Type: size_t)";
	p.val = "1000";
	params[var] = p;

	var = "mcmc_max_chains";
	p.des = "Maximum number of MCMC chains to be used for Parallel Tempering. Actual number of chains = min(num_mpi_ranks, mcmc_max_chains). (Default: 20) (Type: size_t)";
	p.val = "20";
	params[var] = p;

	var = "mcmc_chain_mixing_rate";
	p.des = "For Parallel Tempering only: how frequent (percentage in [0.0, 1.0]) to mix chains. (Default: 0.2) (Type: double in [0.0, 1.0])";
	p.val = "0.2";
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
	p.des = "Miminum number of cells in x-direction. (Default: 100) (Type: size_t)";
	p.val = "100";
	params[var] = p;

	var = "ns_min_ncy";
	p.des = "Miminum number of cells in y-direction. (Default: 20) (Type: size_t)";
	p.val = "20";
	params[var] = p;

	var = "ns_resx";
	p.des = "Resolution multiplier. Actual resolution ncx = ns_resx * ns_min_ncx. (Default: 1) (Type: size_t)";
	p.val = "1";
	params[var] = p;

	var = "ns_resy";
	p.des = "Resolution multiplier. Actual resolution ncy = ns_resy * ns_min_ncy. (Default: 1) (Type: size_t)";
	p.val = "1";
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
	p.des = "North boundary type. (Default: noslip) (Type: string. Options: inlet|outlet|noslip|freeslip)";
	p.val = "noslip";
	params[var] = p;

	var = "ns_boundary_south";
	p.des = "South boundary type. (Default: noslip) (Type: string. Options: inlet|outlet|noslip|freeslip)";
	p.val = "noslip";
	params[var] = p;

	var = "ns_boundary_east";
	p.des = "East boundary type. (Default: outlet) (Type: string. Options: inlet|outlet|noslip|freeslip)";
	p.val = "outlet";
	params[var] = p;

	var = "ns_boundary_west";
	p.des = "West boundary type. (Default: inlet) (Type: string. Options: inlet|outlet|noslip|freeslip)";
	p.val = "inlet";
	params[var] = p;

	var = "ns_obs_sizes";
	p.des = "Sizes of obstacles (Default: 0.4 0.4  0.4 0.4  0.4 0.4  0.4 0.4) (Note: provide vector in one line separated by space)";
	p.val = "0.4 0.4  0.4 0.4  0.4 0.4  0.4 0.4";
	params[var] = p;

	var = "ns_obs_locs";
	p.des = "Obstacle locations (real locations, this is for testing/verification purpose only) (Default: 1.0 0.8  3.0 1.5  5.5 0.2  8.2 1.0) (Note: provide vector in one line separated by space)";
	p.val = "1.0 0.8  3.0 1.5  5.5 0.2  8.2 1.0";
	params[var] = p;

	var = "ns_output_times";
	p.des = "Simulation output sampiling time instances. (Default: 2.5 5.0 7.5 10.0) (Note: provide vector in one line separated by space)";
	p.val = "2.5 5.0 7.5 10.0";
	params[var] = p;

	var = "ns_output_locations";
	p.des = "Simulation output sampling locations: loc1x, loc1y, loc2x, loc2y, .... (Default: <10 locations>) (Note: provide vector in one line separated by space)";
	p.val = "1.5 0.6  3.1 0.6  4.7 0.6  6.3 0.6  7.9 0.6  1.5 1.3  3.1 1.3  4.7 1.3  6.3 1.3  7.9 1.3";
	params[var] = p;
}

void Config::parse_file()
{	
	// Open config file
	ifstream f(config_file);
	if (!f) {
		fflush(NULL);
		printf("\nWARNING: cannot open config file.\n");
		return;
	}
	// Read lines
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
	for (int i=0; i < argc; ++i) {
		if (params.find(argv[i]) == params.end()) {
			continue;
		} else {
			// For a valid parameter
			if (i+1 < argc) {
				params[argv[i]].val = argv[i+1];
				i += 1;
			}
		}
	}
	return;
}

void Config::print_help(int argc, char** argv)
{	
	for (int i=argc-1; i >= 0; --i) {
		string s(argv[i]);
		if (s=="-h" || s=="--h" || s=="-help" || s=="--help") {
			cout << "\n[USAGE]: (executable) <option1> <value2> <option2> <value2> ...\n";
			cout << "\n[Example 1]: To execute with all default values...";
			cout << "\n\t(executable)\n";
			cout << "\n[Example 2]: To execute with specified values (command line values override confige_file values, config_file values override default values)...";
			cout << "\n\t(executable) config_file \"./input/ns.dat\" global_is_resume yes global_observed_data \"1 2 3 4 5\"\n";
		
			cout << "\nAvailable config options:\n" << endl;
			cout << "   " << "config_file";
			cout << "\n   \t>> Description: Configuration file. (Default: ) (Type: string)";
			cout << "\n   \t>> Current value: " << config_file << "\n" << endl;
			print_config();
			return;
		} else {
			continue;
		}
	}
	return;
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

string tools::sample_to_string(vector<double> const& v)
{
	std::ostringstream oss;
	oss << "[ " << std::fixed << std::setprecision(6);
	for (int i=0; i < v.size(); ++i)
		oss << v[i] << " ";
	oss << "]";
	return oss.str();
}

string tools::samplepos_to_string(vector<double> const& v)
{
	std::ostringstream oss;
	oss << "[ " << std::fixed << std::setprecision(6);
	for (int i=0; i < v.size()-1; ++i)
		oss << v[i] << " ";
	oss << "] " << v.back();
	return oss.str();
}

/**
 * Compute the normalized Euclidean norm of two vectors
 * normalized_l2_norm = 0.5 * |(u-umean)-(v-vmean)|^2 / (|u-umean|^2 + |v-vmean|^2), |x|:= l2norm of x = sqrt(sum(x_i^2))
 *
 * Normalized l2-norm transforms vector into unit vector, and then compare,
 * which mean only compares vector orientation, ignoring vector actual magnitude
 */
double tools::compute_normalizedl2norm(
		vector<double> const& u,
		vector<double> const& v)
{
	if (u.size() != v.size()) {
		fflush(NULL);
		printf("ERROR: vectors size mismatch. Program abort!\n");
		exit(EXIT_FAILURE);
	}
	double u_mean = 0.0;
	for (auto i: u) u_mean += i;
	u_mean /= u.size();

	double v_mean = 0.0;
	for (auto i: v) v_mean += i;
	v_mean /= v.size();

	double uu2 = 0.0;
	for (auto i: u) uu2 += (i-u_mean)*(i-u_mean);

	double vv2 = 0.0;
	for (auto i: v) vv2 += (i-v_mean)*(i-v_mean);

	double uv2 = 0.0;
	for (int i=0; i < u.size(); ++i)
		uv2 += (u[i]-u_mean-v[i]+v_mean)*(u[i]-u_mean-v[i]+v_mean);

	return (0.5 * uv2) / (uu2 + vv2);
}

/**
 * Compute the Euclidean norm (l2-norm) of a vector 
 * |x|:= l2norm of x = sqrt(sum(x_i^2))
 */
double tools::compute_l2norm(
		vector<double> const& v)
{
	double sum = 0.0;
	for (int i=0; i < v.size(); ++i) {
		sum += v[i]*v[i];
	}
	return sqrt(sum);
}

/**
 * Compute the Euclidean norm (l2-norm) of two vectors 
 * l2_norm = sqrt(sum((xi-yi)^2)), |x|:= l2norm of x = sqrt(sum(x_i^2))
 *
 * l2-norm compares both magnitude and oriention of two vectors
 */
double tools::compute_l2norm_diff(
		vector<double> const& u,
		vector<double> const& v)
{
	if (u.size() != v.size()) {
		fflush(NULL);
		printf("ERROR: vectors size mismatch. Program abort!\n");
		exit(EXIT_FAILURE);
	}
	double sum = 0.0;
	for (int i=0; i < u.size(); ++i) {
		sum += (u[i]-v[i])*(u[i]-v[i]);
	}
	return sqrt(sum);
}

/**
 * Compute the Euclidean norm (l2-norm) of two vectors 
 * l2_norm = sqrt(sum((xi-yi)^2)), |x|:= l2norm of x = sqrt(sum(x_i^2))
 *
 * l2-norm compares both magnitude and oriention of two vectors
 */
double tools::compute_l2norm_sum(
		vector<double> const& u,
		vector<double> const& v)
{
	if (u.size() != v.size()) {
		fflush(NULL);
		printf("ERROR: vectors size mismatch. Program abort!\n");
		exit(EXIT_FAILURE);
	}
	double sum = 0.0;
	for (int i=0; i < u.size(); ++i) {
		sum += (u[i]+v[i])*(u[i]+v[i]);
	}
	return sqrt(sum);
}


string tools::exec(const char* cmd) 
{
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}
