// eBayes - Elastic Bayesian Inference Framework with iMPI
// Copyright (C) 2015-today Ao Mo-Hellenbrand
//
// All copyrights remain with the respective authors.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include <tools/ErrorAnalysis.hpp>

using namespace std;


void ErrorAnalysis::add_test_points(std::size_t n)
{
	std::random_device rd;
	std::mt19937 eng (rd());
	pair<double,double> range;
	std::size_t input_size = cfg.get_input_size();
	test_points.resize(n);
	test_points_data.resize(n);

	par.info();
	printf("EA: adding test points...\n");

	for (std::size_t k=0; k < n; ++k) {
		test_points[k].resize(input_size);
		for (std::size_t i=0; i < input_size; ++i) {
			range = fullmodel.get_input_space(i);
			uniform_real_distribution<double> udist (range.first, range.second);
			test_points[k][i] = udist(eng);
		}
		test_points_data[k] = fullmodel.run(test_points[k]);
	}
}

void ErrorAnalysis::add_test_point_at(vector<double> const& m)
{
	test_points.push_back( vector<double>(m.cbegin(), m.cend()) );
	test_points_data.push_back( fullmodel.run(m) );
	return;
}

void ErrorAnalysis::copy_test_points(ErrorAnalysis const& that)
{
	this->test_points = that.test_points;
	this->test_points_data = that.test_points_data;
	return;
}

double ErrorAnalysis::compute_surrogate_error()
{
	if (test_points.size() < 1) {
		par.info();
		printf("ERROR: EA compute surrogate error failed due to no test points. Program abort!\n");
		exit(EXIT_FAILURE);
	}
	if (test_points.size() != test_points_data.size()) {
		par.info();
		printf("ERROR: EA compute surrogate error failed, test points and data dimension mismatch. Program abort!\n");
		exit(EXIT_FAILURE);
	}
	std::size_t n = test_points.size();
	double denom, err, sum = 0.0;
	for (int i=0; i < n; ++i) {
		// err := l2norm( g(x) - f(x) ) / l2norm( g(x) + f(x) )
		// err in [0.0, 1.0]
		denom = tools::compute_l2norm_sum(test_points_data[i], surrogate.run(test_points[i]));
		if (denom == 0.0) {
			err = 0.0;
		} else {
			err = tools::compute_l2norm_diff(test_points_data[i], surrogate.run(test_points[i])) / denom;
			if (isnan(err) || isinf(err) || err > 1.0) err = 1.0;
		}
		sum += err;
	}
	return sum/double(n);
}

double ErrorAnalysis::compute_surrogate_error_at(std::vector<double> const& m)
{
	vector<double> fd = fullmodel.run(m);
	vector<double> sd = surrogate.run(m);
	double denom = tools::compute_l2norm_sum(fd, sd);
	if (denom == 0.0) {
		return 0.0;
	}
	// err := l2norm( g(x) - f(x) ) / l2norm( g(x) + f(x) )
	// err in [0.0, 1.0]
	return tools::compute_l2norm_diff(fd, sd) / denom;
}

bool ErrorAnalysis::eval_model_spmd(double tol)
{
	double local_err = compute_surrogate_error();
	double err = 0.0;
	MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	err = err/double(par.size);

#if (EA_LOCALINFO == 1)
	par.info();
	printf("EA: %lu test points, local error %.8f, global error %.8f\n",
			test_points.size(), local_err, err);
#endif

	if (par.is_master()) {
		fflush(NULL);
		printf("EA: Surrogate error %.8f, tol %.2f\n", err, tol);
	}
	return (err <= tol) ? true : false;
}


/**
 * This function only MASTER does the surrogate error eval, then it Bcast error to other ranks
 */
bool ErrorAnalysis::eval_model_master(double tol)
{
	double err = 0.0;

	if (par.is_master()) {
		err = compute_surrogate_error();
	}
	MPI_Bcast(&err, 1, MPI_DOUBLE, par.master, MPI_COMM_WORLD);
	err = err/double(par.size);

#if (EA_LOCALINFO == 1)
	par.info();
	printf("EA: %lu test points, local error %.8f, global error %.8f\n",
			test_points.size(), local_err, err);
#endif

	if (par.is_master()) {
		fflush(NULL);
		printf("EA: Surrogate error %.8f | tol %.2f\n", err, tol);
	}
	return (err <= tol) ? true : false;
}


void ErrorAnalysis::read_test_points(std::string infile)
{
	// Open input
	ifstream f(infile);
	if (!f) {
		fflush(NULL);
		printf("\nWARNING: EA read_test_points cannot open input file.\n");
		return;
	}
	// Read lines
	test_points.clear();
	string s;
	vector<double> tp;
	while (std::getline(f, s)) {
		// Read line into string stream
		istringstream iss(s);
		vector<string> tokens {istream_iterator<string>{iss}, istream_iterator<string>{}};
		// Ignore empty lines
		if (tokens.size() < 1) continue;
		// Ignore comment lines (start with // or #)
		tokens[0] = tools::trim_white_space(tokens[0]);
		if ((tokens[0].substr(0,2) == "//") || (tokens[0].substr(0,1) == "#")) continue;

		// For a valid line
		tp.clear();
		for (auto it=tokens.begin(); it != tokens.end(); ++it) {
			 tp.push_back( std::atof((*it).c_str()) );
		}
		test_points.push_back(tp);
	}//end while
	f.close();
}

void ErrorAnalysis::write_test_points(std::string outfile)
{
	// Open output
	ofstream f(outfile);
	if (!f) {
		fflush(NULL);
		printf("\nWARNING: EA write_test_points cannot open input file.\n");
		return;
	}

	// Write lines
	for (auto it=test_points.begin(); it != test_points.end(); ++it) {
		f << vec_to_string(*it) << "\n";
	}
	f.close();
}

void ErrorAnalysis::print_test_points()
{
	par.info();
	printf("EA test points:\n");
	for (int i=0; i < test_points.size(); ++i) {
		printf("#%02d: %s\n", i, vec_to_string(test_points[i]).c_str());
	}
	printf("\n");
	return;
}

string ErrorAnalysis::vec_to_string(vector<double> const& v)
{
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(6);
	for (int i=0; i < v.size()-1; ++i)
		oss << v[i] << " ";
	oss << v.back();
	return oss.str();
}
