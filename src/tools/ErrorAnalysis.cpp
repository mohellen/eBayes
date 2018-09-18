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

	fflush(NULL);
	printf("EA: Rank[%d|%d](%d) adding test points...\n",
			par.rank, par.size, par.status);

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
		fflush(NULL);
		printf("ERROR: EA compute surrogate error failed due to no test points. Program abort!\n");
		exit(EXIT_FAILURE);
	}
	if (test_points.size() != test_points_data.size()) {
		fflush(NULL);
		printf("ERROR: EA compute surrogate error failed, test points and data dimension mismatch. Program abort!\n");
		exit(EXIT_FAILURE);
	}
	std::size_t n = test_points.size();
	double err, sum = 0.0;
	int count = 0;
	for (int i=0; i < n; ++i) {
		err = tools::compute_l2norm(test_points_data[i], surrogate.run(test_points[i]));
		if (std::isnan(err) || std::isinf(err)) continue;
		sum += err;
		count++;
	}
	return sum/double(count);
}

double ErrorAnalysis::compute_surrogate_error_at(std::vector<double> const& m)
{
	vector<double> fd = fullmodel.run(m);
	vector<double> sd = surrogate.run(m);
	return tools::compute_l2norm(fd, sd);
}

bool ErrorAnalysis::mpi_is_model_accurate(double tol)
{
	double local_err = compute_surrogate_error();
	vector<double> err (par.size);
	MPI_Allgather(&local_err, 1, MPI_DOUBLE, &err[0], 1, MPI_DOUBLE, MPI_COMM_WORLD);
	// Exclude invalid values
	double mean = 0.0;
	int count = 0;
	for (double e: err) {
		if (std::isnan(e) || std::isinf(e)) continue;
		mean += e;
		count++;
	}
	mean = mean / double(count);

#if (EA_LOCALINFO == 1)
	fflush(NULL);
	printf("EA: Rank[%d|%d](%d) %lu test points, local error %.8f, global error %.8f\n",
			par.rank, par.size, par.status, test_points.size(), local_err, mean);
#endif

	if (par.is_master()) {
		fflush(NULL);
		printf("EA: Surrogate error %.8f, tol %.2f\n", mean, tol);
	}
	return (mean <= tol) ? true : false;
}

