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

ErrorAnalysis::ErrorAnalysis (ForwardModel* full, ForwardModel* surr)
{
	this->fullmodel = full;
	this->surrogate = surr;
	this->input_size = fullmodel->get_input_size();
	this->output_size = fullmodel->get_output_size();
}


void ErrorAnalysis::update_surrogate(ForwardModel* surrogatemodel)
{
	this->surrogate = surrogatemodel;
}

void ErrorAnalysis::add_test_point(const double* m)
{
	// Add a new test_point slot and its data
	test_points.push_back( unique_ptr<double[]>(new double[input_size]) );
	test_points_data.push_back( unique_ptr<double[]>(new double[output_size]) );

	// If m is not nullptr, copy it.
	// Otherwize, generate a random test point.
	if (m) {
		for (size_t i=0; i<input_size; i++)
			test_points.back()[i] = m[i];
	} else {
		double dmin, dmax;
		default_random_engine gen;
		for (size_t i=0; i<input_size; i++) {
			fullmodel->get_input_space(i, dmin, dmax);
			uniform_real_distribution<double> udist (dmin, dmax);
			test_points.back()[i] = udist(gen);
		}
	}
	// Compute test point data
	fullmodel->run(test_points.back().get(), test_points_data.back().get());
	return;
}

void ErrorAnalysis::add_test_points(int M)
{
	printf("Error Analysis adding %d points with full model.\n", M);
	for (int k=0; k < M; k++)
		add_test_point();
}

void ErrorAnalysis::copy_test_points(const ErrorAnalysis* that)
{
	int num_tps = that->test_points.size();
	this->test_points.resize(num_tps);
	this->test_points_data.resize(num_tps);

	for (int k=0; k < num_tps; k++) {
		// copy k-th test point
		this->test_points[k].reset(new double[input_size]);
		for (size_t i=0; i < input_size; i++)
			this->test_points[k][i] = that->test_points[k][i];

		// copy k-th test point data
		this->test_points_data[k].reset(new double[output_size]);
		for (size_t j=0; j < output_size; j++)
			this->test_points_data[k][j] = that->test_points_data[k][j];
	}
	return;
}

double ErrorAnalysis::compute_model_error()
{
	size_t num_tps = test_points.size();
	unique_ptr<double[]> d (new double[output_size]);

	// Compute surrogate model error for each test point
	double mean = 0.0;
	for (size_t i=0; i < num_tps; i++) {
		surrogate->run(test_points[i].get(), d.get());
		mean += ForwardModel::compute_l2norm(test_points_data[i].get(), d.get(), output_size);
	}
	return mean/double(num_tps);
}

double ErrorAnalysis::compute_model_error(const double* m)
{
	unique_ptr<double[]> data_full (new double[output_size]);
	fullmodel->run(m, data_full.get());

	unique_ptr<double[]> data_surr (new double[output_size]);
	surrogate->run(m, data_surr.get());

	return ForwardModel::compute_l2norm(data_full.get(), data_surr.get(), output_size);
}






