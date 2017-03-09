// This file is part of BayeSIFSG - Bayesian Statistical Inference Framework with Sparse Grid
// Copyright (C) 2015-today Ao Mo-Hellenbrand.
//
// SIPFSG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// SIPFSG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License.
// If not, see <http://www.gnu.org/licenses/>.

#include <eanalysis/EA.hpp>


using namespace std;

EA::EA (ForwardModel* fullmodel, ForwardModel* surrogatemodel, const double* m)
{
	this->fm = fullmodel;
	this->sm = surrogatemodel;
	this->input_size = fm->get_input_size();
	this->output_size = fm->get_output_size();

	this->test_point.reset(new double[input_size]);
	for (std::size_t i=0; i < input_size; i++)
		test_point[i] = m[i];

	this->fm_data.reset(new double[output_size]);
	fm->run(m, fm_data.get());
}


void EA::set_test_point(double* m)
{
	test_point.reset(m);
	fm->run(m, fm_data.get());
}

double EA::err()
{
	auto d = unique_ptr<double[]>(new double[output_size]);
	sm->run(test_point.get(), d.get());
	double err = ForwardModel::compute_l2norm(fm_data.get(), d.get(), output_size);
	printf("\nSurrogate l2norm error: %.6f\n", err);
	return err;
}

double EA::err(const double* m)
{
	fm->run(m, fm_data.get());
	auto d = unique_ptr<double[]>(new double[output_size]);
	sm->run(m, d.get());
	double err = ForwardModel::compute_l2norm(fm_data.get(), d.get(), output_size);
	printf("\nSurrogate l2norm error: %.6f\n", err);
	return err;
}






