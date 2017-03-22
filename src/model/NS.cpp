// iBayes - Elastic Bayesian Inference Framework with Sparse Grid
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

#include "model/NS.hpp"

using namespace std;
using namespace Eigen;


/************************************
 *  Public Methods
 ************************************/

NS::~NS()
{
	this->obs.clear();
	this->out_times.clear();
	this->out_locs.clear();
}

NS::NS(const string& input_file, int resx, int resy)
		: ForwardModel()
{
	ifstream infile(input_file);
	string s;
	while (std::getline(infile, s)) {
		istringstream iss(s);
		vector<string> tokens {istream_iterator<string>{iss}, istream_iterator<string>{}};

		// Ignore empty line
		if (tokens.size() <= 0) continue;

		// Ignore comment line
		tokens[0] = trim_white_space(tokens[0]);
		if (tokens[0].substr(0,2) == "//") continue;

		// Find parameters
		transform(tokens[0].begin(), tokens[0].end(), tokens[0].begin(), ::tolower);
		if (tokens[0] == "domain_size_x") {
			this->domain_size_x = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "domain_size_y") {
			this->domain_size_y = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "min_ncx") {
			this->ncx = stoi(tokens[1]);
			continue;
		}
		if (tokens[0] == "min_ncy") {
			this->ncy = stoi(tokens[1]);
			continue;
		}
		if (tokens[0] == "initial_velocity_x") {
			this->initial_velocity_x = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "initial_velocity_y") {
			this->initial_velocity_y = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "initial_pressure") {
			this->initial_pressure = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "inlet_velocity_x") {
			this->inlet_velocity_x = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "inlet_velocity_y") {
			this->inlet_velocity_y = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "external_force_x") {
			this->external_force_x = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "external_force_y") {
			this->external_force_y = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "re") {
			this->re = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "tau") {
			this->tau = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "alpha") {
			this->alpha = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "omega") {
			this->omega = stod(tokens[1]);
			continue;
		}
		if (tokens[0] == "boundary_north") {
			transform(tokens[1].begin(), tokens[1].end(), tokens[1].begin(), ::tolower);
        	if (tokens[1] == "inlet")
        		this->boundary_north = BOUNDARY_TYPE_INLET;
            else if (tokens[1] == "outlet")
            	this->boundary_north = BOUNDARY_TYPE_OUTLET;
            else if (tokens[1] == "noslip")
            	this->boundary_north = BOUNDARY_TYPE_NOSLIP;
            else if (tokens[1] == "freeslip")
            	this->boundary_north = BOUNDARY_TYPE_FREESLIP;
            continue;
		}
		if (tokens[0] == "boundary_south") {
			transform(tokens[1].begin(), tokens[1].end(), tokens[1].begin(), ::tolower);
        	if (tokens[1] == "inlet")
        		this->boundary_south = BOUNDARY_TYPE_INLET;
            else if (tokens[1] == "outlet")
            	this->boundary_south = BOUNDARY_TYPE_OUTLET;
            else if (tokens[1] == "noslip")
            	this->boundary_south = BOUNDARY_TYPE_NOSLIP;
            else if (tokens[1] == "freeslip")
            	this->boundary_south = BOUNDARY_TYPE_FREESLIP;
            continue;
		}
		if (tokens[0] == "boundary_east") {
			transform(tokens[1].begin(), tokens[1].end(), tokens[1].begin(), ::tolower);
        	if (tokens[1] == "inlet")
        		this->boundary_east = BOUNDARY_TYPE_INLET;
            else if (tokens[1] == "outlet")
            	this->boundary_east = BOUNDARY_TYPE_OUTLET;
            else if (tokens[1] == "noslip")
            	this->boundary_east = BOUNDARY_TYPE_NOSLIP;
            else if (tokens[1] == "freeslip")
            	this->boundary_east = BOUNDARY_TYPE_FREESLIP;
            continue;
		}
		if (tokens[0] == "boundary_west") {
			transform(tokens[1].begin(), tokens[1].end(), tokens[1].begin(), ::tolower);
        	if (tokens[1] == "inlet")
        		this->boundary_west = BOUNDARY_TYPE_INLET;
            else if (tokens[1] == "outlet")
            	this->boundary_west = BOUNDARY_TYPE_OUTLET;
            else if (tokens[1] == "noslip")
            	this->boundary_west = BOUNDARY_TYPE_NOSLIP;
            else if (tokens[1] == "freeslip")
            	this->boundary_west = BOUNDARY_TYPE_FREESLIP;
            continue;
		}
		if (tokens[0] == "obstacle") {
			this->obs.push_back(
					Obstacle(stod(tokens[1]), stod(tokens[2]), stod(tokens[3]), stod(tokens[4])) );
			continue;
		}
		if (tokens[0] == "output_time") {
			this->out_times.push_back( stod(tokens[1]) );
			continue;
		}
		if (tokens[0] == "output_location") {
			this->out_locs.push_back(
					pair<double, double>(stod(tokens[1]), stod(tokens[2])) );
			continue;
		}
	}//end while
	infile.close();

	this->input_size  = std::size_t(this->obs.size() * 2);
	this->output_size = std::size_t(this->out_times.size() * this->out_locs.size());
	this->ncx *= max(1, resx);
	this->ncy *= max(1, resy);
	this->dx = this->domain_size_x / double(this->ncx);
	this->dy = this->domain_size_y / double(this->ncy);
	return;
}

std::size_t NS::get_input_size()
{
	return input_size;
}

std::size_t NS::get_output_size()
{
	return output_size;
}

void NS::get_input_space(
		int dim,
		double& min,
		double& max)
{
	min = 0.0;
	if (dim%2 == 0)
		max = domain_size_x - obs[int(dim/2)].sizex;
	else
		max = domain_size_y - obs[int(dim/2)].sizey;
	return;
}

void NS::get_resolution(
		std::size_t& nx,
		std::size_t& ny)
{
	nx = ncx;
	ny = ncy;
}

void NS::run(const double* m, double* d)
{
	double t = 0.0;
	double dt = 0.0;
	double t_end = out_times.back();
	int t_out_idx = 0;

	double u,v,ve;
	std::size_t i,j,k;

	/**********************************************************
	 * 2D Arrays: Row-major storage, including boundary cells
	 * 	 dim_0 along y-direction (rows,    j, ncy)
	 * 	 dim_1 along x-direction (columns, i, ncx)
	 *
	 * 	 Inner domain: m[j][i], i=1..ncx, j=1..ncy
	 * 	 Boundaries: LEFT   m[j][0],
	 * 	             RIGHT  m[j][ncx+1],
	 * 	             TOP    m[ncy+1][i],
	 * 	             BOTTOM m[0][i]
	 **********************************************************/
	// allocate computation arrays
	double** U = alloc_matrix<double>(ncy+2, ncx+2, true); /// velocity in x-direction
	double** V = alloc_matrix<double>(ncy+2, ncx+2, true); /// velocity in y-direction
	double** P = alloc_matrix<double>(ncy+2, ncx+2, true); /// pressure
	double** F = alloc_matrix<double>(ncy+2, ncx+2, true); /// temporary array: F value
	double** G = alloc_matrix<double>(ncy+2, ncx+2, true); /// temporary array: G value
	// initialize arrays
	init_matrix<double>(U, ncy+2, ncx+2, true, initial_velocity_x);
	init_matrix<double>(V, ncy+2, ncx+2, true, initial_velocity_y);
	init_matrix<double>(P, ncy+2, ncx+2, true, initial_pressure);
	init_matrix<double>(F, ncy+2, ncx+2, true, 0.0);
	init_matrix<double>(G, ncy+2, ncx+2, true, 0.0);
	// Create geometry mask
	int** M = create_geometry_mask(m);

#if (NS_USE_DIRECT_SOLVER == 1)
	// System matrix for the pressure equation: A*p = rhs
	SparseMatrix<double> A = create_system_matrix();
	double** RHS = alloc_matrix<double>(ncy+2, ncx+2, true);
#endif

	update_boundaries_uv(U, V);
	update_domain_uv(M, U, V);
	while(t < t_end)
	{
		// 1. Compute time step size
		compute_dt(U, V, dt);

		// 2. Compute F,G
		compute_fg(dt, U, V, F, G);
		update_boundaries_fg(U, V, F, G);
		update_domain_fg(M, U, V, F, G);

		// 3. Solve for pressure P: A*p = rhs
#if (NS_USE_DIRECT_SOLVER == 1)
		compute_rhs(dt, F, G, P, RHS);
		solve_for_p_direct(A, RHS, P);
#else
		solve_for_p_sor(dt, F, G, P);
#endif
		update_boundaries_p(P);
		update_domain_p(M, P);

		// 4. Compute velocity U,V
		compute_uv(dt, F, G, P, U, V);
		update_boundaries_uv(U, V);
		update_domain_uv(M, U, V);

		// 5. Update simulation time
		t += dt;

		// 6. Generate output data at out_times and out_locs
		if (t >= out_times[t_out_idx]) {
			for(k = 0; k < out_locs.size(); k++) {
				// Convert sampling coordinates (x,y)
				// to cell[i][j]'s top-right corner (point-data)
				i = size_t(round(out_locs[k].first/dx));
				j = size_t(round(out_locs[k].second/dy));

				// Artificially darken (set zero) points lie in obstacle cells
				if (is_point_in_obs_cell(M, j, i)) {
					ve = 0.0;
				} else {
					u = 0.5 * (U[j][i] + U[j+1][i]);
					v = 0.5 * (V[j][i] + V[j][i+1]);
					ve = sqrt(u*u + v*v);
				}
				d[t_out_idx*out_locs.size() + k] = ve;
			}
			t_out_idx ++;
		}
	}//end while

	free_matrix<int>(M);
	free_matrix<double>(U);
	free_matrix<double>(V);
	free_matrix<double>(P);
	free_matrix<double>(F);
	free_matrix<double>(G);
#if (NS_USE_DIRECT_SOLVER == 1)
	free_matrix<double>(RHS);
	A.resize(0,0);
#endif
	return;
}//end fun()

void NS::sim(const string& output_prefix)
{
	double t = 0.0;
	double dt = 0.0;
	double t_end = out_times.back();
	int t_out_idx = 0;

	double u,v,ve;
	std::size_t i,j,k;

	double vtk_freq = 0.05;
	int vtk_cnt = 0;
	std::string vtk_outfile =
			(output_prefix == "") ? "output/ns_sim" : output_prefix;

	/**********************************************************
	 * 2D Arrays: Row-major storage, including boundary cells
	 * 	 dim_0 along y-direction (rows,    j, ncy)
	 * 	 dim_1 along x-direction (columns, i, ncx)
	 *
	 * 	 Inner domain: m[j][i], i=1..ncx, j=1..ncy
	 * 	 Boundaries: LEFT   m[j][0],
	 * 	             RIGHT  m[j][ncx+1],
	 * 	             TOP    m[ncy+1][i],
	 * 	             BOTTOM m[0][i]
	 **********************************************************/
	// allocate computation arrays
	double** U = alloc_matrix<double>(ncy+2, ncx+2, true); /// velocity in x-direction
	double** V = alloc_matrix<double>(ncy+2, ncx+2, true); /// velocity in y-direction
	double** P = alloc_matrix<double>(ncy+2, ncx+2, true); /// pressure
	double** F = alloc_matrix<double>(ncy+2, ncx+2, true); /// temporary array: F value
	double** G = alloc_matrix<double>(ncy+2, ncx+2, true); /// temporary array: G value
	// initialize arrays
	init_matrix<double>(U, ncy+2, ncx+2, true, initial_velocity_x);
	init_matrix<double>(V, ncy+2, ncx+2, true, initial_velocity_y);
	init_matrix<double>(P, ncy+2, ncx+2, true, initial_pressure);
	init_matrix<double>(F, ncy+2, ncx+2, true, 0.0);
	init_matrix<double>(G, ncy+2, ncx+2, true, 0.0);
	// create geometry mask  with input from file
	double* m = new double[input_size];
	for (k=0; k < obs.size(); k++) {
		m[k*2 + 0] = obs[k].locx;
		m[k*2 + 1] = obs[k].locy;
	}
	int** M = create_geometry_mask(m);
//	print_mask(M); // DEBUG only

#if (NS_USE_DIRECT_SOLVER == 1)
	// System matrix for the pressure equation: A*p = rhs
	SparseMatrix<double> A = create_system_matrix();
	double** RHS = alloc_matrix<double>(ncy+2, ncx+2, true);
#endif

	update_boundaries_uv(U, V);
	update_domain_uv(M, U, V);

	// Initial VTK output
	write_vtk_file(vtk_outfile.c_str(), vtk_cnt, M, U, V, P);
	vtk_cnt ++;

	while(t < t_end)
	{
		// 1. Compute time step
		compute_dt(U, V, dt);

		// 2. Compute F,G
		compute_fg(dt, U, V, F, G);
		update_boundaries_fg(U, V, F, G);
		update_domain_fg(M, U, V, F, G);

		// 3. Solve the pressure P: A*p = rhs
#if (NS_USE_DIRECT_SOLVER == 1)
		compute_rhs(dt, F, G, P, RHS);
		solve_for_p_direct(A, RHS, P);
#else
		solve_for_p_sor(dt, F, G, P);
#endif
		update_boundaries_p(P);
		update_domain_p(M, P);

		// 4. Compute velocity U,V
		compute_uv(dt, F, G, P, U, V);
		update_boundaries_uv(U, V);
		update_domain_uv(M, U, V);

		// 5. Update simulation time
		t += dt;
		cout << "Simulation completed at t = " << t << ", dt = " << dt << endl;

		// 6. Write VTK output data (if enabled) */
		if (floor(t/vtk_freq) >= vtk_cnt) {
			cout << "Writing VTK output at t = " << t << endl;
			write_vtk_file(vtk_outfile.c_str(), vtk_cnt, M, U, V, P);
			vtk_cnt ++;
		}
	}//end while

	free_matrix<int>(M);
	free_matrix<double>(U);
	free_matrix<double>(V);
	free_matrix<double>(P);
	free_matrix<double>(F);
	free_matrix<double>(G);
#if (NS_USE_DIRECT_SOLVER == 1)
	free_matrix<double>(RHS);
	A.resize(0,0);
#endif
	return;
}

void NS::print_info()
{
	printf("------ NS Object info ------\n");
	printf("-- Domain size x:      %6.2f\n", domain_size_x);
	printf("-- Domain size y:      %6.2f\n", domain_size_y);
	printf("-- Initial velocity x: %6.2f\n", initial_velocity_x);
	printf("-- Initial velocity y: %6.2f\n", initial_velocity_y);
	printf("-- Initial pressure:   %6.2f\n", initial_pressure);
	printf("-- Inlet velocity x:   %6.2f\n", inlet_velocity_x);
	printf("-- Inlet velocity y:   %6.2f\n", inlet_velocity_y);
	printf("-- External force x:   %6.2f\n", external_force_x);
	printf("-- External force y:   %6.2f\n", external_force_y);
	printf("-- Reynolds:           %6.2f\n", re);
	printf("-- Tau:                %6.2f\n", tau);
	printf("-- Alpha:              %6.2f\n", alpha);
	printf("-- Omega:              %6.2f\n", omega);
	printf("-- \n");
	printf("-- Boundary north type: %d\n", boundary_north);
	printf("-- Boundary south type: %d\n", boundary_south);
	printf("-- Boundary east type:  %d\n", boundary_east);
	printf("-- Boundary west type:  %d\n", boundary_west);
	printf("-- \n");
	printf("-- Input size:  %lu\n", input_size);
	printf("-- Output size: %lu\n", output_size);
	printf("-- Ncx:         %lu\n", ncx);
	printf("-- Ncy:         %lu\n", ncy);
	printf("-- Dx:        %.2f\n", dx);
	printf("-- Dy:        %.2f\n", dy);
	printf("-- \n");
	for (int i=0; i < obs.size(); i++)
		printf("-- Obstacle: %.2f, %.2f, %.2f, %.2f\n", obs[i].locx, obs[i].locy, obs[i].sizex, obs[i].sizey);
	printf("-- \n");
	for (int i=0; i < out_times.size(); i++)
		printf("-- Output time: %.2f\n", out_times[i]);
	printf("-- \n");
	for (int i=0; i < out_locs.size(); i++)
		printf("-- Output location: %.2f, %.2f\n", out_locs[i].first, out_locs[i].second);
	printf("----------------------------\n");
}

void NS::print_mask(int **& M)
{
	printf("\n");
	for (std::size_t j=0; j<=ncy+1; j++) {
		for (std::size_t i=0; i<=ncx+1; i++)
			printf("%5d ", M[(ncy+1)-j][i]);
		printf("\n");
	}
	printf("\n");
	return;
}

/*****************************************
 * Internal core functions
 *****************************************/

int** NS::create_geometry_mask(const double* m) /// Input parameter vector: locations of obstacles [obs0_x, obs0_y, obs1_x, obs1_y, ...]
{
	std::size_t i,j,k; // looping indices
	std::size_t jl, jh, il, ih; // low & high index of columns

	int** M = alloc_matrix<int>(ncy+2, ncx+2, true);
	init_matrix<int>(M, ncy+2, ncx+2, true, 0);

	// 1. Initialize all non-fluid cells to 10000
	// 1.1 Setup boundaries
	if (boundary_west == BOUNDARY_TYPE_NOSLIP || boundary_west == BOUNDARY_TYPE_FREESLIP) {
		for (j=1; j <= ncy; j++)
			M[j][0] = 10000;
	}
    if (boundary_east == BOUNDARY_TYPE_NOSLIP || boundary_east == BOUNDARY_TYPE_FREESLIP) {
		for (j=1; j <= ncy; j++)
			M[j][ncx+1] = 10000;
	}
    if (boundary_north == BOUNDARY_TYPE_NOSLIP || boundary_north == BOUNDARY_TYPE_FREESLIP) {
		for (i=0; i <= ncx+1; i++)
			M[ncy+1][i] = 10000;
	}
    if (boundary_south == BOUNDARY_TYPE_NOSLIP || boundary_south == BOUNDARY_TYPE_FREESLIP) {
		for (i=0; i <= ncx+1; i++)
			M[0][i] = 10000;
	}
	// 1.2 Construct obs based on input locations, setup inner domain
	for (k=0; k < obs.size(); k++) {
		il = size_t(round(m[k*2 + 0]/dx) + 1);
		ih = size_t(round((m[k*2 + 0] + obs[k].sizex)/dx));
		jl = size_t(round(m[k*2 + 1]/dy) + 1);
		jh = size_t(round((m[k*2 + 1] + obs[k].sizey)/dy));
		for (j=jl; j<=jh; j++)
			for (i=il; i<=ih; i++)
				M[j][i] = 10000;
	}
	// 1.3 Re-check boundaries
	for (j=1; j <= ncy; j++) { // West
		if (M[j][1] != FLUID) M[j][0] = 10000;
	}
	for (j=1; j <= ncy; j++) { // East
		if (M[j][ncx] != FLUID) M[j][ncx+1] = 10000;
	}
	for (i=0; i <= ncx+1; i++) { // North
		if (M[ncy][i] != FLUID) M[ncy+1][i] = 10000;
	}
	for (i=0; i <= ncx+1; i++) { // South
		if (M[1][i] != FLUID) M[0][i] = 10000;
	}
	// 2. set up inner domain non-fluid cell types
	for (j=1; j <= ncy; j++) {
		for (i=1; i <= ncx; i++) {
			if (M[j][i] > 0) {
				if (M[j][i+1] > 0) M[j][i] += 1000; //East
				if (M[j][i-1] > 0) M[j][i] += 100;  //West
				if (M[j+1][i] > 0) M[j][i] += 10;   //North
				if (M[j-1][i] > 0) M[j][i] += 1;    //South
			}
		}
	}
	return M;
}

void NS::update_boundaries_uv(
		double** &U,  /// Input/Output
		double** &V)  /// Input/Output
{
	std::size_t i,j;

	// Top Boundary: wall (no-slip)
//	for (size_t i=0; i <= (ncx+1); i++) {
//		U[ncy+1][i] = -U[ncy][i];
//		V[ncy  ][i] = 0.0;
//	}
    //north
	if (boundary_north == BOUNDARY_TYPE_INLET) {
		for (i=0; i<=ncx; i++)
			U[ncy+1][i] = inlet_velocity_x*2.0 - U[ncy][i];
		for (i=1; i<=ncx; i++)
			V[ncy][i] = inlet_velocity_y;
	}
	else if (boundary_north == BOUNDARY_TYPE_OUTLET) {
		for (i=0; i<=ncx; i++)
			U[ncy+1][i] = U[ncy][i];
		for (i=1; i<=ncx; i++)
			V[ncy][i] = V[ncy-1][i];
	}
	else if (boundary_north == BOUNDARY_TYPE_NOSLIP) {
		for (i=0; i<=ncx; i++)
			U[ncy+1][i] = -U[ncy][i];
		for (i=1; i<=ncx; i++)
			V[ncy][i] = 0.0;
	}
	else if (boundary_north == BOUNDARY_TYPE_FREESLIP) {
		for (i=0; i<=ncx; i++)
			U[ncy+1][i] = U[ncy][i];
		for (i=1; i<=ncx; i++)
			V[ncy][i] = 0.0;
	}

	// Bottom Boundary: wall (no-slip)
//	for (size_t i=0; i <= (ncx+1); i++) {
//		U[0][i] = -U[1][i];
//		V[0][i] = 0;
//	}
    //south
	if (boundary_south == BOUNDARY_TYPE_INLET) {
		for (i=0; i<=ncx; i++)
			U[0][i] = inlet_velocity_x*2.0 - U[1][i];
		for (i=1; i<=ncx; i++)
			V[0][i] = inlet_velocity_y;
	}
	else if (boundary_south == BOUNDARY_TYPE_OUTLET) {
		for (i=0; i<=ncx; i++)
			U[0][i] = U[1][i];
		for (i=1; i<=ncx; i++)
			V[0][i] = V[1][i];
	}
	else if (boundary_south == BOUNDARY_TYPE_NOSLIP) {
		for (i=0; i<=ncx; i++)
			U[0][i] = -U[1][i];
		for (i=1; i<=ncx; i++)
			V[0][i] = 0.0;
	}
	else if (boundary_south == BOUNDARY_TYPE_FREESLIP) {
		for (i=0; i<=ncx; i++)
			U[0][i] = U[1][i];
		for (i=1; i<=ncx; i++)
			V[0][i] = 0.0;
	}

	// Left Boundary: Inflow
//	for (size_t j=1; j <= ncy; j++) {
//		U[j][0] = inlet_velocity_x;
//		V[j][0] = inlet_velocity_y*2.0 - V[j][1];
//	}
    //west
	if (boundary_west == BOUNDARY_TYPE_INLET) {
		for (j=1; j <= ncy; j++)
			U[j][0] = inlet_velocity_x;
		for (j=0; j <= ncy; j++)
			V[j][0] = inlet_velocity_y*2.0 - V[j][1];
	}
	else if (boundary_west == BOUNDARY_TYPE_OUTLET) {
		for (j=1; j <= ncy; j++)
			U[j][0] = U[j][1];
		for (j=0; j <= ncy; j++)
			V[j][0] = V[j][1];
	}
	else if (boundary_west == BOUNDARY_TYPE_NOSLIP) {
		for (j=1; j <= ncy; j++)
			U[j][0] = 0.0;
		for (j=0; j <= ncy; j++)
			V[j][0] = -V[j][1];
	}
	else if (boundary_west == BOUNDARY_TYPE_FREESLIP) {
		for (j=1; j <= ncy; j++)
			U[j][0] = 0.0;
		for (j=0; j <= ncy; j++)
			V[j][0] = V[j][1];
	}

	// Right Boundary: outflow
//	for (size_t j=1; j <= ncy; j++) {
//		U[j][ncx  ] = U[j][ncx-1];
//		U[j][ncx+1] = U[j][ncx  ];
//		V[j][ncx+1] = V[j][ncx  ];
//	}
	if (boundary_east == BOUNDARY_TYPE_INLET) {
		for (j=1; j <= ncy; j++)
			U[j][ncx] = inlet_velocity_x;
		for (j=0; j <= ncy; j++)
			V[j][ncx+1] = inlet_velocity_y*2.0 - V[j][ncx];
	}
	else if (boundary_east == BOUNDARY_TYPE_OUTLET) {
		for (j=1; j <= ncy; j++)
			U[j][ncx] = U[j][ncx-1];
		for (j=0; j <= ncy; j++)
			V[j][ncx+1] = V[j][ncx];
	}
	else if (boundary_east == BOUNDARY_TYPE_NOSLIP) {
		for (j=1; j <= ncy; j++)
			U[j][ncx] = 0.0;
		for (j=0; j <= ncy; j++)
			V[j][ncx+1] = -V[j][ncx];
	}
	else if (boundary_east == BOUNDARY_TYPE_FREESLIP) {
		for (j=1; j <= ncy; j++)
			U[j][ncx] = 0.0;
		for (j=0; j <= ncy; j++)
			V[j][ncx+1] = V[j][ncx];
	}
	return;
}

void NS::update_boundaries_fg(
		double**& U,
		double**& V,
		double**& F,
		double**& G)  /// Input/Output
{
	// Left Boundary:
	for (size_t j=1; j<=ncy; j++)
		F[j][0] = U[j][0];

	// Right Boundary:
	for (size_t j=1; j<=ncy; j++)
		F[j][ncx] = U[j][ncx];

	// Top Boundary:
	for (size_t i=1; i<=ncx; i++)
		G[ncy][i] = V[ncy][i];

	// Bottom Boundary:
	for (size_t i=1; i<=ncx; i++)
		G[0][i] = V[0][i];
	return;
}

void NS::update_boundaries_p(
		double** &P)  /// Input/Output
{
	// Left Boundary:
	for (size_t j=1; j<=ncy; j++)
		P[j][0] = P[j][1];

	// Right Boundary:
	for (size_t j=1; j<=ncy; j++)
		P[j][ncx+1] = P[j][ncx];

	// Top Boundary:
	for (size_t i=1; i<=ncx; i++)
		P[ncy+1][i] = P[ncy][i];

	// Bottom Boundary:
	for (size_t i=1; i<=ncx; i++)
		P[0][i] = P[1][i];
	return;
}

void NS::update_domain_uv(
		int **& M, /// Input
		double **&    U, /// Input/Output
		double **&    V) /// Input/Output
{
	for (size_t j=1; j<=ncy; j++) {
		for (size_t i=1; i<=ncx; i++) {
			if(M[j][i] == FLUID) { //fluid cell
				continue;
			} else if (M[j][i] == B_N) { // North edge cell
				U[j][i] = -U[j+1][i];
				V[j][i] = 0.0;
			} else if (M[j][i] == B_S) { // South edge cell
				U[j][i] = -U[j-1][i];
				V[j][i] = 0.0;
				V[j-1][i] = 0.0;
			} else if (M[j][i] == B_E) { // East edge cell
				U[j][i] = 0.0;
				V[j][i] = -V[j][i+1];
			} else if (M[j][i] == B_W) { // West edge cell
				U[j][i] = 0.0;
				V[j][i] = -V[j][i-1];
				U[j][i-1] = 0.0;
			} else if (M[j][i] == B_NE) { // North-east corner cell
				U[j][i] = 0.0;
				V[j][i] = 0.0;
			} else if (M[j][i] == B_SE) { // South-east corner cell
				U[j][i] = 0.0;
				V[j][i] = -V[j][i+1];
				V[j-1][i] = 0.0;
			} else if (M[j][i] == B_NW) { // North-west corner cell
				U[j][i] = -U[j+1][i];
				V[j][i] = 0.0;
				U[j][i-1] = 0.0;
			} else if (M[j][i] == B_SW) { // South-west corner cell
				U[j][i] = -U[j-1][i];
				V[j][i] = -V[j][i-1];
				U[j][i-1] = 0.0;
				V[j-1][i] = 0.0;
			} else if (M[j][i] == B_IN) {
				U[j][i] = 0.0;
				V[j][i] = 0.0;
			} else { //forbidden cases
				cout << "Forbidden cell. Simulation abort! "
					 << "@update_domain_uv: "
					 << " M[" << j << "][" << i << "] = " << M[j][i] << endl;
				exit(EXIT_FAILURE);
			}
		}
	}
	return;
}

void NS::update_domain_fg(
		int**&    M,	/// Input
		double**& U,	/// Input
		double**& V,	/// Input
		double**& F,	/// Input/Output
		double**& G)	/// Input/Output
{
	for (size_t j=1; j<=ncy; j++) {
		for (size_t i=1; i<=ncx; i++) {
			if(M[j][i] == FLUID) { //fluid cell
				continue;
			} else if (M[j][i] == B_N) {
				G[j][i] = V[j][i];
			} else if (M[j][i] == B_S) {
				G[j][i] = V[j][i];
				G[j-1][i] = V[j-1][i];
			} else if (M[j][i] == B_E) {
				F[j][i] = U[j][i];
			} else if (M[j][i] == B_W) {
				F[j][i] = U[j][i];
				F[j][i-1] = U[j][i-1];
			} else if (M[j][i] == B_NE) {
				F[j][i] = U[j][i];
				G[j][i] = V[j][i];
			} else if (M[j][i] == B_SE) {
				F[j][i] = U[j][i];
				G[j-1][i] = V[j-1][i];
			} else if (M[j][i] == B_NW) {
				F[j][i-1] = U[j][i-1];
				G[j][i] = V[j][i];
			} else if (M[j][i] == B_SW) {
				F[j][i-1] = U[j][i-1];
				G[j-1][i] = V[j-1][i];
			} else if (M[j][i] == B_IN) {
				F[j][i] = U[j][i];
				G[j][i] = V[j][i];
			} else { //forbidden cases
				cout << "Forbidden cell. Simulation abort! "
					 << "@update_domain_fg: "
					 << " M[" << j << "][" << i << "] = " << M[j][i] << endl;
				exit(EXIT_FAILURE);
			}
		}
	}
	return;
}

void NS::update_domain_p(
		int**&    M, 	/// Input
		double**& P)	/// Input/Output
{
	for (size_t j=1; j<=ncy; j++) {
		for (size_t i=1; i<=ncx; i++) {
			if (M[j][i] == FLUID) { //fluid cell
				continue;
			} else if (M[j][i] == B_N) {
				P[j][i] = P[j+1][i];
			} else if (M[j][i] == B_S) {
				P[j][i] = P[j-1][i];
			} else if (M[j][i] == B_E) {
				P[j][i] = P[j][i+1];
			} else if (M[j][i] == B_W) {
				P[j][i] = P[j][i-1];
			} else if (M[j][i] == B_NE) {
				P[j][i] = (P[j+1][i] + P[j][i+1]) / 2.0;
			} else if (M[j][i] == B_SE) {
				P[j][i] = (P[j-1][i] + P[j][i+1]) / 2.0;
			} else if (M[j][i] == B_NW) {
				P[j][i] = (P[j+1][i] + P[j][i-1]) / 2.0;
			} else if (M[j][i] == B_SW) {
				P[j][i] = (P[j-1][i] + P[j][i-1]) / 2.0;
			} else if (M[j][i] == B_IN) {
				P[j][i] = 0;
			} else { //forbidden cases
				cout << "Forbidden cell. Simulation abort! "
					 << "@update_domain_p: "
					 << " M[" << j << "][" << i << "] = " << M[j][i] << endl;
				exit(EXIT_FAILURE);
			}
		}
	}
	return;
}

void NS::compute_dt(
		double**& U,  /// Input
		double**& V,  /// Input
		double&  dt)  /// Output
{
	double umax = fabs(U[0][0]);
	for (size_t j=0; j<=ncy+1; j++)
		for (size_t i=0; i<=ncx+1; i++)
			if(umax < fabs(U[j][i])) umax = fabs(U[j][i]);

	double vmax = fabs(V[0][0]);
	for (size_t j=0; j<=ncy+1; j++)
		for (size_t i=0; i<=ncx+1; i++)
			if(vmax < fabs(V[j][i])) vmax = fabs(V[j][i]);

	dt = tau * fmin((re*dx*dx*dy*dy)/(2.0*(dx*dx + dy*dy)), fmin(dx/umax, dy/vmax));
	return;
}

void NS::compute_fg(
	double&  dt,	/// Input
	double**& U,	/// Input
	double**& V,	/// Input
	double**& F,	/// Output
	double**& G)	/// Output
{
	/* the traversals for F and G differ slightly, for cache efficiency */

	/* we compute F and G in separate loops, since as their traversals differ,
	   one single loop to compute both would be quite complicated. Also if
	   the resolution is large, keeping all of the variables U, V, F and G
	   together in cache might be impossible */

	//compute F
	for (size_t j=1; j<=ncy; j++) {
		F[j][0] = U[j][0];

		for (size_t i=1; i<=ncx; i++) {
			F[j][i] = U[j][i] + dt * ( laplacian(U,j,i,dx,dy)/re
					                   - FD_x_U2(U,j,i,dx,alpha)
					                   - FD_y_UV(U,V,j,i,dy,alpha)
					                   + external_force_x );
		}
		F[j][ncx] = U[j][ncx];
		F[j][ncx+1] = U[j][ncx+1]; // remove?
	}
	//compute G
	for (size_t i=1; i<=ncx; i++) {
		G[0][i] = V[0][i];
	}
	for (size_t j=1; j<=ncy; j++) {
		for (size_t i=1; i<=ncx; i++) {
			G[j][i] = V[j][i] + dt * ( laplacian(V,j,i,dx,dy)/re
					                   - FD_x_UV(U,V,j,i,dx,alpha)
					                   - FD_y_V2(V,j,i,dy,alpha)
					                   + external_force_y );
		}
	}
	for (size_t i=1; i<=ncx; i++) {
		G[ncy][i] = V[ncy][i];
		G[ncy+1][i] = V[ncy+1][i]; // remove?
	}
	return;
}

void NS::compute_uv(
		double&  dt,	/// Input
		double**& F,	/// Input
		double**& G,	/// Input
		double**& P,	/// Input
		double**& U, 	/// Output
		double**& V)	/// Output
{
	for (size_t j=1; j<=ncy; j++)
		for (size_t i=1; i<=ncx-1; i++)
			U[j][i] = F[j][i] - dt * (P[j][i+1] - P[j][i]) / dx;

	for (size_t j=1; j<=ncy-1; j++)
		for (size_t i=1; i<=ncx; i++)
			V[j][i] = G[j][i] - dt * (P[j+1][i] - P[j][i]) / dy;

	return;
}

void NS::solve_for_p_sor(
		double& dt,
		double**& F,
		double**& G,
		double**& P)
{
	size_t ITERMAX = 10000;
	double TOLERANCE = 0.0001;

	double inv_dx2 = 1.0 / (dx*dx);
	double inv_dy2 = 1.0 / (dy*dy);
	double inv_dt_dx = 1.0 / (dt*dx);
	double inv_dt_dy = 1.0 / (dt*dy);
	double a = 1.0 - omega;
	double b = 0.5 * omega / (inv_dx2 + inv_dy2);

	double res, tmp;
	for (size_t it=0; it < ITERMAX; it++) {
		// Swip over P
		for (size_t j=1; j<=ncy; j++) {
			for (size_t i=1; i<=ncx; i++) {
				P[j][i] = a * P[j][i] + b * ( inv_dx2*(P[j][i+1]+P[j][i-1])
											+ inv_dy2*(P[j+1][i]+P[j-1][i])
											- (inv_dt_dx*(F[j][i]-F[j][i-1]) +
											   inv_dt_dy*(G[j][i]-G[j-1][i]))
											);
			}
		}
		// Compute residual
		res = 0.0;
		for (size_t j=1; j<=ncy; j++) {
			for (size_t i=1; i<=ncx; i++) {
				tmp = inv_dx2 * (P[j][i+1]-2.0*P[j][i]+P[j][i-1]) +
				      inv_dy2 * (P[j+1][i]-2.0*P[j][i]+P[j-1][i]) -
				     (inv_dt_dx * (F[j][i]-F[j][i-1]) +
				      inv_dt_dy * (G[j][i]-G[j-1][i]));
				res += tmp*tmp;
			}
		}
		res = sqrt( res/(ncy*ncx) );
		// Check residual
		if (res <= TOLERANCE) {
			cout << "SOR solver converged at iter " << it << endl;
			return;
		}
	}
	cout << "SOR solver did not converge for " << ITERMAX << " iterations!" << endl;
	return;
}

void NS::solve_for_p_direct(
		SparseMatrix<double>& A,	/// Input
		double**& RHS,				/// Input
		double**& P)				/// Output
{
	/**
	 * Use a bi conjugate gradient stabilized solver for sparse square problems:
	 * Solve for Ax = b, where A is a square sparse matrix
	 */
	BiCGSTAB<SparseMatrix<double> > solver;  // setup solver
	solver.setMaxIterations(10000);
	solver.setTolerance(0.000001);
	solver.compute(A);

	size_t i,j,n;

	// Vectorize right-hand-side array
	VectorXd b(ncy*ncx);
	for (size_t j=1; j<=ncy; j++) {
		for (size_t i=1; i<=ncx; i++) {
			n = (j-1)*ncx + (i-1);
			b[n] = RHS[j][i];
		}
	}
	// Solve for pressure x
	VectorXd x(ncy*ncx);
	x = solver.solve(b);
	// Restore x into P array
	for (size_t j=1; j<=ncy; j++) {
		for (size_t i=1; i<=ncx; i++) {
			n = (j-1)*ncx + (i-1);
			P[j][i] = x[n];
		}
	}
	return;
}

SparseMatrix<double> NS::create_system_matrix()
{
	size_t n;
	size_t N = ncy*ncx;
	SparseMatrix<double> A (N, N); // System matrix excluding boundaries

	double xnei = 1/(dx*dx);
	double ynei = 1/(dy*dy);
	double diag = -2.0 * (xnei + ynei);

	std::vector<Triplet<double> > tripletList;
	tripletList.reserve(N*5);

	// NOTE: row-major storage, i.e.,
	//       cell index: n = (j-1)*ncx + (i-1), where j=1:ncy, i=1:ncx
	for (size_t j=1; j<=ncy; j++) {
		for (size_t i=1; i<=ncx; i++) {
		// cell index
		n = (j-1)*ncx + (i-1);

		// Set self (diagonal)
		tripletList.push_back( Triplet<double>(n,n,diag) );

		// NOTE: left-most column's WEST neighbor are boundary cells,
		//       which will be added in RHS
		// Same for right-most column's EAST neighbors,
		//      top row's NORTH neighbors, and bottom row's SOUTH neighbors

		// Set WEST neighbor (if not the left-most column)
		if (i != 1)
			tripletList.push_back( Triplet<double>(n,n-1,xnei) );

		// Set EAST neighbor (if not the right-most column)
		if (i != ncx)
			tripletList.push_back( Triplet<double>(n,n+1,xnei) );

		// Set NORTH neighbor (if not the top row)
		if (j != ncy)
			tripletList.push_back( Triplet<double>(n,n+ncx,ynei) );

		// Set SOUTH neighbor (if not the bottom row)
		if (j != 1)
			tripletList.push_back( Triplet<double>(n,n-ncx,ynei) );
		}
	}
	A.setFromTriplets(tripletList.begin(),tripletList.end());
	tripletList.clear();
	return A;
}

void NS::compute_rhs(
	double & dt,	/// Input
	double**& F,	/// Input
	double**& G,	/// Input
	double**& P,	/// Input
	double**& RHS)	/// Output
{
	size_t i,j;
	// Inner domain
	for(size_t j=1; j<=ncy; j++) {
		for(size_t i=1; i<=ncx; i++) {
			RHS[j][i] = ((F[j][i] - F[j][i-1])/dx + (G[j][i] - G[j-1][i])/dy)/dt;
		}
	}
	// Left-most column's WEST neighbor
	for(size_t j=1; j<=ncy; j++) {
		// i = 1
		RHS[j][1] -= P[j][0]/(dx*dx);
	}
	// Right-most column's EAST neighbor
	for(size_t j=1; j<=ncy; j++) {
		// i = ncx
		RHS[j][ncx] -= P[j][ncx+1]/(dx*dx);
	}
	// Top row's NORTH neighbor
	for(size_t i=1; i<=ncx; i++) {
		// j = ncy
		RHS[ncy][i] -= P[ncy+1][i]/(dy*dy);
	}
	// Bottom row's SOUTH neighbor
	for(size_t i=1; i<=ncx; i++) {
		// j = 1
		RHS[1][i] -= P[0][i]/(dy*dy);
	}
	return;
}

/*****************************************
 * Visualization related helper methods
 *****************************************/

void NS::write_vtk_file(
		const char *szProblem,
		int timeStepNumber,
		int**&    M,
		double**& U,
		double**& V,
		double**& P)
{
	char outfile[80];
	sprintf(outfile, "%s.%i.vtk", szProblem, timeStepNumber );
	std::ofstream fout;
	fout.open(outfile, std::ofstream::out);
	if (!fout.is_open()) {
		std::cout << "Fail to open output VTK file. Operation aborted!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// write header and coordinates
	write_vtk_header_coord(fout);
	// write velocity
	fout << "POINT_DATA " << (ncy+1)*(ncx+1) << std::endl;
	fout << "VECTORS velocity float" << std::endl;
	for(size_t j=0; j<=ncy; j++) {
		for(size_t i=0; i<=ncx; i++) {
			if (is_point_in_obs_cell(M,j,i))
				fout << 0.0 << " " << 0.0 << " " << 0.0 << std::endl;
			else
				fout << (U[j][i] + U[j+1][i]) * 0.5 << " "
				     << (V[j][i] + V[j][i+1]) * 0.5 << " "
					 << 0.0 << std::endl;
		}
	}
	// write pressure
	fout << std::endl;
	fout << "CELL_DATA " << (ncy)*(ncx) << std::endl;
	fout << "SCALARS pressure float 1 \n" << std::endl;
	fout << "LOOKUP_TABLE default \n" << std::endl;
	for(size_t j=1; j<=ncy; j++) {
		for(size_t i=1; i<=ncx; i++) {
			if (M[j][i] == 0)
				fout << P[j][i] << std::endl;
			else
				fout << 0.0 << std::endl;
		}
	}
	fout.close();
	return;
}

void NS::write_vtk_header_coord(std::ofstream& fout)
{
	fout << "# vtk DataFile Version 2.0" << std::endl;
	fout << "generated by CombiSim" << std::endl;
	fout << "ASCII" << std::endl;
	fout << std::endl;
	fout << "DATASET STRUCTURED_GRID" << std::endl;
	fout << "DIMENSIONS  "<< ncx+1 << " " << ncy+1 << " " << 1 << std::endl;
	fout << "POINTS " << (ncx+1)*(ncy+1) <<" float" << std::endl;
	fout << std::endl;
	double originX = 0.0;
	double originY = 0.0;
	for(size_t j=0; j<=ncy; j++)
		for(size_t i=0; i<=ncx; i++)
			fout << originX+(i*dx) << " "
			     << originY+(j*dy) << " " << 0 << std::endl;
	return;
}


