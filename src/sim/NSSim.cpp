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

#include <sim/NSSim.hpp>

#define IMPLAUSIBLE_VALUE -100.0
#define USE_DIRECT_SOLVER 0

using namespace std;
using namespace Eigen;

// Necessary for static const arrays
constexpr double NSSim::OBS_XMIN[];
constexpr double NSSim::OBS_XMAX[];
constexpr double NSSim::OBS_YMIN[];
constexpr double NSSim::OBS_YMAX[];


/************************************
 *  Public Methods
 ************************************/

NSSim::NSSim(
		std::size_t num_cells_x,
		std::size_t num_cells_y)
{
	ncx = num_cells_x;
	ncy = num_cells_y;
	dx = DOMAIN_SIZE_X/double(num_cells_x);
	dy = DOMAIN_SIZE_Y/double(num_cells_y);
}

void NSSim::run()
{
	double t = 0.0;
	double dt = 0.0;
	double t_end = 5.0;

	double vtk_freq = 0.05;
	int vtk_cnt = 0;
	std::string vtk_outfile = "out/nssim";

	/**
	 * Create domain mask
	 * 	  "0" for a fluid cell
	 * 	  "1xxxx" for an obstacle cell
	 */
	int** M = create_geometry_mask();
//	tools::print_matrix<int>(M, ncy+2, ncx+2, true); // DEBUG only

	/**
	 * 2D Arrays: Row-major storage, including boundary cells
	 * 	 dim_0 along y-direction (rows,    j, ncy)
	 * 	 dim_1 along x-direction (columns, i, ncx)
	 *
	 * 	 Inner domain: m[j][i], i=1..ncx, j=1..ncy
	 * 	 Boundaries: LEFT   m[j][0],
	 * 	             RIGHT  m[j][ncx+1],
	 * 	             TOP    m[ncy+1][i],
	 * 	             BOTTOM m[0][i]
	 */
	// allocate arrays
	double** U = tools::alloc_matrix<double>(ncy+2, ncx+2, true); /// velocity in x-direction
	double** V = tools::alloc_matrix<double>(ncy+2, ncx+2, true); /// velocity in y-direction
	double** P = tools::alloc_matrix<double>(ncy+2, ncx+2, true); /// pressure
	double** F = tools::alloc_matrix<double>(ncy+2, ncx+2, true); /// temporary array: F value
	double** G = tools::alloc_matrix<double>(ncy+2, ncx+2, true); /// temporary array: G value
    // initialize arrays
	tools::init_matrix<double>(U, ncy+2, ncx+2, true, INITIAL_VELOCITY_X);
	tools::init_matrix<double>(V, ncy+2, ncx+2, true, INITIAL_VELOCITY_Y);
	tools::init_matrix<double>(P, ncy+2, ncx+2, true, INITIAL_PRESSURE);
	tools::init_matrix<double>(F, ncy+2, ncx+2, true, 0.0);
	tools::init_matrix<double>(G, ncy+2, ncx+2, true, 0.0);

#if (USE_DIRECT_SOLVER == 1)
	/**
	 * System matrix for the pressure equation: A*p = rhs
	 */
	SparseMatrix<double> A = create_system_matrix();
	double** RHS = tools::alloc_matrix<double>(ncy+2, ncx+2, true);
#endif

	update_boundaries_uv(U, V);
	update_domain_uv(M, U, V);

	write_vtk_file(vtk_outfile.c_str(), vtk_cnt, M, U, V, P);
	vtk_cnt ++;

	while(t < t_end)
	{
		/* 1. Compute time step */
		compute_dt(U, V, dt);

		/* 2. Compute F and G */
		compute_fg(dt, U, V, F, G);
		update_boundaries_fg(U, V, F, G);
		update_domain_fg(M, U, V, F, G);

		/* 3. Solve the pressure equation: A*p = rhs */
#if (USE_DIRECT_SOLVER == 1)
		compute_rhs(dt, F, G, P, RHS);
		solve_for_p_direct(A, RHS, P);
#else
		solve_for_p_sor(dt, M, F, G, P);
#endif
		update_boundaries_p(P);
		update_domain_p(M, P);

		/* 4. Compute velocity U,V */
		compute_uv(dt, F, G, P, U, V);
		update_boundaries_uv(U, V);
		update_domain_uv(M, U, V);

		/* 5. Update simulation time */
		t += dt;
		cout << "Simulation completed at t = " << t << ", dt = " << dt << endl;

		/* 6. Generate output data */
		if (floor(t/vtk_freq) >= vtk_cnt) {
			cout << "Writing VTK output at t = " << t << endl;
			write_vtk_file(vtk_outfile.c_str(), vtk_cnt, M, U, V, P);
			vtk_cnt ++;
		}
	} // end while

	tools::free_matrix<int>(M);
	tools::free_matrix<double>(U);
	tools::free_matrix<double>(V);
	tools::free_matrix<double>(P);
	tools::free_matrix<double>(F);
	tools::free_matrix<double>(G);
#if (USE_DIRECT_SOLVER == 1)
	tools::free_matrix<double>(RHS);
	A.resize(0,0);
#endif
	return;
}

/*****************************************
 * Internal core functions
 *****************************************/

int** NSSim::create_geometry_mask()
{
	/************************************************
	 *  Fluid cells are marked with "0".
	 *  Non-fluid cells are marked with "1xxxx".
	 *
	 *  Each non-fluid cells contains 5-bit:
	 *      [ C  E  W  S  N ]
	 *   C: center, 		cell [j  ][i  ]
	 *   E: east  neighbor, cell [j  ][i+1]
	 *   W: west  neighbor, cell [j  ][i-1]
	 *   N: north neighbor, cell [j+1][i  ]
	 *   S: south neighbor, cell [j-1][i  ]
	 *
	 *  "1" indicates an obstacle cell
	 *  "0" indicates a fluid cell
	 * **********************************************/

	size_t j; 			// index along y-direction
	size_t i; 			// index along x-direction
	size_t k;
	size_t nrl, nrh; 	// low & high index of rows
	size_t ncl, nch; 	// low & high index of columns

	int** M = tools::alloc_matrix<int>(ncy+2, ncx+2, true);
	tools::init_matrix<int>(M, ncy+2, ncx+2, true, 0);

	// 1. Initialize all non-fluid cells to 10000
	// 1.1 Setup boundaries (corner cells belongs to WALL type boundary)
	if (BOUNDARY_LEFT == 3) {
		for (j=0; j<=ncy+1; j++)
			M[j][0] = B_E;
	}
	if (BOUNDARY_RIGHT == 3) {
		for (j=0; j<=ncy+1; j++)
			M[j][ncx+1] = B_W;
	}
	if (BOUNDARY_TOP == 3) {
		for (i=0; i<=ncx+1; i++)
			M[ncy+1][i] = B_S;
	}
	if (BOUNDARY_BOTTOM == 3) {
		for (i=0; i<=ncx+1; i++)
			M[0][i] = B_N;
	}
	// 1.2 Setup inner domain
	for (k=0; k < NUM_OBS; k++) {
		ncl = size_t(round(OBS_XMIN[k]/dx) + 1);
		nch = size_t(round(OBS_XMAX[k]/dx));
		nrl = size_t(round(OBS_YMIN[k]/dy) + 1);
		nrh = size_t(round(OBS_YMAX[k]/dy));
		for (j=nrl; j<=nrh; j++)
			for (i=ncl; i<=nch; i++)
				M[j][i] = 10000;
	}
	// 1.3 Re-check boundaries
	if (BOUNDARY_LEFT == 3) {
		for (j=0; j<=ncy+1; j++)
			if (M[j][1] > 0) M[j][0] = B_IN;
	}
	if (BOUNDARY_RIGHT == 3) {
		for (j=0; j <= ncy+1; j++)
			if (M[j][ncx] > 0) M[j][ncx+1] = B_IN;
	}
	if (BOUNDARY_TOP == 3) {
		for (i=0; i<=ncx+1; i++)
			if (M[ncy][i] > 0) M[ncy+1][i] = B_IN;
	}
	if (BOUNDARY_BOTTOM == 3) {
		for (i=0; i<=ncx+1; i++)
			if (M[1][i] > 0) M[0][i] = B_IN;
	}
	//2. set up inner domain non-fluid cell types
	for (j=1; j<=ncy; j++) {
		for (i=1; i<=ncx; i++) {
			if (M[j][i] > 0) {
				if (M[j][i+1] > 0) M[j][i] += 1000; //East
				if (M[j][i-1] > 0) M[j][i] += 100;  //West
				if (M[j+1][i] > 0) M[j][i] += 10;   //North
				if (M[j-1][i] > 0) M[j][i] += 1;    //South
			}
		}
	} // end of for(j)
	return M;
}

void NSSim::update_boundaries_uv(
		double** &U,  /// Input/Output
		double** &V)  /// Input/Output
{
	// Left Boundary: Inflow
	for (size_t j=1; j <= ncy; j++) {
		U[j][0] = INFLOW_VELOCITY_X;
		V[j][0] = INFLOW_VELOCITY_Y*2.0 - V[j][1];
	}
	// Right Boundary: outflow
	for (size_t j=1; j <= ncy; j++) {
		U[j][ncx  ] = U[j][ncx-1];
		U[j][ncx+1] = U[j][ncx  ];
		V[j][ncx+1] = V[j][ncx  ];
	}
	// Top Boundary: wall (no-slip)
	for (size_t i=0; i <= (ncx+1); i++) {
		U[ncy+1][i] = -U[ncy][i];
		V[ncy  ][i] = 0.0;
	}
	// Bottom Boundary: wall (no-slip)
	for (size_t i=0; i <= (ncx+1); i++) {
		U[0][i] = -U[1][i];
		V[0][i] = 0;
	}
	return;
}

void NSSim::update_boundaries_fg(
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

void NSSim::update_boundaries_p(
		double**& P)  /// Input/Output
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

void NSSim::update_domain_uv(
		int**&    M, /// Input
		double**& U, /// Input/Output
		double**& V) /// Input/Output
{
	for (size_t j=1; j<=ncy; j++) {
		for (size_t i=1; i<=ncx; i++) {
			if(M[j][i] == 0) { //fluid cell
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

void NSSim::update_domain_fg(
		int**&    M,	/// Input
		double**& U,	/// Input
		double**& V,	/// Input
		double**& F,	/// Input/Output
		double**& G)	/// Input/Output
{
	for (size_t j=1; j<=ncy; j++) {
		for (size_t i=1; i<=ncx; i++) {
			if(M[j][i] == 0) { //fluid cell
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

void NSSim::update_domain_p(
		int**&    M, 	/// Input
		double**& P)	/// Input/Output
{
	for (size_t j=1; j<=ncy; j++) {
		for (size_t i=1; i<=ncx; i++) {
			if (M[j][i] == 0) { //fluid cell
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

void NSSim::compute_dt(
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

	dt = TAU * fmin((RE*dx*dx*dy*dy)/(2.0*(dx*dx + dy*dy)), fmin(dx/umax, dy/vmax));
	return;
}

void NSSim::compute_fg(
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
			F[j][i] = U[j][i] + dt * ( laplacian(U,j,i,dx,dy)/RE
					                   - FD_x_U2(U,j,i,dx,ALPHA)
					                   - FD_y_UV(U,V,j,i,dy,ALPHA)
					                   + EXTERNAL_FORCE_X );
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
			G[j][i] = V[j][i] + dt * ( laplacian(V,j,i,dx,dy)/RE
					                   - FD_x_UV(U,V,j,i,dx,ALPHA)
					                   - FD_y_V2(V,j,i,dy,ALPHA)
					                   + EXTERNAL_FORCE_Y );
		}
	}
	for (size_t i=1; i<=ncx; i++) {
		G[ncy][i] = V[ncy][i];
		G[ncy+1][i] = V[ncy+1][i]; // remove?
	}
	return;
}

void NSSim::compute_uv(
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

void NSSim::solve_for_p_sor(
		double& dt,
		int**& M,
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
	double a = 1.0 - OMEGA;
	double b = 0.5 * OMEGA / (inv_dx2 + inv_dy2);

	double res, tmp;
	for (size_t it=0; it < ITERMAX; it++) {
		// Swip over P
		for (size_t j=1; j<=ncy; j++) {
			for (size_t i=1; i<=ncx; i++) {
				P[j][i] = a * P[j][i] + b * (
						inv_dx2*(P[j][i+1]+P[j][i-1]) + inv_dy2*(P[j+1][i]+P[j-1][i])
						- (inv_dt_dx*(F[j][i]-F[j][i-1]) + inv_dt_dy*(G[j][i]-G[j-1][i]))
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

void NSSim::solve_for_p_direct(
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

SparseMatrix<double> NSSim::create_system_matrix()
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

void NSSim::compute_rhs(
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

void NSSim::write_vtk_file(
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

void NSSim::write_vtk_header_coord(std::ofstream& fout)
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

