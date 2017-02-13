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

#include <sim/NSSimDist.hpp>


using namespace std;

NSSimDist::NSSimDist(
			std::size_t num_cells_x,
			std::size_t num_cells_y,
			MPIObject* mpi_obj)
{
	mpi = mpi_obj;
	block.reset( DomainDecomposer2D::gen_block(mpi->size,
					1, num_cells_x, 1, num_cells_y, mpi->rank) );
	gncx = num_cells_x;
	gncy = num_cells_y;
	lncx = (block->id < 0) ? 0 : block->xmax - block->xmin + 1;
	lncy = (block->id < 0) ? 0 : block->ymax - block->ymin + 1;
	dx = NSSim::DOMAIN_SIZE_X / double(gncx);
	dy = NSSim::DOMAIN_SIZE_Y / double(gncy);

#if (ENABLE_IMPI == YES)
	CarryOver.M = nullptr;
	CarryOver.U = nullptr;
	CarryOver.V = nullptr;
	CarryOver.P = nullptr;
	CarryOver.F = nullptr;
	CarryOver.G = nullptr;
	CarryOver.t = 0.0;
	CarryOver.tic = 0.0;
	CarryOver.vtk_cnt = 0;
#endif
}

void NSSimDist::run()
{
	double t = 0.0;
	double dt = 0.0;
	double t_end = 5.0;

	int vtk_cnt = 0;
	double vtk_freq = 0.1;
	std::string vtk_outfile = string(OUTPUT_PATH) + "/nssim";

#if (ENABLE_IMPI == YES)
	double toc = 0;
	double tic = MPI_Wtime();
#endif

	int**    M;	/// Geometry mask: cell types
	double** U;	/// velocity in x-direction
	double** V;	/// velocity in y-direction
	double** P;	/// pressure
	double** F;	/// temporary array: F value
	double** G;	/// temporary array: G value

#if (ENABLE_IMPI == YES)
	if (mpi->status != MPI_ADAPT_STATUS_JOINING) {
#endif
	// 1. Create LOCAL computational arrays
	M = create_geometry_mask();
	U = tools::alloc_matrix<double>(lncy+2, lncx+2, true);
	V = tools::alloc_matrix<double>(lncy+2, lncx+2, true);
	P = tools::alloc_matrix<double>(lncy+2, lncx+2, true);
	F = tools::alloc_matrix<double>(lncy+2, lncx+2, true);
	G = tools::alloc_matrix<double>(lncy+2, lncx+2, true);
	// initialize arrays
	tools::init_matrix<double>(U, lncy+2, lncx+2, true, NSSim::INITIAL_VELOCITY_X);
	tools::init_matrix<double>(V, lncy+2, lncx+2, true, NSSim::INITIAL_VELOCITY_Y);
	tools::init_matrix<double>(P, lncy+2, lncx+2, true, NSSim::INITIAL_PRESSURE);
	tools::init_matrix<double>(F, lncy+2, lncx+2, true, 0.0);
	tools::init_matrix<double>(G, lncy+2, lncx+2, true, 0.0);

	// 3. Initial output
	update_boundaries_uv(U, V);
	update_domain_uv(M, U, V);
	exchange_ghost_layers_mpi(U, V);

	write_vtk_dist(vtk_outfile.c_str(), vtk_cnt, M, U, V, P);
	vtk_cnt ++;

#if (ENABLE_IMPI == YES)
	} else {
		M = CarryOver.M;
		U = CarryOver.U;
		V = CarryOver.V;
		P = CarryOver.P;
		F = CarryOver.F;
		G = CarryOver.G;
		t = CarryOver.t;
		tic = CarryOver.tic;
		vtk_cnt = CarryOver.vtk_cnt;
	}
#endif

	// 4. Main loop
	while(t < t_end)
	{
		// 4.1 Compute time step (MPI communication)
		compute_dt_mpi(U, V, dt);

		// 4.2 Compute F and G
		compute_fg(dt, U, V, F, G);
		exchange_ghost_layers_mpi(F, G);
		update_boundaries_fg(U, V, F, G);
		update_domain_fg(M, U, V, F, G);

		// 4.3 Solve the pressure equation: A*p = rhs
		solve_for_p_sor_mpi(dt, U, V, F, G, P);
		exchange_ghost_layers_mpi(P);
		update_boundaries_p(P);
		update_domain_p(M, P);

		// 4.4 Compute velocity U,V
		compute_uv(dt, F, G, P, U, V);
		exchange_ghost_layers_mpi(U, V);
		update_boundaries_uv(U, V);
		update_domain_uv(M, U, V);

		// 4.5 Update simulation time
		t += dt;
		if (mpi->rank == MASTER)
			cout << "Simulation completed at t = " << t
					<< ", dt = " << dt << endl;

		// 4.6 Write VTK
		if (floor(t/vtk_freq) >= vtk_cnt) {
			if (mpi->rank == MASTER)
				cout << "Writing VTK output at t = " << t << endl;
			write_vtk_dist(vtk_outfile.c_str(), vtk_cnt, M, U, V, P);
			vtk_cnt ++;
		}

#if (ENABLE_IMPI == YES)
		toc = MPI_Wtime()-tic;
		if (toc >= IMPI_ADAPT_INTERVAL) {

			CarryOver.M = M;
			CarryOver.U = U;
			CarryOver.V = V;
			CarryOver.P = P;
			CarryOver.F = F;
			CarryOver.G = G;
			CarryOver.t = t;
			CarryOver.tic = tic;
			CarryOver.vtk_cnt = vtk_cnt;

			impi_adapt();

			M = CarryOver.M;
			U = CarryOver.U;
			V = CarryOver.V;
			P = CarryOver.P;
			F = CarryOver.F;
			G = CarryOver.G;
			t = CarryOver.t;
			tic = CarryOver.tic;
			vtk_cnt = CarryOver.vtk_cnt;

			tic = MPI_Wtime(); // reset timer
		}
#endif
	} // end while
	tools::free_matrix<int>(M);
	tools::free_matrix<double>(U);
	tools::free_matrix<double>(V);
	tools::free_matrix<double>(P);
	tools::free_matrix<double>(F);
	tools::free_matrix<double>(G);
	return;
}

#if (ENABLE_IMPI == YES)
void NSSimDist::impi_adapt()
{
	int adapt_flag;
	MPI_Info info;
	MPI_Comm intercomm;
	MPI_Comm newcomm;
	double tic, toc;

	tic = MPI_Wtime();
	MPI_Probe_adapt(&adapt_flag, &mpi->status, &info);
	toc = MPI_Wtime() - tic;
	cout << "Rank " << mpi->rank << " [STATUS " << mpi->status << "]: "
			<< "MPI_Probe_adapt " << toc << " seconds" << endl;

	if (adapt_flag == MPI_ADAPT_TRUE){

		tic = MPI_Wtime();
		MPI_Comm_adapt_begin(&intercomm, &newcomm, 0, 0);
		toc = MPI_Wtime() - tic;
		cout << "Rank " << mpi->rank << " [STATUS " << mpi->status << "]: "
				<< "MPI_Comm_adapt_begin " << toc << " seconds" << endl;
		//************************ ADAPT WINDOW ****************************
		// Update and redistribute domain & data
		// This will update MPIObject, DomainBlock, and ALL local data array
		if (mpi->status != MPI_ADAPT_STATUS_RETREATING) {
			update_redistribute_domain_mpi(
					newcomm, CarryOver.M, CarryOver.U, CarryOver.V,
					CarryOver.P, CarryOver.F, CarryOver.G);
		}
		MPI_Bcast(&(CarryOver.t), 1, MPI_DOUBLE, MASTER, newcomm);
		MPI_Bcast(&(CarryOver.tic), 1, MPI_DOUBLE, MASTER, newcomm);
		MPI_Bcast(&(CarryOver.vtk_cnt), 1, MPI_INT, MASTER, newcomm);

		//************************ ADAPT WINDOW ****************************
		tic = MPI_Wtime();
		MPI_Comm_adapt_commit(&adapt_flag);
		toc = MPI_Wtime() - tic;
		cout << "Rank " << mpi->rank << " [STATUS " << mpi->status << "]: "
				<< "MPI_Comm_adapt_commit " << toc << " seconds" << endl;

		mpi->update(MPI_COMM_WORLD);
		mpi->status = MPI_ADAPT_STATUS_STAYING;
	}
	return;
}
#endif

/*********************************
 ******* Internal Methods  *******
 *********************************/

int** NSSimDist::create_geometry_mask()
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return nullptr;

	NSSim gsim (gncx, gncy);
	int** global_M = gsim.create_geometry_mask();
	int** M = tools::alloc_matrix<int>(lncy+2, lncx+2, true);
	for (size_t j=block->ymin-1; j<=block->ymax+1; j++) {
		for (size_t i=block->xmin-1; i<=block->xmax+1; i++) {
			M[j-(block->ymin-1)][i-(block->xmin-1)] = global_M[j][i];
		}
	}
	tools::free_matrix(global_M);
	return M;
}

void NSSimDist::update_boundaries_uv(
		double** &U,  		/// Input/Output
		double** &V)  		/// Input/Output
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;

	// NOTE: When the neighbor does not exist,
	//       it means the block is bordering boundary
	// Left Boundary: Inflow
	if (block->wnei < 0) {
		for (size_t j=1; j <= lncy; j++) {
			U[j][0] = NSSim::INFLOW_VELOCITY_X;
			V[j][0] = NSSim::INFLOW_VELOCITY_Y*2.0 - V[j][1];
		}
	}
	// Right Boundary: Outflow
	if (block->enei < 0) {
		for (size_t j=1; j <= lncy; j++) {
			U[j][lncx  ] = U[j][lncx-1];
			U[j][lncx+1] = U[j][lncx  ];
			V[j][lncx+1] = V[j][lncx  ];
		}
	}
	// Top Boundary: wall (no-slip)
	if (block->nnei < 0) {
		for (size_t i=0; i <= (lncx+1); i++) {
			U[lncy+1][i] = -U[lncy][i];
			V[lncy  ][i] = 0.0;
		}
	}
	// Bottom Boundary: wall (no-slip)
	if (block->snei < 0) {
		for (size_t i=0; i <= (lncx+1); i++) {
			U[0][i] = -U[1][i];
			V[0][i] = 0;
		}
	}
	return;
}

void NSSimDist::update_boundaries_fg(
		double**& U,	/// Input
		double**& V,	/// Input
		double**& F,	/// Input/Output
		double**& G)  	/// Input/Output
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;

	// Left Boundary:
	if (block->wnei < 0) {
		for (size_t j=1; j<=lncy; j++)
			F[j][0] = U[j][0];
	}
	// Right Boundary:
	if (block->enei < 0) {
		for (size_t j=1; j<=lncy; j++)
			F[j][lncx] = U[j][lncx];
	}
	// Top Boundary:
	if (block->nnei < 0) {
		for (size_t i=1; i<=lncx; i++)
			G[lncy][i] = V[lncy][i];
	}
	// Bottom Boundary:
	if (block->snei < 0) {
		for (size_t i=1; i<=lncx; i++)
			G[0][i] = V[0][i];
	}
	return;
}

void NSSimDist::update_boundaries_p(
		double**& P)  /// Input/Output
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;

	// Left Boundary:
	if (block->wnei < 0) {
		for (size_t j=1; j<=lncy; j++)
			P[j][0] = P[j][1];
	}
	// Right Boundary:
	if (block->enei < 0) {
		for (size_t j=1; j<=lncy; j++)
			P[j][lncx+1] = P[j][lncx];
	}
	// Top Boundary:
	if (block->nnei < 0) {
		for (size_t i=1; i<=lncx; i++)
			P[lncy+1][i] = P[lncy][i];
	}
	// Bottom Boundary:
	if (block->snei < 0) {
		for (size_t i=1; i<=lncx; i++)
			P[0][i] = P[1][i];
	}
	return;
}

void NSSimDist::update_domain_uv(
		int** &M,
		double** &U,
		double** &V)
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;

	// Check non-fluid cells
	// NOTE: we need to include ghost layers, but exclude boundary layers!!

	int jmin=0, jmax=int(lncy)+1;  // index boundary
	int imin=0, imax=int(lncx)+1;

	int nrl = (block->snei < 0) ? 1 : 0;  // looping boundary
	int nrh = (block->nnei < 0) ? int(lncy) : int(lncy)+1;
	int ncl = (block->wnei < 0) ? 1 : 0;
	int nch = (block->enei < 0) ? int(lncx) : int(lncx)+1;

	for (int j=nrl; j<=nrh; j++) {
		for (int i=ncl; i<=nch; i++) {
			if(M[j][i] == 0) { //fluid cell
				continue;
			} else if (M[j][i] == NSSim::B_N) { // North edge cell
				if ((j+1) <= jmax) U[j][i] = -U[j+1][i];
				V[j][i] = 0.0;
			} else if (M[j][i] == NSSim::B_S) { // South edge cell
				if ((j-1) >= jmin) {
					U[j][i] = -U[j-1][i];
					V[j-1][i] = 0.0;
				}
				V[j][i] = 0.0;
			} else if (M[j][i] == NSSim::B_E) { // East edge cell
				U[j][i] = 0.0;
				if ((i+1) <= imax) V[j][i] = -V[j][i+1];
			} else if (M[j][i] == NSSim::B_W) { // West edge cell
				U[j][i] = 0.0;
				if ((i-1) >= imin) {
					V[j][i] = -V[j][i-1];
					U[j][i-1] = 0.0;
				}
			} else if (M[j][i] == NSSim::B_NE) { // North-east corner cell
				U[j][i] = 0.0;
				V[j][i] = 0.0;
			} else if (M[j][i] == NSSim::B_SE) { // South-east corner cell
				U[j][i] = 0.0;
				if ((i+1) <= imax) V[j][i] = -V[j][i+1];
				if ((j-1) >= jmin) V[j-1][i] = 0.0;
			} else if (M[j][i] == NSSim::B_NW) { // North-west corner cell
				if ((j+1) <= jmax) U[j][i] = -U[j+1][i];
				V[j][i] = 0.0;
				if ((i-1) >= imin) U[j][i-1] = 0.0;
			} else if (M[j][i] == NSSim::B_SW) { // South-west corner cell
				if ((j-1) >= jmin) {
					U[j][i] = -U[j-1][i];
					V[j-1][i] = 0.0;
				}
				if ((i-1) >= imin) {
					V[j][i] = -V[j][i-1];
					U[j][i-1] = 0.0;
				}
			}
			else if (M[j][i] == NSSim::B_IN) {
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

void NSSimDist::update_domain_fg(
		int**&    M,	/// Input
		double**& U,	/// Input
		double**& V,	/// Input
		double**& F,	/// Input/Output
		double**& G)	/// Input/Output
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;

	// Check non-fluid cells
	// NOTE: we need to include ghost layers, but exclude boundary layers!!

	int jmin=0, jmax=int(lncy)+1;  // index boundary
	int imin=0, imax=int(lncx)+1;

	int nrl = (block->snei < 0) ? 1 : 0;  // looping boundary
	int nrh = (block->nnei < 0) ? int(lncy) : int(lncy)+1;
	int ncl = (block->wnei < 0) ? 1 : 0;
	int nch = (block->enei < 0) ? int(lncx) : int(lncx)+1;

	for (int j=nrl; j<=nrh; j++) {
		for (int i=ncl; i<=nch; i++) {
			if(M[j][i] == 0) { //fluid cell
				continue;
			} else if (M[j][i] == NSSim::B_N) {
				G[j][i] = V[j][i];
			} else if (M[j][i] == NSSim::B_S) {
				G[j][i] = V[j][i];
				if ((j-1) >= jmin) G[j-1][i] = V[j-1][i];
			} else if (M[j][i] == NSSim::B_E) {
				F[j][i] = U[j][i];
			} else if (M[j][i] == NSSim::B_W) {
				F[j][i] = U[j][i];
				if ((i-1) >= imin) F[j][i-1] = U[j][i-1];
			} else if (M[j][i] == NSSim::B_NE) {
				F[j][i] = U[j][i];
				G[j][i] = V[j][i];
			} else if (M[j][i] == NSSim::B_SE) {
				F[j][i] = U[j][i];
				if ((j-1) >= jmin) G[j-1][i] = V[j-1][i];
			} else if (M[j][i] == NSSim::B_NW) {
				if ((i-1) >= imin) F[j][i-1] = U[j][i-1];
				G[j][i] = V[j][i];
			} else if (M[j][i] == NSSim::B_SW) {
				if ((i-1) >= imin) F[j][i-1] = U[j][i-1];
				if ((j-1) >= jmin) G[j-1][i] = V[j-1][i];
			} else if (M[j][i] == NSSim::B_IN) {
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

void NSSimDist::update_domain_p(
		int**&    M, 	/// Input
		double**& P)	/// Input/Output
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;

	// Check non-fluid cells
	// NOTE: we need to include ghost layers, but exclude boundary layers!!

	int jmin=0, jmax=int(lncy)+1;  // index boundary
	int imin=0, imax=int(lncx)+1;

	int nrl = (block->snei < 0) ? 1 : 0;  // looping boundary
	int nrh = (block->nnei < 0) ? int(lncy) : int(lncy)+1;
	int ncl = (block->wnei < 0) ? 1 : 0;
	int nch = (block->enei < 0) ? int(lncx) : int(lncx)+1;

	for (int j=nrl; j<=nrh; j++) {
		for (int i=ncl; i<=nch; i++) {
			if (M[j][i] == 0) { //fluid cell
				continue;
			} else if (M[j][i] == NSSim::B_N) {
				if ((j+1) <= jmax) P[j][i] = P[j+1][i];
			} else if (M[j][i] == NSSim::B_S) {
				if ((j-1) >= jmin) P[j][i] = P[j-1][i];
			} else if (M[j][i] == NSSim::B_E) {
				if ((i+1) <= imax) P[j][i] = P[j][i+1];
			} else if (M[j][i] == NSSim::B_W) {
				if ((i-1) >= imin) P[j][i] = P[j][i-1];
			} else if (M[j][i] == NSSim::B_NE) {
				if (((j+1) <= jmax) && ((i+1) <= imax))
					P[j][i] = (P[j+1][i] + P[j][i+1]) / 2.0;
			} else if (M[j][i] == NSSim::B_SE) {
				if (((j-1) >= jmin) && ((i+1) <= imax))
					P[j][i] = (P[j-1][i] + P[j][i+1]) / 2.0;
			} else if (M[j][i] == NSSim::B_NW) {
				if (((j+1) <= jmax) && ((i-1) >= imin))
					P[j][i] = (P[j+1][i] + P[j][i-1]) / 2.0;
			} else if (M[j][i] == NSSim::B_SW) {
				if (((j-1) >= jmin) && ((i-1) >= imin))
					P[j][i] = (P[j-1][i] + P[j][i-1]) / 2.0;
			} else if (M[j][i] == NSSim::B_IN) {
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

void NSSimDist::compute_dt_mpi(
		double**& U,	/// Input
		double**& V,	/// Input
		double&  dt)  	/// Output
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	//       but MPI collective communication is involved,
	//       cannot exclude non-valid blocks right now.

	double umax = 0;
	double vmax = 0;
	// ONLY valid blocks find local Umax, Vmax
	if (block->id >= 0) {
		for (size_t j=0; j<=lncy+1; j++) {
			for (size_t i=0; i<=lncx+1; i++)
				if(umax < fabs(U[j][i])) umax = fabs(U[j][i]);
		}
		for (size_t j=0; j<=lncy+1; j++) {
			for (size_t i=0; i<=lncx+1; i++)
				if(vmax < fabs(V[j][i])) vmax = fabs(V[j][i]);
		}
	}
	// Find global Umax, Vmax
	if (mpi->size > 1) {
		double tmp = umax;
		MPI_Allreduce(&tmp, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi->comm);
		tmp = vmax;
		MPI_Allreduce(&tmp, &vmax, 1, MPI_DOUBLE, MPI_MAX, mpi->comm);
	}
	// Compute dt
	dt = NSSim::TAU * fmin(
			(NSSim::RE*dx*dx*dy*dy)/(2.0*(dx*dx + dy*dy)),
			fmin(dx/umax, dy/vmax));
	return;
}

void NSSimDist::compute_fg(
		double&  dt,	/// Input
		double**& U,	/// Input
		double**& V,	/// Input
		double**& F,	/// Output
		double**& G)	/// Output
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;;

	/* the traversals for F and G differ slightly, for cache efficiency */
	// 1. Compute F
	// Left boundary only
	if (block->wnei < 0) {
		for (size_t j=1; j<=lncy; j++)
			F[j][0] = U[j][0];
	}
	// Inner domain
	for (size_t j=1; j<=lncy; j++) {
		for (size_t i=1; i<=lncx; i++) {
			F[j][i] = U[j][i] + dt * (
					NSSim::laplacian(U,j,i,dx,dy)/NSSim::RE
					- NSSim::FD_x_U2(U,j,i,dx,NSSim::ALPHA)
					- NSSim::FD_y_UV(U,V,j,i,dy,NSSim::ALPHA)
					+ NSSim::EXTERNAL_FORCE_X );
		}
	}
	// Right boundary only
	if (block->enei < 0) {
		for (size_t j=1; j<=lncy; j++) {
			F[j][lncx] = U[j][lncx];
			F[j][lncx+1] = U[j][lncx+1]; // remove?
		}
	}

	// 2. Compute G
	// Bottom boundary only
	if (block->snei < 0) {
		for (size_t i=1; i<=lncx; i++)
			G[0][i] = V[0][i];
	}
	// Inner domain
	for (size_t j=1; j<=lncy; j++) {
		for (size_t i=1; i<=lncx; i++) {
			G[j][i] = V[j][i] + dt * (
					NSSim::laplacian(V,j,i,dx,dy)/NSSim::RE
					- NSSim::FD_x_UV(U,V,j,i,dx,NSSim::ALPHA)
					- NSSim::FD_y_V2(V,j,i,dy,NSSim::ALPHA)
					+ NSSim::EXTERNAL_FORCE_Y );
		}
	}
	// Top boundary only
	if (block->nnei < 0) {
		for (size_t i=1; i<=lncx; i++) {
			G[lncy][i] = V[lncy][i];
			G[lncy+1][i] = V[lncy+1][i]; // remove?
		}
	}
	return;
}

void NSSimDist::compute_uv(
		double&  dt,	/// Input
		double**& F,	/// Input
		double**& G,	/// Input
		double**& P,	/// Input
		double**& U, 	/// Output
		double**& V)	/// Output
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;

	// Check non-fluid cells
	// NOTE: we need to include ghost layers, but exclude boundary layers!!
	size_t nrh = (block->nnei < 0) ? int(lncy)-1 : int(lncy);
	size_t nch = (block->enei < 0) ? int(lncx)-1 : int(lncx);

	for (size_t j=1; j<=lncy; j++)
		for (size_t i=1; i<=nch; i++)
			U[j][i] = F[j][i] - dt * (P[j][i+1] - P[j][i]) / dx;

	for (size_t j=1; j<=nrh; j++)
		for (size_t i=1; i<=lncx; i++)
			V[j][i] = G[j][i] - dt * (P[j+1][i] - P[j][i]) / dy;

	return;
}

void NSSimDist::solve_for_p_sor_mpi(
		double& dt,
		double**& U,
		double**& V,
		double**& F,
		double**& G,
		double**& P)
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	//       but MPI collective communication is involved,
	//       cannot exclude non-valid blocks right now.

	size_t ITERMAX = 10000;
	double TOLERANCE = 0.0001;
	double inv_dx2 = 1.0 / (dx*dx);
	double inv_dy2 = 1.0 / (dy*dy);
	double inv_dt_dx = 1.0 / (dt*dx);
	double inv_dt_dy = 1.0 / (dt*dy);
	double a = 1.0 - NSSim::OMEGA;
	double b = 0.5 * NSSim::OMEGA / (inv_dx2 + inv_dy2);
	double lres, gres, tmp;

	for (size_t it=0; it < ITERMAX; it++) {
		// 1. SOR swip over local P
		if (block->id >= 0) {
			for (size_t j=1; j<=lncy; j++)
				for (size_t i=1; i<=lncx; i++)
					P[j][i] = a * P[j][i] + b * (
							inv_dx2*(P[j][i+1]+P[j][i-1])
							+ inv_dy2*(P[j+1][i]+P[j-1][i])
							- (inv_dt_dx*(F[j][i]-F[j][i-1]) +
							   inv_dt_dy*(G[j][i]-G[j-1][i]))
												);
		}
		// 2. Exchange ghost layer
		exchange_ghost_layers_mpi(P);

		// 3. Compute partial residual sum
		lres = 0.0;
		if (block->id >= 0) {
			for (size_t j=1; j<=lncy; j++)
				for (size_t i=1; i<=lncx; i++)
					tmp = inv_dx2 * (P[j][i+1]-2.0*P[j][i]+P[j][i-1]) +
						  inv_dy2 * (P[j+1][i]-2.0*P[j][i]+P[j-1][i]) -
						 (inv_dt_dx * (F[j][i]-F[j][i-1]) +
						  inv_dt_dy * (G[j][i]-G[j-1][i]));
			lres += tmp*tmp;
		}
		// 4. Allreduce global sum, compute residual norm
		if (mpi->size > 1) {
			MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, mpi->comm);
		} else {
			gres = lres;
		}
		gres = sqrt( gres/(gncy*gncx) );
		// 5. Check residual
		if (gres <= TOLERANCE) {
			if (mpi->rank == MASTER)
				cout << "SOR solver converged at iter " << it << endl;
			return;
		}
	}
	if (mpi->rank == MASTER)
		cout << "SOR solver did not converge for "
				<< ITERMAX << " iterations!" << endl;
	return;
}

void NSSimDist::exchange_ghost_layers_mpi(
		double**& m1)
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;

	// NOTE: Since we have row major storage, NORTH and SOUTH direction
	//       communication can use exact P array, no buffer needed
	unique_ptr<double[]> sbuf_e, rbuf_e;
	unique_ptr<double[]> sbuf_w, rbuf_w;
	double* sbuf_n, * rbuf_n;
	double* sbuf_s, * rbuf_s;

	// 1. Send (Non-blocking)
	unique_ptr<MPI_Request[]> sreq (new MPI_Request[4]);

	// EAST
	if (block->enei >= 0) {
		sbuf_e.reset(new double[lncy]);
		for (size_t j=1; j<=lncy; j++)
			sbuf_e[j-1] = m1[j][lncx];
		MPI_Isend(sbuf_e.get(), lncy, MPI_DOUBLE, block->enei, block->id, mpi->comm, &sreq[0]);
	} else {
		sreq[0] = MPI_REQUEST_NULL;
	}
	// WEST
	if (block->wnei >= 0) {
		sbuf_w.reset(new double[lncy]);
		for (size_t j=1; j<=lncy; j++)
			sbuf_w[j-1] = m1[j][1];
		MPI_Isend(sbuf_w.get(), lncy, MPI_DOUBLE, block->wnei, block->id, mpi->comm, &sreq[1]);
	} else {
		sreq[1] = MPI_REQUEST_NULL;
	}
	// NORTH
	if (block->nnei >= 0) {
		sbuf_n = &m1[lncy][1];
		MPI_Isend(sbuf_n, lncx, MPI_DOUBLE, block->nnei, block->id, mpi->comm, &sreq[2]);
	} else {
		sreq[2] = MPI_REQUEST_NULL;
	}
	// SOUTH
	if (block->snei >= 0) {
		sbuf_s = &m1[1][1];
		MPI_Isend(sbuf_s, lncx, MPI_DOUBLE, block->snei, block->id, mpi->comm, &sreq[3]);
	} else {
		sreq[3] = MPI_REQUEST_NULL;
	}

	// 2. Receive (Blocking)
	// WEST
	if (block->wnei >= 0) {
		rbuf_w.reset(new double[lncy]);
		MPI_Recv(rbuf_w.get(), lncy, MPI_DOUBLE, block->wnei, block->wnei, mpi->comm, MPI_STATUS_IGNORE);
		for (size_t j=1; j<=lncy; j++) {
			m1[j][0] = rbuf_w[j-1];
		}
	}
	// EAST
	if (block->enei >= 0) {
		rbuf_e.reset(new double[lncy]);
		MPI_Recv(rbuf_e.get(), lncy, MPI_DOUBLE, block->enei, block->enei, mpi->comm, MPI_STATUS_IGNORE);
		for (size_t j=1; j<=lncy; j++) {
			m1[j][lncx+1] = rbuf_e[j-1];
		}
	}
	// SOUTH
	if (block->snei >= 0) {
		rbuf_s = &m1[0][1];
		MPI_Recv(rbuf_s, lncx, MPI_DOUBLE, block->snei, block->snei, mpi->comm, MPI_STATUS_IGNORE);
	}
	// NORTH
	if (block->nnei >= 0) {
		rbuf_n = &m1[lncy+1][1];
		MPI_Recv(rbuf_n, lncx, MPI_DOUBLE, block->nnei, block->nnei, mpi->comm, MPI_STATUS_IGNORE);
	}
	MPI_Waitall(4, sreq.get(), MPI_STATUS_IGNORE);
	return;
}

void NSSimDist::exchange_ghost_layers_mpi(
		double**& m1,
		double**& m2)
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;

	unique_ptr<double[]> sbuf_e, rbuf_e;
	unique_ptr<double[]> sbuf_w, rbuf_w;
	unique_ptr<double[]> sbuf_n, rbuf_n;
	unique_ptr<double[]> sbuf_s, rbuf_s;

	// 1. Send (Non-blocking)
	unique_ptr<MPI_Request[]> sreq (new MPI_Request[4]);
	// EAST
	if (block->enei >= 0) {
		// Prepare send buffer: order U V P F G
		sbuf_e.reset(new double[lncy * 2]);
		for (size_t j=1; j<=lncy; j++)
			sbuf_e[         (j-1)] = m1[j][lncx];
		for (size_t j=1; j<=lncy; j++)
			sbuf_e[lncy*1 + (j-1)] = m2[j][lncx];
		MPI_Isend(sbuf_e.get(), lncy*2, MPI_DOUBLE, block->enei, block->id, mpi->comm, &sreq[0]);
	} else {
		sreq[0] = MPI_REQUEST_NULL;
	}
	// WEST
	if (block->wnei >= 0) {
		// Prepare send buffer: order U V P F G
		sbuf_w.reset(new double[lncy * 2]);
		for (size_t j=1; j<=lncy; j++)
			sbuf_w[         (j-1)] = m1[j][1];
		for (size_t j=1; j<=lncy; j++)
			sbuf_w[lncy*1 + (j-1)] = m2[j][1];
		MPI_Isend(sbuf_w.get(), lncy*2, MPI_DOUBLE, block->wnei, block->id, mpi->comm, &sreq[1]);
	} else {
		sreq[1] = MPI_REQUEST_NULL;
	}
	// NORTH
	if (block->nnei >= 0) {
		// Prepare send buffer: order U V P F G
		sbuf_n.reset(new double[lncx * 2]);
		for (size_t i=1; i<=lncx; i++)
			sbuf_n[         (i-1)] = m1[lncy][i];
		for (size_t i=1; i<=lncx; i++)
			sbuf_n[lncx*1 + (i-1)] = m2[lncy][i];
		MPI_Isend(sbuf_n.get(), lncx*2, MPI_DOUBLE, block->nnei, block->id, mpi->comm, &sreq[2]);
	} else {
		sreq[2] = MPI_REQUEST_NULL;
	}
	// SOUTH
	if (block->snei >= 0) {		// Prepare send buffer: order U V P F G
		sbuf_s.reset(new double[lncx * 2]);
		for (size_t i=1; i<=lncx; i++)
			sbuf_s[         (i-1)] = m1[1][i];
		for (size_t i=1; i<=lncx; i++)
			sbuf_s[lncx*1 + (i-1)] = m2[1][i];
		MPI_Isend(sbuf_s.get(), lncx*2, MPI_DOUBLE, block->snei, block->id, mpi->comm, &sreq[3]);
	} else {
		sreq[3] = MPI_REQUEST_NULL;
	}
	// 2. Receive (Blocking)
	// WEST
	if (block->wnei >= 0) {
		rbuf_w.reset(new double[lncy*2]);
		MPI_Recv(rbuf_w.get(), lncy*2, MPI_DOUBLE, block->wnei, block->wnei, mpi->comm, MPI_STATUS_IGNORE);
		for (size_t j=1; j<=lncy; j++)
			m1[j][0] = rbuf_w[         (j-1)];
		for (size_t j=1; j<=lncy; j++)
			m2[j][0] = rbuf_w[lncy*1 + (j-1)];
	}
	// EAST
	if (block->enei >= 0) {
		rbuf_e.reset(new double[lncy*2]);
		MPI_Recv(rbuf_e.get(), lncy*2, MPI_DOUBLE, block->enei, block->enei, mpi->comm, MPI_STATUS_IGNORE);
		for (size_t j=1; j<=lncy; j++)
			m1[j][lncx+1] = rbuf_e[         (j-1)];
		for (size_t j=1; j<=lncy; j++)
			m2[j][lncx+1] = rbuf_e[lncy*1 + (j-1)];
	}
	// SOUTH
	if (block->snei >= 0) {
		rbuf_s.reset(new double[lncx*2]);
		MPI_Recv(rbuf_s.get(), lncx*2, MPI_DOUBLE, block->snei, block->snei, mpi->comm, MPI_STATUS_IGNORE);
		for (size_t i=1; i<=lncx; i++)
			m1[0][i] = rbuf_s[         (i-1)];
		for (size_t i=1; i<=lncx; i++)
			m2[0][i] = rbuf_s[lncx*1 + (i-1)];
	}
	// NORTH
	if (block->nnei >= 0) {
		rbuf_n.reset(new double[lncx*2]);
		MPI_Recv(rbuf_n.get(), lncx*2, MPI_DOUBLE, block->nnei, block->nnei, mpi->comm, MPI_STATUS_IGNORE);
		for (size_t i=1; i<=lncx; i++)
			m1[lncy+1][i] = rbuf_n[         (i-1)];
		for (size_t i=1; i<=lncx; i++)
			m2[lncy+1][i] = rbuf_n[lncx*1 + (i-1)];
	}
	MPI_Waitall(4, sreq.get(), MPI_STATUS_IGNORE);
	return;
}

void NSSimDist::write_vtk_dist(
		const char *outfile_prefix,
		int timeStepNumber,
		int**&    M,
		double**& U,
		double**& V,
		double**& P)
{
	/************************************************************************
	 * VTK XML File Formats
	 * Type: StructuredGrid - Each StructuredGrid piece specifies its extent
	 *       within the dataset's whole extent. The points are described
	 *       explicitly by the Points element. The cells are described
	 *       implicitly by the extent.
	 ************************************************************************
	 * <VTKFile type="StructuredGrid" ...>
	 * <StructuredGrid WholeExtent="x1 x2 y1 y2 z1 z2">
	 * <Piece Extent="x1 x2 y1 y2 z1 z2">
	 * <PointData>...</PointData>
	 * <CellData>...</CellData>
	 * <Points>...</Points>
	 * </Piece>
	 * </StructuredGrid>
	 * </VTKFile>
	 ************************************************************************/

	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;

	// Create and open file
	char outfile[100];
	sprintf(outfile, "%s.block%i.%i.vts", outfile_prefix, block->id, timeStepNumber);
	std::ofstream fout;
	fout.open(outfile, std::ofstream::out);
	if (!fout.is_open()) {
		std::cout << "Fail to open output VTK file. Operation aborted!" << std::endl;
		exit(EXIT_FAILURE);
	}

	// Write VTK Header.
	fout << "<?xml version=\"1.0\"?>" << endl;
	fout << "<VTKFile type=\"StructuredGrid\">" << endl;
	fout << "<StructuredGrid WholeExtent=\""
			<< block->xmin-1 << " " << block->xmax << " "
			<< block->ymin-1 << " " << block->ymax << " 0 0\">" << endl;
	fout << "<Piece Extent=\""
			<< block->xmin-1 << " " << block->xmax << " "
			<< block->ymin-1 << " " << block->ymax << " 0 0\">" << endl;

	// Write grid points
	fout << "<Points>" << endl;
	fout << "<DataArray NumberOfComponents=\"3\" type=\"Float32\" format=\"ascii\">" << endl;
	double x_offset = double(block->xmin-1) * dx;
	double y_offset = double(block->ymin-1) * dy;
	for(size_t j=0; j<=lncy; j++)
		for(size_t i=0; i<=lncx; i++)
			fout << x_offset+(i*dx) << " "
			     << y_offset+(j*dy) << " " << 0 << endl;
	fout << "</DataArray>" << endl;
	fout << "</Points>" << endl;

	// Write velocity (Point data)
	fout << "<PointData>" << endl;
	fout << "<DataArray Name=\"Velocity\" NumberOfComponents=\"3\" type=\"Float32\" format=\"ascii\">" << endl;
	for(size_t j=0; j<=lncy; j++) {
		for(size_t i=0; i<=lncx; i++) {
			if (NSSim::is_point_in_obs_cell(M,j,i))
				fout << 0.0 << " " << 0.0 << " " << 0.0 << std::endl;
			else
				fout << (U[j][i] + U[j+1][i]) * 0.5 << " "
				     << (V[j][i] + V[j][i+1]) * 0.5 << " "
					 << 0.0 << std::endl;
		}
	}
	fout << "</DataArray>" << endl;
	fout << "</PointData>" << endl;

	// Write pressure (Cell data)
	fout << "<CellData>" << endl;
	fout << "<DataArray Name=\"Pressure\" NumberOfComponents=\"1\" type=\"Float32\" format=\"ascii\">" << endl;
	for(size_t j=1; j<=lncy; j++) {
		for(size_t i=1; i<=lncx; i++) {
			if (M[j][i] == 0)
				fout << P[j][i] << std::endl;
			else
				fout << 0.0 << std::endl;
		}
	}
	fout << "</DataArray>" << endl;
	fout << "</CellData>" << endl;

	// closing
	fout << "</Piece>" << endl;
	fout << "</StructuredGrid>" << endl;
	fout << "</VTKFile>" << endl;
	fout.close();
	return;
}

/*********************************
 *******    iMPI Stuff     *******
 *********************************/

double** NSSimDist::gather_global_matrix_mpi(double**& m)
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return nullptr;

	double** global_m;
	// MASTER collects the global matrix
	if (mpi->rank == MASTER) {
		// Allocate space for output array
		global_m = tools::alloc_matrix<double>(gncy+2, gncx+2, true);

		vector<DomainBlock*> blocks =
				DomainDecomposer2D::gen_blocks(mpi->size, 1, gncx, 1, gncy);
		DomainBlock* b;
		int pncx, pncy, pn;
		int j, i, jmin, jmax, imin, imax;

		// It's possible that NOT ALL processes have a valid block
		for (int p=0; p < blocks.size(); p++) {
			if (p == mpi->rank) {
				jmin = (block->snei < 0) ? 0 : 1;
				jmax = (block->nnei < 0) ? lncy+1 : lncy;
				imin = (block->wnei < 0) ? 0 : 1;
				imax = (block->enei < 0) ? lncx+1 : lncx;
				for (j=jmin; j <= jmax; j++)
					for (i=imin; i <= imax; i++)
						global_m[j + (block->ymin-1)][i + (block->xmin-1)]
								= m[j][i];
			} else {
				// Receive
				b = blocks[p];
				pncx = b->xmax - b->xmin + 1;
				pncy = b->ymax - b->ymin + 1;
				pn = (pncy+2) * (pncx+2);

				unique_ptr<double[]> rbuf (new double[pn]);
				MPI_Recv(rbuf.get(), pn, MPI_DOUBLE, p, p,
						mpi->comm, MPI_STATUS_IGNORE);

				jmin = (blocks[p]->snei < 0) ? 0 : 1;
				jmax = (blocks[p]->nnei < 0) ? pncy+1 : pncy;
				imin = (blocks[p]->wnei < 0) ? 0 : 1;
				imax = (blocks[p]->enei < 0) ? pncx+1 : pncx;
				for (j=jmin; j <= jmax; j++)
					for (i=imin; i <= imax; i++)
						global_m[ yl2g(j,b) ][ xl2g(i,b) ]
								= rbuf[j * (pncx+2) + i];
			} // end if-else
		} // end for(p)
		for (int p=0; p < blocks.size(); p++)
			delete blocks[p];
		blocks.clear();
	} else {
		// Others send their part only when they have a valid DomainBlock
		if (block->id >= 0)
			MPI_Send(m[0], (lncy+2)*(lncx+2), MPI_DOUBLE, MASTER,
					mpi->rank, mpi->comm);
	}
	return global_m;
}

int** NSSimDist::gather_global_matrix_mpi(int**& m)
{
	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return nullptr;

	int** global_m = nullptr;
	int count, pncx, pncy;
	int j, i, jmin, jmax, imin, imax;
	// ROOT collects the global matrix
	if (mpi->rank == MASTER) {
		// Allocate space for output array
		global_m = tools::alloc_matrix<int>(gncy+2, gncx+2, true);

		// Good to know about everyone else :)
		vector<DomainBlock*> blocks =
				DomainDecomposer2D::gen_blocks(mpi->size, 1, gncx, 1, gncy);
		DomainBlock* b;
		int pncx, pncy, pn;
		int j, i, jmin, jmax, imin, imax;

		// It's possible that NOT ALL processes have a valid block
		for (int p=0; p < blocks.size(); p++) {
			if (p == mpi->rank) {
				jmin = (block->snei < 0) ? 0 : 1;
				jmax = (block->nnei < 0) ? lncy+1 : lncy;
				imin = (block->wnei < 0) ? 0 : 1;
				imax = (block->enei < 0) ? lncx+1 : lncx;
				for (j=jmin; j <= jmax; j++)
					for (i=imin; i <= imax; i++)
						global_m[j + (block->ymin-1)][i + (block->xmin-1)]
								= m[j][i];
			} else {
				// Receive
				b = blocks[p];
				pncx = b->xmax - b->xmin + 1;
				pncy = b->ymax - b->ymin + 1;
				pn = (pncy+2) * (pncx+2);

				unique_ptr<int[]> rbuf (new int[count]);
				MPI_Recv(rbuf.get(), pn, MPI_INT, p, p,
						mpi->comm, MPI_STATUS_IGNORE);

				jmin = (blocks[p]->snei < 0) ? 0 : 1;
				jmax = (blocks[p]->nnei < 0) ? pncy+1 : pncy;
				imin = (blocks[p]->wnei < 0) ? 0 : 1;
				imax = (blocks[p]->enei < 0) ? pncx+1 : pncx;
				for (j=jmin; j <= jmax; j++)
					for (i=imin; i <= imax; i++)
						global_m[ yl2g(j,b) ][ xl2g(i,b) ]
								= rbuf[j * (pncx+2) + i];
			} // end if-else
		} // end for(p)
		for (int p=0; p < blocks.size(); p++)
			delete blocks[p];
		blocks.clear();
	} else {
		// Others send their part only when they have a valid DomainBlock
		if (block->id >= 0)
			MPI_Send(m[0], (lncy+2)*(lncx+2), MPI_INT, MASTER,
					mpi->rank, mpi->comm);
	}
	return global_m;
}

void NSSimDist::update_redistribute_domain_mpi(
		MPI_Comm newcomm,	/// Input
		int**& M,	  /// Output
		double**& U,  /// Input/Output
		double**& V,  /// Input/Output
		double**& P,  /// Input/Output
		double**& F,  /// Input/Output
		double**& G)  /// Input/Output
{
	double** gU = nullptr;
	double** gV = nullptr;
	double** gP = nullptr;
	double** gF = nullptr;
	double** gG = nullptr;

	// 1. MASTER gather global matrices
#if (ENABLE_IMPI == YES)
	if (mpi->status != MPI_ADAPT_STATUS_JOINING) {
#endif
	gU = gather_global_matrix_mpi(U);
	gV = gather_global_matrix_mpi(V);
	gP = gather_global_matrix_mpi(P);
	gF = gather_global_matrix_mpi(F);
	gG = gather_global_matrix_mpi(G);
#if (ENABLE_IMPI == YES)
	}
#endif

	// 2. Update MPI object
	mpi->update(newcomm);

	// 3. Update domain block
	block.reset( DomainDecomposer2D::gen_block(
			mpi->size, 1, gncx, 1, gncy, mpi->rank) );
	lncx = (block->id < 0) ? 0 : size_t(block->xmax - block->xmin) + 1;
	lncy = (block->id < 0) ? 0 : size_t(block->ymax - block->ymin) + 1;

	// NOTE: It's possible NOT ALL processes participate in computation.
	if (block->id < 0) return;

	// 4. Resize local matrices (free then reallocate)
	tools::free_matrix<int>(M);
	tools::free_matrix<double>(U);
	tools::free_matrix<double>(V);
	tools::free_matrix<double>(P);
	tools::free_matrix<double>(F);
	tools::free_matrix<double>(G);
	M = create_geometry_mask();
	U = tools::alloc_matrix<double>(lncy+2, lncx+2, true);
	V = tools::alloc_matrix<double>(lncy+2, lncx+2, true);
	P = tools::alloc_matrix<double>(lncy+2, lncx+2, true);
	F = tools::alloc_matrix<double>(lncy+2, lncx+2, true);
	G = tools::alloc_matrix<double>(lncy+2, lncx+2, true);

	// 4. For the five computational arrays, MASTER send, others receive
	if (mpi->rank == MASTER) {
		vector<DomainBlock*> blocks =
				DomainDecomposer2D::gen_blocks(mpi->size, 1, gncx, 1, gncy);
		unique_ptr<MPI_Request[]> sreqs (new MPI_Request[blocks.size()]);
		vector<unique_ptr<double[]> > sbufs (blocks.size());
		DomainBlock* b;
		size_t pncx, pncy, pn, offset;

		// It's possible that NOT ALL processes have a valid block
		for (int p=0; p < blocks.size(); p++) {
			if (p == mpi->rank) {
				sbufs[p].reset(nullptr);
				sreqs[p] = MPI_REQUEST_NULL;
				b = block.get();
				// U
				for (size_t j=0; j <= lncy+1; j++)
					for (size_t i=0; i <= lncx+1; i++)
						U[j][i] = gU[ yl2g(int(j),b) ][ xl2g(int(i),b) ];
				// V
				for (size_t j=0; j <= lncy+1; j++)
					for (size_t i=0; i <= lncx+1; i++)
						V[j][i] = gV[ yl2g(int(j),b) ][ xl2g(int(i),b) ];
				// P
				for (size_t j=0; j <= lncy+1; j++)
					for (size_t i=0; i <= lncx+1; i++)
						P[j][i] = gP[ yl2g(int(j),b) ][ xl2g(int(i),b) ];
				// F
				for (size_t j=0; j <= lncy+1; j++)
					for (size_t i=0; i <= lncx+1; i++)
						F[j][i] = gF[ yl2g(int(j),b) ][ xl2g(int(i),b) ];
				// G
				for (size_t j=0; j <= lncy+1; j++)
					for (size_t i=0; i <= lncx+1; i++)
						G[j][i] = gG[ yl2g(int(j),b) ][ xl2g(int(i),b) ];
			} else {
				// Prepare send buffer
				b = blocks[p];
				pncx = size_t(b->xmax - b->xmin + 1);
				pncy = size_t(b->ymax - b->ymin + 1);
				pn = (pncy+2) * (pncx+2);
				sbufs[p].reset(new double[pn * 5]);
				// U
				offset = 0;
				for (size_t j=0; j <= pncy+1; j++)
					for (size_t i=0; i <= pncx+1; i++)
						(sbufs[p])[offset + j*(pncx+2)+i] = gU[ yl2g(int(j),b) ][ xl2g(int(i),b) ];
				// V
				offset = pn * 1;
				for (size_t j=0; j <= pncy+1; j++)
					for (size_t i=0; i <= pncx+1; i++)
						(sbufs[p])[offset + j*(pncx+2)+i] = gV[ yl2g(int(j),b) ][ xl2g(int(i),b) ];
				// P
				offset = pn * 2;
				for (size_t j=0; j <= pncy+1; j++)
					for (size_t i=0; i <= pncx+1; i++)
						(sbufs[p])[offset + j*(pncx+2)+i] = gP[ yl2g(int(j),b) ][ xl2g(int(i),b) ];
				// F
				offset = pn * 3;
				for (size_t j=0; j <= pncy+1; j++)
					for (size_t i=0; i <= pncx+1; i++)
						(sbufs[p])[offset + j*(pncx+2)+i] = gF[ yl2g(int(j),b) ][ xl2g(int(i),b) ];
				// G
				offset = pn * 4;
				for (size_t j=0; j <= pncy+1; j++)
					for (size_t i=0; i <= pncx+1; i++)
						(sbufs[p])[offset + j*(pncx+2)+i] = gG[ yl2g(int(j),b) ][ xl2g(int(i),b) ];
				// Send
				MPI_Isend(sbufs[p].get(), pn*5, MPI_DOUBLE, p, MASTER, mpi->comm, &sreqs[p]);
			} // end if-else
		} // end for(p)

		// CLEAN UP
		MPI_Waitall(blocks.size(), sreqs.get(), MPI_STATUS_IGNORE);
		for (int p=0; p < blocks.size(); p++)
			delete blocks[p];
		blocks.clear();
	} else {
		// Prepare recv buffer
		unique_ptr<double[]> rbuf (new double[(lncy+2)*(lncx+2)*5]);
		MPI_Recv(rbuf.get(), (lncy+2)*(lncx+2)*5, MPI_DOUBLE, MASTER, MASTER,
				mpi->comm, MPI_STATUS_IGNORE);
		// Unpack...
		// U
		size_t offset = 0;
		for (size_t j=0; j <= lncy+1; j++)
			for (size_t i=0; i <= lncx+1; i++)
				U[j][i] = rbuf[offset + j*(lncx+2)+i];
		// V
		offset = (lncy+2)*(lncx+2) * 1;
		for (size_t j=0; j <= lncy+1; j++)
			for (size_t i=0; i <= lncx+1; i++)
				V[j][i] = rbuf[offset + j*(lncx+2)+i];
		// P
		offset = (lncy+2)*(lncx+2) * 2;
		for (size_t j=0; j <= lncy+1; j++)
			for (size_t i=0; i <= lncx+1; i++)
				P[j][i] = rbuf[offset + j*(lncx+2)+i];
		// F
		offset = (lncy+2)*(lncx+2) * 3;
		for (size_t j=0; j <= lncy+1; j++)
			for (size_t i=0; i <= lncx+1; i++)
				F[j][i] = rbuf[offset + j*(lncx+2)+i];
		// G
		offset = (lncy+2)*(lncx+2) * 4;
		for (size_t j=0; j <= lncy+1; j++)
			for (size_t i=0; i <= lncx+1; i++)
				G[j][i] = rbuf[offset + j*(lncx+2)+i];
	}
	tools::free_matrix<double>(gU);
	tools::free_matrix<double>(gV);
	tools::free_matrix<double>(gP);
	tools::free_matrix<double>(gF);
	tools::free_matrix<double>(gG);
	return;
}



