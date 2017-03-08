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

#ifndef CONFIG_H_
#define CONFIG_H_

#include <mpi.h>


/******************************
 * General setting
 ******************************/
//#define YES		10
//#define NO		-10

// System related
#define FULLMODEL	1	// 1: Navier-Stokes simulation, 2: Heat simulation, 3: Acoustic simulation
#define OUTPUT_PATH	"./out"	// IMPORTANT: output path for all intermediate computation files and vtk outputs

// MPI related
//#define MASTER			0   // Fix the rank of master process (master rank should not change during invasion)
//#define ENABLE_IMPI		NO	// IMPI switch
//#define IMPI_ADAPT_INTERVAL 30// // How often (in seconds) to probe the iRM (resource manager)

/******************************
 * Full Model: Navier-Stokes
 ******************************/
//#define NS_USE_DIRECT_SOLVER 	YES		// To use direct solver (from Eigen) or SOR solver
//#define NS_VTK_OUTPUT			YES 	// To enable VTK output or not
//#define NS_VTK_INTERVAL			0.05	// Write a VTK output every NS_VTK_INTERVAL seconds
//#define NS_NUM_OBS				4		// Simulation with 1 or 2 or 3 or 4 obstacles in the flow channel
#define NS_SGI_INIT_LEVEL		6		// (obs4|dim8) lv4 gps 1121, lv5 gps 6401, lv6 gps 31745
//#define NS_NCX					400		// small run: 100, big run: 200
//#define NS_NCY					80		// small run: 20,  big run: 40

/****************************************************
 * Surrogate Model: Sparse Grid Interpolation (SGI)
 ****************************************************/
#define SGI_OUT_TIMER		YES		// For performance analysis
#define SGI_OUT_RANK_PROGRESS	YES
#define SGI_OUT_GRID_POINTS	NO		// For debug only
#define SGI_MPIMW_TRUNK_SIZE	3		// Trunk size for master-minion scheme

/******************************
 * MCMC Solver Setting
 ******************************/
// in MCMCSolver
#define MCMC_NOISE_IN_DATA 		0.2		// Estimated noise in observed data
#define MCMC_RANDOM_WALK_SIZE 	0.05	// Random walk size = n% of domain size

// TODO: remove this
// iMPI stuff
//#define MPI_ADAPT_FALSE				0
//#define MPI_ADAPT_TRUE				1
//#define MPI_ADAPT_STATUS_NEW		2
//#define MPI_ADAPT_STATUS_JOINING	3
//#define MPI_ADAPT_STATUS_STAYING	4
//#define MPI_ADAPT_STATUS_RETREATING	5
//typedef int MPI_Info;
//int MPI_Init_adapt(int *argc, char ***argv, int *local_status);
//int MPI_Probe_adapt(int *current_operation, int *local_status, MPI_Info *info);
//int MPI_Comm_adapt_begin(MPI_Comm *intercomm, MPI_Comm *new_comm_world, int, int);
//int MPI_Comm_adapt_commit(int *phase_index);

#endif /* CONFIG_H_ */
