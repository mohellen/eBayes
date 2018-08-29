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

#include <surrogate/SGI.hpp>

using namespace std;
using namespace	sgpp::base;


SGI::SGI(
		Config const& c,
		Parallel & p,
		ForwardModel & m)
		: ForwardModel(c), par(p), fullmodel(m)
{
	alphas.resize(cfg.get_output_size());
	grid = nullptr;
	eval = nullptr;
	bbox = nullptr;
	maxpos_list.clear();
#if (IMPI==1)
	impi_gpoffset = 0;
#endif
}

vector<double> SGI::run(
		vector<double> const& m)
{
	// Grid check
	if (!eval) {
		cout << tools::red << "ERROR: Rank " << par.rank << " SGI model is not properly built. Progam abort!" << tools::reset << endl;
		exit(EXIT_FAILURE);
	}
	// Convert m into DataVector
	DataVector point (m);
	// Evaluate m
	std::size_t output_size = cfg.get_output_size();
	vector<double> d (output_size);
	for (std::size_t j=0; j < output_size; j++) {
		d[j] = eval->eval(alphas[j], point);
	}
	return d;
}

void SGI::build()
{
	// Get config variables
	std::size_t input_size = cfg.get_input_size();
	std::size_t output_size = cfg.get_output_size();
	std::size_t init_level = cfg.get_param_sizet("sgi_init_level");
	double refine_portion = cfg.get_param_double("sgi_refine_portion");
	bool is_masterworker = cfg.get_param_bool("sgi_is_masterworker");
	bool is_fromfiles = cfg.get_param_bool("sgi_is_from_files");
	// find out whether it's grid initialization or refinement
	bool is_init = (!this->eval) ? true : false;
	std::size_t num_points;

#if (IMPI==1)
	if (par.status != MPI_ADAPT_STATUS_JOINING) {
#endif
		if (is_init) {
			if (par.is_master())
				cout << "\nSGI: Initializing SGI model..." << endl;
			// 1. All: Construct grid
			grid.reset(Grid::createModLinearGrid(input_size).release()); // create empty grid
			grid->getGenerator().regular(init_level); // populate grid points
			bbox.reset(create_boundingbox());
			grid->setBoundingBox(*bbox); // set up bounding box
			impi_gpoffset = 0; // NEW ranks set this variable needed by the JOINING ranks
			num_points = grid->getSize();
		} else {
			if (par.is_master())
				cout << "\nSGI: Refining SGI model..." << endl;
			// 1. All: refine grid
			impi_gpoffset = grid->getSize(); // STAYING ranks set this variable needed by the JOINING ranks
			refine_grid(refine_portion); // MASTER refine and write grid, others read grid
			num_points = grid->getSize();
		}
#if (IMPI==1)
	} else {
		impi_adapt();
	}
#endif
	// 2. All: Compute data at each grid point (result written to MPI IO file)
	//		and find the top maxpos points
	compute_grid_points(impi_gpoffset, is_masterworker);
	// 3. All: Compute and hierarchize alphas
	compute_hier_alphas();
	// 4. Update op_eval
	eval.reset(sgpp::op_factory::createOperationEval(*grid).release());

//	cout << tools::yellow << "Rank " << par.rank << ": has " << maxpos_list.size() << " top maxpos points...\n";
//	for (auto it = maxpos_list.begin(); it != maxpos_list.end(); ++it)
//		cout << "\n\t" << it->first << " --- " << tools::sample_to_string(get_gp_coord(it->second));
//	cout << tools::reset << endl;

	// Master: print grogress
	if (par.is_master()) {
		if (is_init) {
			cout << "SGI: Initialize SGI model successful.\n" << endl;
		} else {
			cout << "SGI: Refine SGI model successful.\n" << endl;
		}
	}
	return;
}

vector<double> SGI::get_nth_maxpos(std::size_t n)
{
	vector< pair<double, std::size_t> > list (maxpos_list.size());
	std::copy(maxpos_list.begin(), maxpos_list.end(), &list[0]);
	// maxpos_list stores the MAX in the last entry (ascending order by posterior)
	// getting the n-th maximum should be counting from the list end
	vector<double> samplepos = get_gp_coord( list[list.size()-1-n].second );
	samplepos.push_back( list[list.size()-1-n].first );
	return samplepos;
}

//void SGI::duplicate(
//		string const& gridfile,
//		string const& datafile,
//		string const& posfile)
//{
//	// Set: grid, eval, bbox
//	mpiio_read_grid(gridfile);
//	// Set: alphas
//	compute_hier_alphas(datafile);
//	// Read posterior
//	std::size_t num_gps = grid->getSize();
//	unique_ptr<double[]> pos (new double[num_gps]);
//	mpiio_readwrite_posterior(true, 0, num_gps-1, pos.get(), posfile);
//
//	// Populate top maxpos list
//	std::size_t num_maxpos = ( cfg.get_param_sizet("mcmc_max_chains") < num_gps ) ?
//			cfg.get_param_sizet("mcmc_max_chains") : num_gps;
//	// 1. Fill the first maxpos (pos + gp_seq)
//	for (size_t i=0; i < num_maxpos; ++i) {
//		maxpos_list.emplace(pos[i], i);
//	}
//	// 2. Go through the rest gps, and add the top maxpos ones
//	for (size_t i=num_maxpos; i < num_gps; ++i) {
//		if (pos[i] > (*maxpos_list.begin()).first) { // compare current pos with the MIN pos in list
//			// insert current pos
//			maxpos_list.emplace(pos[i], i);
//			// remove the MIN pos
//			maxpos_list.erase(maxpos_list.begin());
//		}
//	}
//	return;
//}


/*********************************************
 *       		 Private Methods
 *********************************************/
void SGI::impi_adapt()
{
#if (IMPI==1)
	int adapt_flag;
	MPI_Info info;
	MPI_Comm intercomm;
	MPI_Comm newcomm;
	int staying_count, leaving_count, joining_count;
	double tic, toc;
	double tic1, toc1;

	tic = MPI_Wtime();
	//----
	MPI_Probe_adapt(&adapt_flag, &par.status, &info);
	//----
	toc = MPI_Wtime() - tic;
	if (par.is_master())
		cout << "SGI.IMPI: MASTER call MPI_Probe_adapt() in " << toc << " seconds." << endl;

	if (adapt_flag == MPI_ADAPT_TRUE){
		tic1 = MPI_Wtime();

		tic = MPI_Wtime();
		//----
		MPI_Comm_adapt_begin(&intercomm, &newcomm, &staying_count, &leaving_count, &joining_count);
		//----
		toc = MPI_Wtime() - tic;
		if (par.is_master())
			cout << "SGI.IMPI: MASTER call MPI_Comm_adapt_begin() in " << toc << " seconds." << endl;

		//************************ ADAPT WINDOW ****************************
		if (par.status == MPI_ADAPT_STATUS_JOINING)
			mpiio_read_grid();
			
		//TODO: DEBUG only
		bool is_grid_equal = verify_grid_from_read(0, intercomm);
		if (par.is_master()) {
			if (is_grid_equal) {
				cout << "Joining rank 0 read grid correctly!" << endl;
			} else {
				cout << "Joining rank 0 read grid wrong! Abort!!" << endl;
				exit(EXIT_FAILURE);
			}
		}

		if (joining_count > 0)
			MPI_Bcast(&impi_gpoffset, 1, MPI_SIZE_T, par.master, newcomm);
		//************************ ADAPT WINDOW ****************************

		tic = MPI_Wtime();
		//----
		MPI_Comm_adapt_commit();
		//----
		toc = MPI_Wtime() - tic;
		if (par.is_master())
			cout << "SGI.IMPI: MASTER call MPI_Comm_adapt_commit() in " << toc << " seconds." << endl;

		par.mpi_update();

		toc1 = MPI_Wtime() - tic1;
		if (par.is_master())
			cout << "SGI.IMPI: MASTER call impi_adapt() in " << toc1 << " seconds." << endl;
	}
	return;
#endif
}

vector<double> SGI::get_gp_coord(std::size_t seq)
{
	std::size_t input_size = cfg.get_input_size();
	// Get grid point coordinate
	DataVector coord (input_size);
	grid->getStorage().get(seq)->getCoordsBB(coord, grid->getBoundingBox());
	// Output vector
	vector<double> gp (input_size);
	for (size_t i=0; i < input_size; ++i)
		gp[i] = coord[i];
	return gp;
}

BoundingBox* SGI::create_boundingbox()
{
	std::size_t input_size = cfg.get_input_size();
	BoundingBox* bb = new BoundingBox(input_size);
	DimensionBoundary db;
	for(int i=0; i < input_size; i++) {
		db.leftBoundary = fullmodel.get_input_space(i).first;
		db.rightBoundary = fullmodel.get_input_space(i).second;
		db.bDirichletLeft  = false;
		db.bDirichletRight = false;
		bb->setBoundary(i, db);
	}
	return bb;
}

void SGI::compute_hier_alphas()
{
#if (SGI_PRINT_TIMER==1)
	double tic = MPI_Wtime();
#endif
	std::size_t output_size = cfg.get_output_size();
	std::size_t num_gps = grid->getSize();
	// read raw data
	unique_ptr<double[]> data (new double[output_size * num_gps]);
	mpiio_readwrite_data(true, 0, num_gps-1, data.get());
	// re-allocate alphas
	for (std::size_t j=0; j < output_size; j++)
		alphas[j].resize(num_gps);
	// unpack raw data
	for (std::size_t i=0; i < num_gps; i++)
		for (std::size_t j=0; j < output_size; j++)
			alphas[j].set(i, data[i*output_size+j]);
	// hierarchize alphas
	auto hier = sgpp::op_factory::createOperationHierarchisation(*grid);
	for (std::size_t j=0; j<output_size; j++)
		hier->doHierarchisation(alphas[j]);
#if (SGI_PRINT_TIMER==1)
	if (par.is_master())
		cout << "SGI: MASTER created alphas in " << MPI_Wtime()-tic << " seconds." << endl;
#endif
	return;
}

void SGI::compute_grid_points(
		std::size_t gp_offset,
		bool is_masterworker)
{
#if (SGI_PRINT_TIMER==1)
	double tic = MPI_Wtime(); // start the timer
#endif
	// NOTE: "Master-minion" scheme can run under MPI & iMPI
	// 		 "Naive" (aka SIMD) scheme can run under only MPI
	if (is_masterworker) {
		// Master-worker style
		if (par.is_master()) {
			mpimw_master_compute(gp_offset);
		} else {
			mpimw_worker_compute(gp_offset);
		}
		// Sync maxpos_list (master bcast to all)
		mpimw_sync_maxpos();
	} else {
		// MPI native style (default)
		std::size_t num_gps = grid->getSize();
		std::size_t mymin, mymax;
		mpina_get_local_range(gp_offset, num_gps-1, mymin, mymax);
		compute_gp_range(mymin, mymax);
		// Find top maspos
		mpina_find_global_maxpos();
	}
	MPI_Barrier(MPI_COMM_WORLD);

#if (SGI_PRINT_TIMER==1)
	if (par.is_master()) {
		cout << "SGI: Computed " << grid->getSize()-gp_offset
		   	<< " grid points [" << gp_offset << ", " << grid->getSize()-1
			<<"] in " << MPI_Wtime()-tic << " seconds." << endl;
	}
#endif
	return;
}

void SGI::compute_gp_range(
		const std::size_t& seq_min,
		const std::size_t& seq_max)
{
	// IMPORTANT: Ensure workload is at least 1 grid point!
	// This also prevents overriding wrong data to data files!
	if (seq_max < seq_min) return;

	std::size_t output_size = cfg.get_output_size();
	std::size_t load = std::size_t(fmax(0, seq_max - seq_min + 1));
	std::size_t num_maxpos = cfg.get_param_sizet("mcmc_max_chains");

	unique_ptr<double[]> data (new double[load * output_size]);
	unique_ptr<double[]> pos (new double[load]);
	vector<double> dvec;
	double* d = nullptr;
	double* p = nullptr;

	for (std::size_t i=seq_min; i <= seq_max; ++i) {
		// Set output pointer
		d = &data[0] + (i-seq_min) * output_size;
		p = &pos[0] + (i-seq_min);
		// compute with full model
		dvec = fullmodel.run( get_gp_coord(i) );
		std::copy(dvec.begin(), dvec.end(), d);
		// compute posterior
		*p = cfg.compute_posterior(dvec);
		// insert into maxpos_list
		if ( maxpos_list.size() < num_maxpos) { // Case list not full yet
			maxpos_list.emplace(*p, i);
		} else if (*p > maxpos_list.begin()->first) { // Case list is full, erase the min one after insert
			maxpos_list.emplace(*p, i);
			maxpos_list.erase(maxpos_list.begin());
		}
#if (SGI_PRINT_GRIDPOINTS==1)
		cout << tools::blue << "SGI: Rank " << par.rank << "/" << par.size << " completed grid point " << i << " at "
			<< tools::sample_to_string(get_gp_coord(i)) << ", pos = "
			<< *p << tools::reset << endl;
#endif
	}
	// Write results to file
	mpiio_readwrite_data(false, seq_min, seq_max, data.get());
	mpiio_readwrite_posterior(false, seq_min, seq_max, pos.get());
	return;
}

/**
 * TODO: BUG!!!
 * For some strange reason, each rank refine its own grid creates different grids
 * even thought num_gps and refine_gps are the same before refinement.
 * Therefore, we have to use a single-refine strategy: Master refine, others read
 */
#if (1==0)
bool SGI::refine_grid_old(double portion_to_refine)
{
	//TODO: DEBUG only
	cout << "[Rank " << par.rank << "/" << par.size << "]: grid BEFORE refine " << grid->getSize() << endl;

#if (SGI_PRINT_TIMER==1)
	double tic = MPI_Wtime();
#endif
	// Compute threshold number of grid points to be added
	// NOTE: to refine X points, maximum (2*dim*X) points can be added to grid
	int maxi = 10000;
	int thres = int(ceil(maxi / 2 / cfg.get_input_size()));
	// Number of points to refine
	std::size_t num_gps = this->grid->getSize();
	//TODO: DEBUG only
	cout << "[Rank " << par.rank << "/" << par.size << "]: num_gps = " << num_gps << endl;

	int refine_gps = int(ceil(num_gps * portion_to_refine));
	refine_gps = (refine_gps > thres) ? thres : refine_gps;
	//TODO: DEBUG only
	cout << "[Rank " << par.rank << "/" << par.size << "]: refine_gps = " << refine_gps << endl;

	// If not points to refine, return false (meaning grid not refined)
	if (refine_gps < 1) return false;
	// Read posterior from file
	unique_ptr<double[]> pos (new double[num_gps]);
	mpiio_readwrite_posterior(true, 0, num_gps-1, &pos[0]);
	// For each gp, compute the refinement index
	double data_norm;
	DataVector refine_idx (num_gps);
	std::size_t output_size = cfg.get_output_size();
	for (std::size_t i=0; i<num_gps; i++) {
		data_norm = 0;
		for (std::size_t j=0; j < output_size; j++) {
			data_norm += (alphas[j][i] * alphas[j][i]);
		}
		data_norm = sqrt(data_norm);
		// refinement_index = |alpha| * posterior
		refine_idx[i] = data_norm * pos[i];
	}
	// refine grid
	grid->refine(refine_idx, refine_gps);

	//TODO: DEBUG only
	cout << "[Rank " << par.rank << "/" << par.size << "]: grid AFTER refine " << grid->getSize() << endl;

#if (SGI_PRINT_TIMER==1)
	if (par.is_master())
		cout << "SGI: MASTER refined grid in " << MPI_Wtime()-tic << " seconds." << endl;
#endif
	return true;
}
#endif

/**
 * Single-refine scheme: only MASTER refines and write grid, others read grid
 * REASON: see function above
 */
void SGI::refine_grid(double portion_to_refine)
{
	// Master refine and write grid
	if (par.is_master()) {
#if (SGI_PRINT_TIMER==1)
		double tic = MPI_Wtime();
#endif
		// Compute threshold number of grid points to be added
		// NOTE: to refine X points, maximum (2*dim*X) points can be added to grid
		int maxi = 10000;
		int thres = int(ceil(maxi / 2 / cfg.get_input_size()));
		// Number of points to refine
		std::size_t num_gps = this->grid->getSize();
		int refine_gps = int(ceil(num_gps * portion_to_refine));
		refine_gps = (refine_gps > thres) ? thres : refine_gps;
		// If no points to refine abort
		if (refine_gps < 1) {
			fflush(NULL);
			printf("SGI: refine grid failed (no points to refine). Program abort!\n");
			exit(EXIT_FAILURE);	
		};
		// Read posterior from file
		unique_ptr<double[]> pos (new double[num_gps]);
		mpiio_readwrite_posterior(true, 0, num_gps-1, &pos[0]);
		// For each gp, compute the refinement index
		double data_norm;
		DataVector refine_idx (num_gps);
		std::size_t output_size = cfg.get_output_size();
		for (std::size_t i=0; i<num_gps; i++) {
			data_norm = 0;
			for (std::size_t j=0; j < output_size; j++) {
				data_norm += (alphas[j][i] * alphas[j][i]);
			}
			data_norm = sqrt(data_norm);
			// refinement_index = |alpha| * posterior
			refine_idx[i] = data_norm * pos[i];
		}
		// refine grid
		grid->refine(refine_idx, refine_gps);
		mpiio_write_grid();
		std::size_t new_num_gps = grid->getSize();

		fflush(NULL);
		printf("SGI: total %zu gps, %zu added, range [%zu, %zu]\n",
				new_num_gps, new_num_gps-num_gps, num_gps, new_num_gps-1);
#if (SGI_PRINT_TIMER==1)
		printf("SGI: refined grid in %.6f seconds.\n", MPI_Wtime()-tic);
#endif
	}
	// Ensure read grid after its written completely
	MPI_Barrier(MPI_COMM_WORLD);
	// Others read grid
	if (!par.is_master()) mpiio_read_grid();
	
	return;
}

void SGI::mpiio_write_grid()
{
	// Pack grid into Char array
	string sg_str = grid->serialize();
	int count = static_cast<int>(sg_str.size());
	// Copy partial grid to string
	unique_ptr<char[]> buff (new char[count]);
	sg_str.copy(buff.get(), count, 0);
	// Write to file
	string ofile = cfg.get_param_string("global_output_path") + "/grid.mpibin";
	MPI_File fh;
	if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
			MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
		fflush(NULL);
		printf("ERROR: Rank[%d](%d) fail to open %s for grid write. Program abort!\n",
				par.rank, par.status, ofile);
		exit(EXIT_FAILURE);
	}
	if (MPI_File_write(fh, buff.get(), count, MPI_CHAR, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
		fflush(NULL);
		printf("ERROR: Rank[%d](%d) fail to write grid to %s. Program abort!\n",
				par.rank, par.status, ofile);
		exit(EXIT_FAILURE);
	}
	MPI_File_close(&fh);
	return;
}

void SGI::mpiio_read_grid()
{
	// Open file
	string ofile = cfg.get_param_string("global_output_path") + "/grid.mpibin";
	MPI_File fh;
	if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
			!= MPI_SUCCESS) {
		fflush(NULL);
		printf("ERROR: Rank[%d](%d) fail to open %s for grid read. Program abort!\n",
				par.rank, par.status, ofile);
		exit(EXIT_FAILURE);
	}
	// Get file size,create buffer
	long long int count;
	MPI_File_get_size(fh, &count);
	unique_ptr<char[]> buff (new char[count]);
	// Read from file
	if (MPI_File_read(fh, buff.get(), count, MPI_CHAR, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
		fflush(NULL);
		printf("ERROR: Rank[%d](%d) fail to read grid from %s. Program abort!\n",
				par.rank, par.status, ofile);
		exit(EXIT_FAILURE);
	}
	MPI_File_close(&fh);
	// Create serialized grid string
	string sg_str(buff.get());
	// Construct new grid from grid string
	grid.reset(Grid::unserialize(sg_str).release()); // create grid
	bbox.reset(create_boundingbox());
	grid->setBoundingBox(*bbox); // set up bounding box
	eval.reset(sgpp::op_factory::createOperationEval(*grid).release());
	return;
}

void SGI::mpiio_readwrite_data(
		bool is_read,
		std::size_t seq_min,
		std::size_t seq_max,
		double* buff)
{
	// Do something only when seq_min <= seq_max
	if (seq_min > seq_max) return;
	std::size_t output_size = cfg.get_output_size();
	string ofile = cfg.get_param_string("global_output_path") + "/data.mpibin";
	MPI_File fh;
	if (is_read) { // Read data from file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			cout << tools::red << "ERROR: [Rank " << par.rank << ", Status " << par.status
				<< "] failed to open " << ofile << " for data read. Program abort." << tools::reset << endl;
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		if (MPI_File_read_at(fh, seq_min*output_size*sizeof(double), buff,
				(seq_max-seq_min+1)*output_size, MPI_DOUBLE, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			cout << tools::red << "ERROR: [Rank " << par.rank << ", Status " << par.status
				<< "] failed to read data from " << ofile << ". Program abort." << tools::reset << endl;
			exit(EXIT_FAILURE);
		}

	} else { // Write data to file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			cout << tools::red << "ERROR: [Rank " << par.rank << ", Status " << par.status
				<< "] failed to open " << ofile << " for data write. Program abort." << tools::reset << endl;
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		if (MPI_File_write_at(fh, seq_min*output_size*sizeof(double), buff,
				(seq_max-seq_min+1)*output_size, MPI_DOUBLE, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			cout << tools::red << "ERROR: [Rank " << par.rank << ", Status " << par.status
				<< "] failed to write data to " << ofile << ". Program abort." << tools::reset << endl;
			exit(EXIT_FAILURE);
		}
	}
	MPI_File_close(&fh);
	return;
}

void SGI::mpiio_readwrite_posterior(
		bool is_read,
		std::size_t seq_min,
		std::size_t seq_max,
		double* buff)
{
	// Do something only when seq_min <= seq_max
	if (seq_min > seq_max) return;
	string ofile = cfg.get_param_string("global_output_path") + "/pos.mpibin";
	MPI_File fh;
	if (is_read) { // Read data from file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			cout << tools::red << "ERROR: [Rank " << par.rank << ", Status " << par.status
				<< "] failed to open " << ofile << " for posterior read. Program abort." << tools::reset << endl;
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		if (MPI_File_read_at(fh, seq_min*sizeof(double), buff,
				(seq_max-seq_min+1), MPI_DOUBLE, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			cout << tools::red << "ERROR: [Rank " << par.rank << ", Status " << par.status
				<< "] failed to read posterior from " << ofile << ". Program abort." << tools::reset << endl;
			exit(EXIT_FAILURE);
		}

	} else { // Write data to file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			cout << tools::red << "ERROR: [Rank " << par.rank << ", Status " << par.status
				<< "] failed to open " << ofile << " for posterior write. Program abort." << tools::reset << endl;
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		if (MPI_File_write_at(fh, seq_min*sizeof(double), buff,
				(seq_max-seq_min+1), MPI_DOUBLE, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			cout << tools::red << "ERROR: [Rank " << par.rank << ", Status " << par.status
				<< "] failed to write data to " << ofile << ". Program abort." << tools::reset << endl;
			exit(EXIT_FAILURE);
		}
	}
	MPI_File_close(&fh);
	return;
}

void SGI::mpina_get_local_range(
		const std::size_t& gmin,
		const std::size_t& gmax,
		std::size_t& lmin,
		std::size_t& lmax)
{
	std::size_t num_gps = gmax - gmin + 1;
	std::size_t trunk = num_gps / par.size;
	std::size_t rest = num_gps % par.size;
	if (par.rank < rest) {
		lmin = gmin + par.rank * (trunk + 1);
		lmax = lmin + trunk;
	} else {
		lmin = gmin + par.rank * trunk + rest;
		lmax = lmin + trunk - 1;
	}
	return;
}

void SGI::mpina_find_global_maxpos()
{
	if (par.size <= 1) return;
	//1. Prepare buffers
	std::size_t num_maxpos = cfg.get_param_sizet("mcmc_max_chains");
	// ensure every rank has the same length list
	// NOTE: num_maxpos is a small number, so no problem for padding
	for (std::size_t i=maxpos_list.size(); i < num_maxpos; ++i) {
		maxpos_list.emplace(-1.0, 0);
	}
	typedef std::pair<double, std::size_t> posseq;
	vector<posseq> sbuf (num_maxpos);
	vector<posseq> rbuf (num_maxpos * par.size);
	// copy local maxpos list into send buffer
	std::copy(maxpos_list.begin(), maxpos_list.end(), &sbuf[0]);
	//2. Allgather
	// allgather
	MPI_Allgather(&sbuf[0], sbuf.size(), par.MPI_POSSEQ, &rbuf[0], sbuf.size(), par.MPI_POSSEQ, MPI_COMM_WORLD);
	//3. Each rank picks out the top ones
	maxpos_list.clear();
	for (auto p: rbuf) { // insert to map is not cheap, do it only when necessary
		if (maxpos_list.size() < num_maxpos) { // Case list not full
			maxpos_list.insert(p);
		} else if (p.first > maxpos_list.begin()->first) { // Case list full but need to insert
			maxpos_list.insert(p);
			maxpos_list.erase(maxpos_list.begin());
		}
	}
	return;
}

void SGI::mpimw_master_compute(std::size_t gp_offset)
{
	double impi_adapt_freq = cfg.get_param_double("impi_adapt_freq_sec");
	std::size_t jobsize = cfg.get_param_sizet("sgi_masterworker_jobsize");
	// Determine total # jobs (compute only the newly added points)
	std::size_t added_gps = grid->getSize() - gp_offset;
	std::size_t num_jobs = (added_gps % jobsize > 0) ? (added_gps/jobsize + 1) : (added_gps/jobsize);
	vector<char> jobs (num_jobs, JOBTODO); // JOBTODO, JOBDONE, JOBINPROG
	vector<char> workers (par.size, RANKIDLE); // RANKACTIVE, RANKIDLE
	workers[par.master] = 'x'; // exclude master rank from any search

//TODO: debug output to be removed!
cout << tools::yellow;
print_jobs(jobs);
print_workers(workers);
cout << tools::reset << endl;

#if (IMPI==1)
	double tic = MPI_Wtime();
	double toc;
	int jobs_per_tic = 0;
#endif

	// Seed workers if any
	if (par.size > 1)
		mpimw_master_seed_workers(jobs, workers);

//TODO: debug output to be removed!
cout << tools::yellow;
print_jobs(jobs);
print_workers(workers);
cout << tools::reset << endl;

	// As long as not all jobs are done, keep working...
	while (!std::all_of(jobs.begin(), jobs.end(), [](char i){return i==JOBDONE;})) {

		if (par.size > 1) {
			// #1. Receive a finished job (only if there is any active worker)
			if (std::any_of(workers.begin(), workers.end(), [](char i){return i==RANKACTIVE;})) {
				mpimw_master_receive_done(jobs, workers);
#if (IMPI==1)
				jobs_per_tic++;
#endif
			}
			// #2. Send another job
			mpimw_master_send_todo(jobs, workers); // internally checks for todo jobs and idle workers
		} else { // if there is NO worker
			// #1. Find a todo job
			int jid = std::find(jobs.begin(), jobs.end(), JOBTODO) - jobs.begin();
			if (jid == jobs.size()) continue;
			jobs[jid] = JOBINPROG;
			// #2. Compute a job
			std::size_t seq_min, seq_max;
			mpimw_get_job_range(jid, gp_offset, seq_min, seq_max);
			compute_gp_range(seq_min, seq_max); // this also handles local maxpos list
			jobs[jid] = JOBDONE;
#if (IMPI==1)
			jobs_per_tic++;
#endif
		}

		// #3. Check for adaptation every impi_adapt_freq seconds
#if (IMPI==1)
		toc = MPI_Wtime()-tic;
		if (toc >= impi_adapt_freq) {
			// performance measure: # gps computed per second
			cout << "SGI.PERFORMANCE: # forward simulations per second = "
				<< double(jobs_per_tic * jobsize)/toc << endl;
			// Only when there are remaining jobs, it's worth trying to adapt
			if (std::any_of(jobs.begin(), jobs.end(), [](char i){return i==JOBTODO;})) {
				// Prepare workers for adapt (receive done jobs, then send adapt signal)
				if (par.size > 1)
					mpimw_master_prepare_adapt(jobs, workers, jobs_per_tic);
				// Adapt
				impi_adapt();
				workers.resize(par.size); // Update worker list
				for (auto w: workers) w = RANKIDLE; // Reset all workers to idle status
				workers[par.master] = 'x'; // Exclude master rank from any search
				// Seed workers again
				if (par.size > 1)
					mpimw_master_seed_workers(jobs, workers);
			}
			// reset timer
			tic = MPI_Wtime();
			jobs_per_tic = 0;
		} // end if-toc
#endif
	} // end while

	// All jobs done
	if (par.size > 1) {
		for (int i=1; i < par.size; ++i)
			MPI_Send(&jobs_per_tic, 1, MPI_INT, i, MPIMW_TAG_TERMINATE, MPI_COMM_WORLD);
	}
	return;
}

void SGI::mpimw_worker_compute(std::size_t gp_offset)
{
	// Setup variables
	int job_todo, job_done; // use separate buffers for send and receive
	std::size_t seq_min, seq_max;
	MPI_Status status;

	while (true) {
		// Receive a signal from MASTER
		if (MPI_Recv(&job_todo, 1, MPI_INT, par.master, MPI_ANY_TAG, MPI_COMM_WORLD, &status)
				!= MPI_SUCCESS) {
			cout << tools::red << "ERROR: [Rank " << par.rank << ", Status " << par.status
			   << "] failed to receive from MASTER. Program abort." << tools::reset << endl;
			exit(EXIT_FAILURE);
		}

		if (status.MPI_TAG == MPIMW_TAG_TERMINATE) break;

		if (status.MPI_TAG == MPIMW_TAG_WORK) {
			// get the job range and compute
			mpimw_get_job_range(job_todo, gp_offset, seq_min, seq_max);

			//TODO: DEBUG only
#if (SGI_DEBUG==1)
			if (job_todo != (seq_min - gp_offset)/cfg.get_param_sizet("sgi_masterworker_jobsize")) {
				cout << tools::red << "ERROR [Rank " << par.rank << "/" << par.size << " ("
					<< par.status << ")]: jobid " << job_todo << ", offset " << gp_offset
					<< ", range (" << seq_min << "--" << seq_max << ") mismatch!"
					<< tools::reset << endl;
				exit(EXIT_FAILURE);
			}
#endif

#if (SGI_PRINT_RANKPROGRESS==1)
	cout << tools::green << "[Rank " << par.rank << "/" << par.size << " (" << par.status
		<< ")]: computing job " << job_todo << " range (" << seq_min << "--" << seq_max << ") offset "
		<< gp_offset << tools::reset << endl;
#endif

			compute_gp_range(seq_min, seq_max);
			// tell master the job is done
			job_done = job_todo;
			mpimw_worker_send_done(job_done);
		}
#if (IMPI==1)
		if (status.MPI_TAG == MPIMW_TAG_ADAPT) impi_adapt();
#endif
	} // end while
	return;
}

void SGI::mpimw_sync_maxpos()
{
	if (par.size <= 1) return;
	//Prepare buffers
	std::size_t num_maxpos = cfg.get_param_sizet("mcmc_max_chains");
	typedef std::pair<double, std::size_t> posseq;
	vector<posseq> buf (num_maxpos);
	// copy local maxpos list into send buffer
	if (par.is_master()) {
		for (std::size_t i=maxpos_list.size(); i < num_maxpos; ++i) { // ensure maxpos_list has the right size
			maxpos_list.emplace(-1.0, 0);
		}
		std::copy(maxpos_list.begin(), maxpos_list.end(), &buf[0]);
	}
	// Master broadcast
	MPI_Bcast(&buf[0], num_maxpos, par.MPI_POSSEQ, par.master, MPI_COMM_WORLD);
	// workders unpack
	if (!par.is_master()) {
		maxpos_list.clear();
		for (auto p: buf) {
			maxpos_list.insert(p);
		}
	}
	return;
}	

void SGI::mpimw_get_job_range(
		const std::size_t& jobid,
		const std::size_t& seq_offset,
		std::size_t& seq_min,
		std::size_t& seq_max)
{
	std::size_t jobsize = cfg.get_param_sizet("sgi_masterworker_jobsize");
	seq_min = seq_offset + jobid * jobsize;
	seq_max = min(seq_min + jobsize - 1, grid->getSize()-1);
}

void SGI::mpimw_master_seed_workers(
		vector<char>& jobs,
		vector<char>& workers)
{
	vector<MPI_Request> sreq;
	sreq.reserve(par.size-1);
	vector<int> sbuf; //use unique send buffer for each Isend
	sbuf.reserve(par.size-1);
	// Seed workers avoiding MASTER itself
	for (int i=0; i < par.master; ++i) {
		sbuf.push_back( std::find(jobs.begin(), jobs.end(), JOBTODO)-jobs.begin() ); // fetch a todo job
		if (sbuf.back() >= jobs.size()) break; // no more todo jobs, stop seeding
		sreq.push_back(MPI_Request());
		MPI_Isend(&(sbuf.back()), 1, MPI_INT, i, MPIMW_TAG_WORK, MPI_COMM_WORLD, &(sreq.back()));
		jobs[sbuf.back()] = JOBINPROG; // mark job as "processing"
		workers[i] = RANKACTIVE; // mark worker as "active"
	}
	for (int i=par.master+1; i < par.size; ++i) {
		sbuf.push_back( std::find(jobs.begin(), jobs.end(), JOBTODO)-jobs.begin() ); // fetch a todo job
		if (sbuf.back() >= jobs.size()) break; // no more todo jobs, stop seeding
		sreq.push_back(MPI_Request());
		MPI_Isend(&(sbuf.back()), 1, MPI_INT, i, MPIMW_TAG_WORK, MPI_COMM_WORLD, &(sreq.back()));
		jobs[sbuf.back()] = JOBINPROG; // mark job as "processing"
		workers[i] = RANKACTIVE; // mark worker as "active"
	}
	if (sreq.size() > 0) {
		if (MPI_Waitall(sreq.size(), &sreq[0], MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			cout << tools::red << "ERROR: Master failed to Isend for seeding workers. Program abort."
				<< tools::reset << endl;
			exit(EXIT_FAILURE);
		}
	}
	return;
}

void SGI::mpimw_master_prepare_adapt(
		vector<char>& jobs,
		vector<char>& workers,
		int& jobs_per_tic)
{
#if (IMPI==1)
	vector<MPI_Request> sreq;
	sreq.reserve(par.size-1);
	vector<char> sbuf; // dummy send buffer, use unique send buffer for Isend
	sbuf.reserve(par.size-1);
	// If any worker still active, receive the finished job
	while ( std::any_of(workers.begin(), workers.end(), [](char i){return i==RANKACTIVE;}) ) {
		mpimw_master_receive_done(jobs, workers);
		jobs_per_tic++;
	}
	// Send "adapt signal" to all workers (avoiding MASTER itself)
	for (int i=0; i < par.master; i++) {
		sreq.push_back(MPI_Request());
		sbuf.push_back('1');
		MPI_Isend(&(sbuf.back()), 1, MPI_CHAR, i, MPIMW_TAG_ADAPT,
				MPI_COMM_WORLD, &(sreq.back()));
	}
	for (int i=par.master+1; i < par.size; i++) {
		sreq.push_back(MPI_Request());
		sbuf.push_back('1');
		MPI_Isend(&(sbuf.back()), 1, MPI_CHAR, i, MPIMW_TAG_ADAPT,
				MPI_COMM_WORLD, &(sreq.back()));
	}
	if (sreq.size() > 0) {
		if (MPI_Waitall(sreq.size(), &sreq[0], MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			cout << tools::red << "ERROR: Master failed to Isend adapt signal to workers. Program abort."
				<< tools::reset << endl;
			exit(EXIT_FAILURE);
		}
	}
	return;
#endif
}

void SGI::mpimw_master_send_todo(
		vector<char>& jobs,
		vector<char>& workers)
{
	// find a todo job
	int jid = std::find(jobs.begin(), jobs.end(), JOBTODO) - jobs.begin();
	if (jid >= jobs.size()) return; // no more todo jobs, nothing to do
	// find an idle worker
	int wid = std::find(workers.begin(), workers.end(), RANKIDLE) - workers.begin();
	if (wid >= workers.size()) return; // All workers are busy, nothing to do
 	// Send job (jid) to the idle worker (wid)
	MPI_Send(&jid, 1, MPI_INT, wid, MPIMW_TAG_WORK, MPI_COMM_WORLD);
	jobs[jid] = JOBINPROG; // Mark job as "in process"
	workers[wid] = RANKACTIVE; // Mark worker as "active"
	return;
}

void SGI::mpimw_master_receive_done(
		vector<char>& jobs,
		vector<char>& workers)
{
	// 1. Receive the finished jobid, and workerid
	int jid, wid;
	MPI_Status status;
	if (MPI_Recv(&jid, 1, MPI_INT, MPI_ANY_SOURCE, 12345, MPI_COMM_WORLD, &status) != MPI_SUCCESS) {
		cout << tools::red << "ERROR: MASTER failed to receive a finished jobid. Program abort." << tools::reset << endl;
		exit(EXIT_FAILURE);
	}
	wid = status.MPI_SOURCE;
	jobs[jid] = JOBDONE;
	workers[wid] = RANKIDLE;
	// 2. Receive maxpos list
	std::size_t num_maxpos = cfg.get_param_sizet("mcmc_max_chains");
	typedef std::pair<double, std::size_t> posseq;
	vector<posseq> buf (num_maxpos);
	if (MPI_Recv(&buf[0], num_maxpos, par.MPI_POSSEQ, wid, 123456, MPI_COMM_WORLD, &status) != MPI_SUCCESS) {
		cout << tools::red << "ERROR: MASTER failed to receive maxpos list from rank " << wid
			<< ". Program abort." << tools::reset << endl;
		exit(EXIT_FAILURE);
	}
	for (auto p: buf) {
		if (maxpos_list.size() < num_maxpos) { // Case list not full
			maxpos_list.insert(p);
		} else if (p.first > maxpos_list.begin()->first) { // Case list full but need to insert
			maxpos_list.insert(p);
			maxpos_list.erase(maxpos_list.begin());
		}
	}
	return;
}

void SGI::mpimw_worker_send_done(int jobid)
{
	//1. Send jobid
	MPI_Send(&jobid, 1, MPI_INT, par.master, 12345, MPI_COMM_WORLD);
	//1. Prepare send buffer
	std::size_t num_maxpos = cfg.get_param_sizet("mcmc_max_chains");
	typedef std::pair<double, std::size_t> posseq;
	vector<posseq> buf (num_maxpos);
	// ensure list has the right length
	// NOTE: num_maxpos is a small number, so no problem for padding
	for (std::size_t i=maxpos_list.size(); i < num_maxpos; ++i)
		maxpos_list.emplace(-1.0, 0);
	std::copy(maxpos_list.begin(), maxpos_list.end(), &buf[0]);
	//2. Send to master
	MPI_Send(&buf[0], num_maxpos, par.MPI_POSSEQ, par.master, 123456, MPI_COMM_WORLD);
	maxpos_list.clear();
	return;
}

// For debug only
void SGI::print_workers(vector<char> const& workers)
{
	cout << "Worker status: ";
	for (auto w: workers)
		cout << w << " ... ";
	cout << endl;
}

void SGI::print_jobs(vector<char> const& jobs)
{
	cout << "Job status: ";
	for (auto j: jobs)
		cout << j << " ... ";
	cout << endl;
}

// Verify a joining rank's grid (read from file) against MASTER's grid
// Only MASTER gets the correct compare result
bool SGI::verify_grid_from_read(int joinrank, MPI_Comm intercomm)
{
	if ((par.status == MPI_ADAPT_STATUS_JOINING) && (par.rank == joinrank)) {
		// Pack grid into Char array
		string sg_str = grid->serialize();
		int count = static_cast<int>(sg_str.size());
		// Copy grid to string
		unique_ptr<char[]> sbuf (new char[count]);
		sg_str.copy(sbuf.get(), count, 0);
		// Send to MASTER (in the remote group)
		if (MPI_Send(sbuf.get(), count, MPI_CHAR, par.master, 713, intercomm) != MPI_SUCCESS) {
			cout << tools::red << "ERROR: [Rank " << par.rank << " (" << par.status
				<< ")]: failed to send serialized grid (DEBUG: verify joining rank's grid)."
				<< tools::reset << endl;
			exit(EXIT_FAILURE);
		}
	}
	if (par.is_master()) {
		// Pack grid into Char array
		string sg_str_own = grid->serialize();

		MPI_Status status;
		MPI_Probe(joinrank, 713, intercomm, &status);
		int count;
		MPI_Get_count(&status, MPI_CHAR, &count);
		unique_ptr<char[]> rbuf (new char[count]);
		
		if (MPI_Recv(rbuf.get(), count, MPI_CHAR, joinrank, 713, intercomm, &status) != MPI_SUCCESS) {
			cout << tools::red << "ERROR: [Rank " << par.rank << " (" << par.status
				<< ")]: failed to receive serialized grid (DEBUG: verify joining rank's grid)."
				<< tools::reset << endl;
			exit(EXIT_FAILURE);
		}
		string sg_str_joinrank(rbuf.get());
		
		// compare
		if (sg_str_own.compare(sg_str_joinrank) == 0) {
			return true;
		}
	}
	return false;
}
