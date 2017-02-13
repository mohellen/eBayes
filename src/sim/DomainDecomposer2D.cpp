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

#include <sim/DomainDecomposer2D.hpp>

#include <iostream> // TODO remove this line

#define INVALID -1

using namespace std;

vector<DomainBlock*> DomainDecomposer2D::gen_blocks(
		int num_blocks,
		int global_idx_xmin,
		int global_idx_xmax,
		int global_idx_ymin,
		int global_idx_ymax)
{
	/**
	 * Example: for numBlocks = 10, we use up to 2^3=8 blocks,
	 *    for which the domain is decomposed as
	 *
	 *  		  |   |   |
	 *  		4 | 5 | 6 | 7
	 * 		  ------------------
	 * 	 		0 | 1 | 2 | 3
	 *     		  |   |   |
	 *
	 *    Row-major storage. Inner loop over x-direction.
	 **/

	// Check num_blocks size
	if (num_blocks <= 0) {
		vector<DomainBlock*> blocks;
		blocks.clear();
		return blocks;
	}

	int nx; 	// # blocks in x-direction
	int ny; 	// # blocks in y-direction
	int l = int(floor(log2(num_blocks))); // We only use up to 2^l blocks
	if (l % 2 == 0) {
		nx = int(pow(2, l/2));
		ny = int(pow(2, l/2));
	} else {
		nx = int(pow(2, l/2+1));
		ny = int(pow(2, l/2));
	}

	// Output array of blocks
	vector<DomainBlock*> blocks (nx*ny, nullptr);

	// convenient closure
	auto idx = [](int j, int i, int nx) -> int {return j * nx + i;};

	int myid, enei, wnei, nnei, snei;
	int xmin, xmax, ymin, ymax;
	int xrange = global_idx_xmax - global_idx_xmin + 1;
	int yrange = global_idx_ymax - global_idx_ymin + 1;
	int xtrunk = xrange / nx;
	int ytrunk = yrange / ny;
	int xrest = xrange % nx;
	int yrest = yrange % ny;

	for (int j=0; j < ny; j++) {
		for (int i=0; i < nx; i++) {
			// Compute neighbors
			myid = idx(j,i,nx);
			enei = (i != nx-1) ? ( idx(j,i+1,nx) ) : (INVALID);
			wnei = (i != 0)    ? ( idx(j,i-1,nx) ) : (INVALID);
			nnei = (j != ny-1) ? ( idx(j+1,i,nx) ) : (INVALID);
			snei = (j != 0)    ? ( idx(j-1,i,nx) ) : (INVALID);

			// Compute index ranges
			if (i < xrest) {
				xmin = i * (xtrunk+1) + global_idx_xmin;
				xmax = xmin + xtrunk;
			} else {
				xmin = i * xtrunk + global_idx_xmin + xrest;
				xmax = xmin + xtrunk - 1;
			}
			if (j < yrest) {
				ymin = j * (ytrunk+1) + global_idx_ymin;
				ymax = ymin + ytrunk;
			} else {
				ymin = j * ytrunk + global_idx_ymin + yrest;
				ymax = ymin + ytrunk - 1;
			}
			blocks[myid] = new DomainBlock(myid, enei, wnei, nnei, snei,
											xmin, xmax, ymin, ymax);
		}
	}
	return blocks;
}

DomainBlock* DomainDecomposer2D::gen_block(
		int num_blocks,
		int global_idx_xmin,
		int global_idx_xmax,
		int global_idx_ymin,
		int global_idx_ymax,
		int block_id)
{
	// Check num_blocks size
	if (num_blocks <= 0)
		return (new DomainBlock(-1, -1, -1, -1, -1, -1, -1, -1, -1));

	DomainBlock* b = nullptr; // Output
	int nx; // # blocks in x-direction
	int ny; // # blocks in y-direction
	int l = int(floor(log2(num_blocks))); // We only use up to 2^l blocks
	if (l % 2 == 0) {
		nx = int(pow(2, l/2));
		ny = int(pow(2, l/2));
	} else {
		nx = int(pow(2, l/2+1));
		ny = int(pow(2, l/2));
	}

	// convenient closure
	auto idx = [](int j, int i, int nx) -> int {return j * nx + i;};

	// Get 2D index from block_id
	// block_id = j * nx + i, therefore
	int i = block_id % nx;
	int j = block_id / nx;

	// Compute neighbors
	int enei, wnei, nnei, snei;
	enei = (i != nx-1) ? ( idx(j,i+1,nx) ) : (INVALID);
	wnei = (i != 0)    ? ( idx(j,i-1,nx) ) : (INVALID);
	nnei = (j != ny-1) ? ( idx(j+1,i,nx) ) : (INVALID);
	snei = (j != 0)    ? ( idx(j-1,i,nx) ) : (INVALID);

	// Compute index ranges
	int xmin, xmax, ymin, ymax;
	int xrange = global_idx_xmax - global_idx_xmin + 1;
	int yrange = global_idx_ymax - global_idx_ymin + 1;
	int xtrunk = xrange / nx;
	int ytrunk = yrange / ny;
	int xrest = xrange % nx;
	int yrest = yrange % ny;

	if (i < xrest) {
		xmin = i * (xtrunk+1) + global_idx_xmin;
		xmax = xmin + xtrunk;
	} else {
		xmin = i * xtrunk + global_idx_xmin + xrest;
		xmax = xmin + xtrunk - 1;
	}
	if (j < yrest) {
		ymin = j * (ytrunk+1) + global_idx_ymin;
		ymax = ymin + ytrunk;
	} else {
		ymin = j * ytrunk + global_idx_ymin + yrest;
		ymax = ymin + ytrunk - 1;
	}

	// Produce the object (may or may not be valid based on block_id)
	if ((block_id >= 0) && (block_id < nx*ny)) {
		b = new DomainBlock(block_id, enei, wnei, nnei, snei,
								 xmin, xmax, ymin, ymax);
	} else {
		b = new DomainBlock(-1, -1, -1, -1, -1, -1, -1, -1, -1);
	}
	return b;
}

string DomainBlock::toString()
{
	stringstream ss;
	ss << "Block " << id
			<< ": enei = " << enei << ", wnei = " << wnei
			<< ", nnei = " << nnei << ", snei = " << snei << "\n"
	        << "Domain [xmin, xmax, ymin, ymax] = ["
			<< xmin << ", " << xmax << ", " << ymin << ", " << ymax << "]\n";
	return ss.str();
}
