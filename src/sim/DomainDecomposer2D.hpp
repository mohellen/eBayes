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


#ifndef SIM_DOMAINDECOMPOSER2D_HPP_
#define SIM_DOMAINDECOMPOSER2D_HPP_

#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <memory>

struct DomainBlock
{
public:
	// All indices and block IDs should be positive values
	// A negative value indicates invalid index or non-existing block
	const int id;
	const int enei;
	const int wnei;
	const int nnei;
	const int snei;
	const int xmin;
	const int xmax;
	const int ymin;
	const int ymax;

	DomainBlock(
			int my_id,
			int enei_id,
			int wnei_id,
			int nnei_id,
			int snei_id,
			int local_xmin,
			int local_xmax,
			int local_ymin,
			int local_ymax)
			: id(my_id),
			  enei(enei_id), wnei(wnei_id),
			  nnei(nnei_id), snei(snei_id),
			  xmin(local_xmin), xmax(local_xmax),
			  ymin(local_ymin), ymax(local_ymax)
	{}

	~DomainBlock() {}

	std::string toString();
};

/**
 * This class decomposes a computational grid domain into 2D blocks.
 * NOTE: grid cell/point indices are partitioned, NOT the continuous physical domain
 */
class DomainDecomposer2D
{
public:
	~DomainDecomposer2D() {};

	DomainDecomposer2D() {};

	static
	std::vector<DomainBlock*> gen_blocks(
			int num_blocks,
			int global_idx_xmin,
			int global_idx_xmax,
			int global_idx_ymin,
			int global_idx_ymax);

	static
	DomainBlock* gen_block(
			int num_blocks,
			int global_idx_xmin,
			int global_idx_xmax,
			int global_idx_ymin,
			int global_idx_ymax,
			int block_id);
};

#endif /* SIM_DOMAINDECOMPOSER2D_HPP_ */
