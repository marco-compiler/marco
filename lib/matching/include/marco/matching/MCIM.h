#ifndef MARCO_MATCHING_MCIM_H
#define MARCO_MATCHING_MCIM_H

#include "Range.h"

namespace marco::matching
{
	/**
	 * Multidimensional Compressed Index Map (MCIM).
	 *
	 * It replaces the multidimensional incidence matrices in order to achieve
	 * O(1) scaling.
	 */
	class MCIM
	{
		public:
		MCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

		~MCIM();

		private:
		MultidimensionalRange equationRanges;
		MultidimensionalRange variableRanges;

		bool* data;
	};
}

#endif	// MARCO_MATCHING_MCIM_H
