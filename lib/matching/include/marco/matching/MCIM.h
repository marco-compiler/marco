#ifndef MARCO_MATCHING_MCIM_H
#define MARCO_MATCHING_MCIM_H

#include "AccessFunction.h"
#include "MCIS.h"
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
    class Impl;
		MCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

    void apply(AccessFunction access);

    bool empty() const;
    void clear();

		private:
    std::unique_ptr<Impl> impl;
	};
}

#endif	// MARCO_MATCHING_MCIM_H
