#ifndef MARCO_MATCHING_MCIS_H
#define MARCO_MATCHING_MCIS_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include "Range.h"

namespace marco::matching
{
	/**
	 * Multidimensional Compressed Index Set (MCIS).
	 *
	 * It replaces the multidimensional vectors in order to achieve O(1) scaling.
	 */
	class MCIS
	{
		private:
		using Container = llvm::SmallVector<MultidimensionalRange, 2>;

		public:
		MCIS(llvm::ArrayRef<MultidimensionalRange> ranges);

		MultidimensionalRange operator[](size_t index) const;

		private:
		Container ranges;
	};
}

#endif	// MARCO_MATCHING_MCIS_H
