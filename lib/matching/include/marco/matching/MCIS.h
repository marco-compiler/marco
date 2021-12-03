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
		public:
		MCIS(llvm::ArrayRef<MultidimensionalRange> ranges);

    bool operator==(const MCIS& other) const;
    bool operator!=(const MCIS& other) const;

    MultidimensionalRange& operator[](size_t index);
		const MultidimensionalRange& operator[](size_t index) const;

    bool contains(llvm::ArrayRef<Range::data_type> element) const;
    void add(MultidimensionalRange range);

		private:
    void sort();

    llvm::SmallVector<MultidimensionalRange, 2> ranges;
	};
}

#endif	// MARCO_MATCHING_MCIS_H
