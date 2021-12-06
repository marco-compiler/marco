#ifndef MARCO_MATCHING_MCIS_H
#define MARCO_MATCHING_MCIS_H

#include <list>
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
		MCIS(llvm::ArrayRef<MultidimensionalRange> ranges = llvm::None);

    MultidimensionalRange& operator[](size_t index);
		const MultidimensionalRange& operator[](size_t index) const;

    bool contains(llvm::ArrayRef<Range::data_type> element) const;
    bool contains(const MultidimensionalRange& range) const;

    void add(MultidimensionalRange range);

		private:
    void sort();
    void merge();

    std::list<MultidimensionalRange> ranges;
	};
}

#endif	// MARCO_MATCHING_MCIS_H
