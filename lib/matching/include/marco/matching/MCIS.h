#ifndef MARCO_MATCHING_MCIS_H
#define MARCO_MATCHING_MCIS_H

#include <list>
#include <llvm/ADT/ArrayRef.h>

#include "Range.h"

namespace marco::matching::detail
{
	/**
	 * Multidimensional Compressed Index Set (MCIS).
	 *
	 * It replaces the multidimensional vectors in order to achieve O(1) scaling.
	 */
	class MCIS
	{
    private:
    using Container = std::list<MultidimensionalRange>;

		public:
    using const_iterator = Container::const_iterator;

		MCIS(llvm::ArrayRef<MultidimensionalRange> ranges = llvm::None);

		const MultidimensionalRange& operator[](size_t index) const;

    size_t size();

    const_iterator begin() const;
    const_iterator end() const;

    bool contains(llvm::ArrayRef<Range::data_type> element) const;
    bool contains(const MultidimensionalRange& other) const;

    bool overlaps(const MultidimensionalRange& other) const;
    bool overlaps(const MCIS& other) const;

    MCIS intersect(const MCIS& other) const;

    void add(llvm::ArrayRef<Range::data_type> element);
    void add(const MultidimensionalRange& range);
    void add(const MCIS& other);

		private:
    void sort();
    void merge();

    std::list<MultidimensionalRange> ranges;
	};
}

#endif	// MARCO_MATCHING_MCIS_H
