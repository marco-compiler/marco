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

    friend std::ostream& operator<<(std::ostream& stream, const MCIS& obj);

		const MultidimensionalRange& operator[](size_t index) const;

    MCIS& operator+=(llvm::ArrayRef<Range::data_type> rhs);
    MCIS& operator+=(const MultidimensionalRange& rhs);
    MCIS& operator+=(const MCIS& rhs);

    MCIS operator+(llvm::ArrayRef<Range::data_type> rhs) const;
    MCIS operator+(const MultidimensionalRange& rhs) const;
    MCIS operator+(const MCIS& rhs) const;

    MCIS& operator-=(const MultidimensionalRange& rhs);
    MCIS& operator-=(const MCIS& rhs);

    MCIS operator-(const MultidimensionalRange& rhs) const;
    MCIS operator-(const MCIS& rhs) const;

    bool empty() const;
    size_t size() const;

    const_iterator begin() const;
    const_iterator end() const;

    bool contains(llvm::ArrayRef<Range::data_type> element) const;
    bool contains(const MultidimensionalRange& other) const;

    bool overlaps(const MultidimensionalRange& other) const;
    bool overlaps(const MCIS& other) const;

    MCIS intersect(const MCIS& other) const;

    MCIS complement(const MultidimensionalRange& other) const;

		private:
    void sort();
    void merge();

    std::list<MultidimensionalRange> ranges;
	};
}

#endif	// MARCO_MATCHING_MCIS_H
