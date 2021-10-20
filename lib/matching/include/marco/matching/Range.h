#ifndef MARCO_MATCHING_INDEXSET_H
#define MARCO_MATCHING_INDEXSET_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace marco::matching
{
	class Range
	{
		public:
		Range(long begin, long end);

		long getBegin() const;
		long getEnd() const;

		size_t size() const;

		bool intersects(Range other) const;

		private:
		long begin;
		long end;
	};

	class MultidimensionalRange
	{
		private:
		using Container = llvm::SmallVector<Range, 2>;

		public:
		using iterator = Container::iterator;
		using const_iterator = Container::const_iterator;

		MultidimensionalRange(llvm::ArrayRef<Range> ranges);

		Range operator[](size_t index) const;

		unsigned int rank() const;

		unsigned int flatSize() const;

		bool intersects(MultidimensionalRange other) const;

		iterator begin();
		const_iterator begin() const;

		iterator end();
		const_iterator end() const;

		private:
		Container ranges;
	};
}

#endif	// MARCO_MATCHING_INDEXSET_H
