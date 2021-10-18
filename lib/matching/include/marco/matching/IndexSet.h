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

		private:
		long begin;
		long end;
	};

	class RangeSet
	{
		public:
		RangeSet(llvm::ArrayRef<Range> ranges);

		Range operator[](size_t index) const;

		unsigned int rank();

		private:
		llvm::SmallVector<Range, 2> ranges;
	};
}

#endif	// MARCO_MATCHING_INDEXSET_H
