#pragma once
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/utils/IRange.hpp"
#include "marco/utils/MultiDimensionalIterator.hpp"

namespace marco
{
	class Interval
	{
		public:
		explicit Interval(size_t min, size_t max): minVal(min), maxVal(max)
		{
			assert(min < max);
		}
		Interval(std::initializer_list<size_t> list)
		{
			assert(list.size() == 2);
			minVal = *list.begin();
			maxVal = *(list.begin() + 1);
		}

		[[nodiscard]] bool contains(size_t element) const
		{
			return element < maxVal && element >= minVal;
		}

		[[nodiscard]] size_t min() const { return minVal; }
		[[nodiscard]] size_t max() const { return maxVal; }

		[[nodiscard]] auto begin() const { return IRangeIterator(minVal); }
		[[nodiscard]] auto end() const { return IRangeIterator(maxVal); }

		[[nodiscard]] bool isFullyContained(const Interval& other) const
		{
			return minVal < other.maxVal && minVal >= other.minVal &&
						 maxVal <= other.maxVal && maxVal > other.minVal;
		}

		[[nodiscard]] bool operator==(const Interval& other) const
		{
			return (other.minVal == minVal) && (other.maxVal == maxVal);
		}

		[[nodiscard]] bool operator!=(const Interval& other) const
		{
			return !(*this == other);
		}

		[[nodiscard]] bool operator>(const Interval& other) const
		{
			return minVal > other.minVal;
		}
		[[nodiscard]] bool operator<(const Interval& other) const
		{
			return minVal < other.minVal;
		}
		[[nodiscard]] bool operator>=(const Interval& other) const
		{
			return minVal >= other.minVal;
		}
		[[nodiscard]] bool operator<=(const Interval& other) const
		{
			return minVal <= other.minVal;
		}
		[[nodiscard]] size_t size() const { return maxVal - minVal; }

		private:
		size_t minVal;
		size_t maxVal;
	};

	class MultiDimInterval
	{
		public:
		MultiDimInterval(std::initializer_list<Interval> list)
				: intervals(std::move(list))
		{
		}

		MultiDimInterval(llvm::SmallVector<Interval, 2> intervals)
				: intervals(std::move(intervals))
		{
		}

		MultiDimInterval(llvm::ArrayRef<Interval> intervals)
				: intervals(intervals.begin(), intervals.end())
		{
		}

		MultiDimInterval(llvm::ArrayRef<size_t> point)
		{
			for (size_t p : point)
				intervals.emplace_back(p, p + 1);
		}

		[[nodiscard]] size_t dimensions() const { return intervals.size(); }
		[[nodiscard]] bool contains(llvm::ArrayRef<size_t> point) const;

		template<typename... T>
		[[nodiscard]] bool contains(T... coordinates) const
		{
			return contains({ coordinates... });
		}
		[[nodiscard]] const Interval& at(size_t index) const
		{
			return intervals[index];
		}

		[[nodiscard]] bool contains(const MultiDimInterval& other) const
		{
			assert(dimensions() == other.dimensions());
			for (size_t i : irange(dimensions()))
			{
				if (not intervals[i].contains(other.intervals[i].min()) or
						not intervals[i].contains(other.intervals[i].max() - 1))
					return false;
			}
			return true;
		}

		[[nodiscard]] auto begin() const { return intervals.begin(); }
		[[nodiscard]] auto end() const { return intervals.end(); }
		[[nodiscard]] auto begin() { return intervals.begin(); }
		[[nodiscard]] auto end() { return intervals.end(); }

		[[nodiscard]] llvm::iterator_range<MultiDimensionalIterator> contentRange()
				const;

		[[nodiscard]] bool operator==(const MultiDimInterval& other) const
		{
			return intervals == other.intervals;
		}

		[[nodiscard]] bool operator!=(const MultiDimInterval& other) const
		{
			return !(intervals == other.intervals);
		}

		[[nodiscard]] std::pair<bool, size_t> isExpansionOf(
				const MultiDimInterval& other) const;
		void expand(const MultiDimInterval& other);

		[[nodiscard]] int confront(const MultiDimInterval& other) const;
		[[nodiscard]] bool operator>(const MultiDimInterval& other) const
		{
			return confront(other) > 0;
		}
		[[nodiscard]] bool operator<(const MultiDimInterval& other) const
		{
			return confront(other) < 0;
		}
		[[nodiscard]] bool operator>=(const MultiDimInterval& other) const
		{
			return confront(other) >= 0;
		}
		[[nodiscard]] bool operator<=(const MultiDimInterval& other) const
		{
			return confront(other) <= 0;
		}
		[[nodiscard]] size_t size() const
		{
			if (intervals.empty())
				return 0;

			size_t size = 1;
			for (const auto& el : intervals)
				size *= el.size();

			return size;
		}

		[[nodiscard]] bool empty() const { return size() == 0; }

		[[nodiscard]] MultiDimInterval replacedDimension(
				size_t dimension, size_t newLeft, size_t newRight) const
		{
			assert(intervals.size() > dimension);	 // NOLINT
			auto intervalsCopy = intervals;
			intervalsCopy[dimension] = Interval(newLeft, newRight);
			return MultiDimInterval(std::move(intervalsCopy));
		}

		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		[[nodiscard]] llvm::SmallVector<MultiDimInterval, 3> cutOnDimension(
				size_t dimension, llvm::ArrayRef<size_t> cutLines) const;

		template<typename BackInserter>
		void cutOnDimension(
				size_t dimension,
				llvm::ArrayRef<size_t> cutLines,
				BackInserter backInserter) const
		{
			auto vec = cutOnDimension(dimension, cutLines);
			std::move(vec.begin(), vec.end(), backInserter);
		}

		[[nodiscard]] bool isFullyContained(const MultiDimInterval& other) const;

		private:
		llvm::SmallVector<Interval, 2> intervals;
	};
	[[nodiscard]] bool areDisjoint(const Interval& left, const Interval& right);

	[[nodiscard]] bool areDisjoint(
			const MultiDimInterval& left, const MultiDimInterval& right);
	MultiDimInterval intersection(
			const MultiDimInterval& left, const MultiDimInterval& right);

}	 // namespace marco
