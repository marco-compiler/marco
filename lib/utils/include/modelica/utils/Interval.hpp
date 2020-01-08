#pragma once
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace modelica
{
	class Interval
	{
		public:
		Interval(size_t min, size_t max): minVal(min), maxVal(max)
		{
			assert(min < max);	// NOLINT
		}
		Interval(std::initializer_list<size_t> list)
		{
			assert(list.size() == 2);	 // NOLINT
			minVal = *list.begin();
			maxVal = *(list.begin() + 1);
		}

		[[nodiscard]] bool contains(size_t element) const
		{
			return element < maxVal && element >= minVal;
		}

		[[nodiscard]] size_t min() const { return minVal; }
		[[nodiscard]] size_t max() const { return maxVal; }
		template<typename Callable>
		void for_all(Callable&& c) const
		{
			for (size_t v = minVal; v < maxVal; v++)
				c(v);
		}

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

		[[nodiscard]] auto begin() const { return intervals.begin(); }
		[[nodiscard]] auto end() const { return intervals.end(); }
		[[nodiscard]] auto begin() { return intervals.begin(); }
		[[nodiscard]] auto end() { return intervals.end(); }

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
			size_t size = 1;
			for (const auto& el : intervals)
				size *= el.size();

			return size;
		}

		[[nodiscard]] MultiDimInterval replacedDimension(
				size_t dimension, size_t newLeft, size_t newRight) const
		{
			assert(intervals.size() > dimension);	 // NOLINT
			auto intervalsCopy = intervals;
			intervalsCopy[dimension] = Interval(newLeft, newRight);
			return MultiDimInterval(std::move(intervalsCopy));
		}

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

}	 // namespace modelica
