#ifndef MARCO_MATCHING_INDEXSET_H
#define MARCO_MATCHING_INDEXSET_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace marco::matching
{
	template<typename ValueType>
	class RangeIterator
	{
		public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = ValueType;
		using difference_type = std::ptrdiff_t;
		using pointer = ValueType*;
		using reference = ValueType&;

		RangeIterator(ValueType begin, ValueType start, ValueType end) : begin(begin), current(start), end(end)
		{
			assert(begin <= end);
		}

		operator bool() const
		{
			return current != end;
		}

		bool operator==(const RangeIterator& it) const
		{
			return current == it.current && begin == it.begin && end == it.end;
		}

		bool operator!=(const RangeIterator& it) const
		{
			return current != it.current || begin != it.begin || end != it.end;
		}

		RangeIterator& operator++()
		{
			current = std::min(current + 1, end);
			return *this;
		}

		RangeIterator operator++(int)
		{
			auto temp = *this;
			current = std::min(current + 1, end);
			return temp;
		}

		value_type operator*()
		{
			return current;
		}

		private:
		ValueType begin;
		ValueType current;
		ValueType end;
	};

	/**
	 * 1-D half-open range [a,b).
	 */
	class Range
	{
		public:
		using data_type = long;

		using iterator = RangeIterator<data_type>;
		using const_iterator = RangeIterator<data_type>;

		Range(data_type begin, data_type end);

		long getBegin() const;
		long getEnd() const;

		size_t size() const;

		bool contains(data_type value) const;

		bool intersects(Range other) const;

		iterator begin();
		const_iterator begin() const;

		iterator end();
		const_iterator end() const;

		private:
		data_type _begin;
		data_type _end;
	};

	template<typename ValueType>
	class MultidimensionalRangeIterator
	{
		public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = llvm::ArrayRef<ValueType>;
		using difference_type = std::ptrdiff_t;
		using pointer = llvm::ArrayRef<ValueType>*;
		using reference = llvm::ArrayRef<ValueType>&;

		MultidimensionalRangeIterator(
				llvm::ArrayRef<Range> ranges,
				std::function<RangeIterator<ValueType>(const Range&)> initFunction)
		{
			for (const auto& range : ranges)
			{
				beginIterators.push_back(range.begin());
				auto it = initFunction(range);
				currentIterators.push_back(it);
				endIterators.push_back(range.end());
				indexes.push_back(*it);
			}

			assert(ranges.size() == beginIterators.size());
			assert(ranges.size() == currentIterators.size());
			assert(ranges.size() == endIterators.size());
			assert(ranges.size() == indexes.size());
		}

		operator bool() const
		{
			for (const auto& [current, end] : llvm::zip(currentIterators, endIterators))
				if (current != end)
					return true;

			return false;
		}

		bool operator==(const MultidimensionalRangeIterator& it) const
		{
			return currentIterators == it.currentIterators;
		}

		bool operator!=(const MultidimensionalRangeIterator& it) const
		{
			return currentIterators != it.currentIterators;
		}

		MultidimensionalRangeIterator& operator++()
		{
			fetchNext();
			return *this;
		}

		MultidimensionalRangeIterator operator++(int)
		{
			auto temp = *this;
			fetchNext();
			return temp;
		}

		value_type operator*()
		{
			return indexes;
		}

		private:
		void fetchNext()
		{
			size_t size = indexes.size();

			auto findIndex = [&]() -> std::pair<bool, size_t> {
				for (size_t i = 0, e = size; i < e; ++i)
				{
					size_t pos = e - i - 1;

					if (++currentIterators[pos] != endIterators[pos])
						return std::make_pair(true, pos);
				}

				return std::make_pair(false, 0);
			};

			std::pair<bool, size_t> index = findIndex();

			if (index.first)
			{
				size_t pos = index.second;

				indexes[pos] = *currentIterators[pos];

				for (size_t i = pos + 1; i < size; ++i)
				{
					currentIterators[i] = beginIterators[i];
					indexes[i] = *currentIterators[i];
				}
			}
		}

		llvm::SmallVector<RangeIterator<ValueType>, 3> beginIterators;
		llvm::SmallVector<RangeIterator<ValueType>, 3> currentIterators;
		llvm::SmallVector<RangeIterator<ValueType>, 3> endIterators;
		llvm::SmallVector<ValueType, 3> indexes;
		llvm::ArrayRef<Range> ranges;
	};

	/**
	 * n-D range. Each dimension is half-open as the 1-D range.
	 */
	class MultidimensionalRange
	{
		private:
		using Container = llvm::SmallVector<Range, 2>;

		public:
		using data_type = Range::data_type;

		using iterator = MultidimensionalRangeIterator<data_type>;
		using const_iterator = MultidimensionalRangeIterator<data_type>;

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
