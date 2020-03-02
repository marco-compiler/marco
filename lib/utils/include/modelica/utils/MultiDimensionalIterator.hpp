#pragma once

#include <algorithm>
#include <iterator>
#include <limits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "modelica/utils/IRange.hpp"
namespace modelica
{
	class MultiDimensionalIterator
	{
		public:
		using Content = llvm::SmallVector<size_t, 3>;
		using Bounds = llvm::SmallVector<std::pair<size_t, size_t>, 3>;
		using iterator_category = std::forward_iterator_tag;
		using value_type = Content;
		using difference_type = size_t;
		using pointer = Content*;
		using reference = Content&;

		MultiDimensionalIterator(llvm::ArrayRef<size_t> endValues)
		{
			for (size_t val : endValues)
			{
				content.emplace_back(0);
				bounds.emplace_back(0, val);
			}
		}

		MultiDimensionalIterator(
				llvm::ArrayRef<size_t> startingVal, llvm::ArrayRef<size_t> endValues)

		{
			assert(startingVal.size() == endValues.size());
			llvm::copy(startingVal, std::back_inserter(content));
			for (size_t i : irange(startingVal.size()))
				bounds.emplace_back(startingVal[i], endValues[i]);
		}

		[[nodiscard]] bool operator==(const MultiDimensionalIterator& other) const
		{
			return content == other.content;
		}

		[[nodiscard]] bool operator!=(const MultiDimensionalIterator& other) const
		{
			return !(*this == other);
		}

		[[nodiscard]] const Content& operator*() const { return content; }
		[[nodiscard]] const Content* operator->() const { return &content; }
		[[nodiscard]] MultiDimensionalIterator operator++(int)	// NOLINT
		{
			auto copy = *this;
			++(*this);
			return copy;
		}

		MultiDimensionalIterator& operator++()
		{
			next();
			return *this;
		}

		void next()
		{
			for (size_t i = content.size() - 1;
					 i != std::numeric_limits<size_t>::max();
					 i--)
			{
				if (content[i]++ < bounds[i].second)
					break;
				content[i] = bounds[i].first;
			}
		}

		private:
		Content content;
		Bounds bounds;
	};

}	 // namespace modelica
