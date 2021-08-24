#pragma once

#include <iterator>

namespace marco
{
	template<typename ModExp>
	class ModChildrenIterator
	{
		public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = ModExp;
		using difference_type = size_t;
		using pointer = ModExp*;
		using reference = ModExp&;

		ModChildrenIterator(ModExp& exp): exp(exp), childIndex(0) {}
		ModChildrenIterator(ModExp& exp, size_t current)
				: exp(exp), childIndex(current)
		{
		}
		[[nodiscard]] bool operator==(const ModChildrenIterator& other) const
		{
			return exp == other.exp && childIndex == other.childIndex;
		}

		[[nodiscard]] bool operator!=(const ModChildrenIterator& other) const
		{
			return !(*this == other);
		}
		[[nodiscard]] ModExp& operator*() const { return exp.getChild(childIndex); }
		[[nodiscard]] ModExp& operator*() { return exp.getChild(childIndex); }
		[[nodiscard]] ModExp* operator->() const
		{
			return &exp.getChild(childIndex);
		}
		[[nodiscard]] ModExp* operator->() { return &exp.getChild(childIndex); }
		const ModChildrenIterator operator++(int)	 // NOLINT
		{
			auto copy = *this;
			++(*this);
			return copy;
		}
		ModChildrenIterator& operator++()
		{
			childIndex++;
			return *this;
		}

		private:
		ModExp& exp;
		size_t childIndex;
	};
}	 // namespace marco
