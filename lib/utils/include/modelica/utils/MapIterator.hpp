#pragma once

namespace modelica
{
	template<typename Iterator, typename Value>
	class MapIterator
	{
		public:
		using iterator_category = typename Iterator::iterator_category;
		using value_type = Value&;
		using difference_type = typename Iterator::difference_type;
		using pointer = Value*;
		using reference = Value&;

		MapIterator(Iterator iter): iter(iter) {}

		[[nodiscard]] bool operator==(const MapIterator& other) const
		{
			return iter == other.iter;
		}
		[[nodiscard]] bool operator!=(const MapIterator& other) const
		{
			return iter != other.iter;
		}
		[[nodiscard]] reference operator*() const { return iter->second; }
		[[nodiscard]] reference operator*() { return iter->second; }
		[[nodiscard]] pointer operator->() { return &iter->second; }
		[[nodiscard]] pointer operator->() const { return &iter->second; }
		const MapIterator operator++(int)	 // NOLINT
		{
			auto copy = *this;
			++(*this);
			return copy;
		}
		MapIterator& operator++()
		{
			iter++;
			return *this;
		}
		const MapIterator operator--(int)	 // NOLINT
		{
			auto copy = *this;
			--(*this);
			return copy;
		}
		MapIterator& operator--()
		{
			iter--;
			return *this;
		}

		private:
		Iterator iter;
	};

};	// namespace modelica
