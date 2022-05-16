#pragma once
#include "./StdFunctions.h"
#include "./Print.h"
#include <cstdint>
#include <initializer_list>
//#include <iterator>

/*
#include <array>
#include <cstde::assertt>
#include <initializer_list>
#include <iostream>
#include <vector>
*/
template <typename T>
class ArrayIterator;

/// This class represents how the arrays are described inside the Modelica IR.
/// Note that the Rank template argument is used only to allocate an
/// appropriately sized array of sizes. It is not used during the elements
/// access, where pointer arithmetics are used instead; this last point
/// allows to instantiate iterators on descriptor whose rank is not known
/// at compile time, and thus allows to iterate over descriptors received
/// from the IR calls.
///
/// @tparam T 	  data type
/// @tparam Rank number of dimensions
template<typename T, unsigned int Rank>
class ArrayDescriptor
{
	public:
	using iterator = ArrayIterator<T>;
	using const_iterator = ArrayIterator<const T>;

	using rank_t = uint64_t;
	using dimension_t = uint64_t;

	/// Utility constructor for tests.
	///
	/// @param data  data pointer
	/// @param sizes sizes of the array
	template<unsigned long Size>
	ArrayDescriptor(stde::array<T, Size>& data)
			: data(data.data()), rank(1), sizes{}
	{
		stde::assertt(Rank == rank);
		this->sizes[0] = Size;
	}

	/// Utility constructor for tests.
	///
	/// @param data  data pointer
	/// @param sizes sizes of the array
	ArrayDescriptor(T* data, stde::array<dimension_t, Rank> sizes)
			: data(data), rank(sizes.size()), sizes{}
	{
    for (rank_t i = 0; i < rank; ++i) {
      this->sizes[i] = sizes[i];
    }
	}

	/// Get element at offset.
	///
	/// @param offset	 index of the flat array
	/// @return value
	T& operator[](dimension_t offset)
	{
		stde::assertt(data != nullptr);
		return data[offset];
	}

	/// Get element at offset.
	///
	/// @param offset	 index of the flat array
	/// @return value
	const T& operator[](dimension_t offset) const
	{
		stde::assertt(data != nullptr);
		return data[offset];
	}

	void dump() const
	{	stde::ostream os;
		dump(os);
	}

	void dump(stde::ostream& os) const
	{	/*
		os << "Array descriptor\n";
		os << "  - address: " << getData() << "\n";
		os << "  - rank: " << getRank() << "\n";
		os << "  - values: " << *this << "\n";
		os << "\n";*/
		print_char("Array descriptor\n\r");
		print_char(" - address: ");
		print_serial( getData());
		print_char("\n\r");
		print_char(" - rank: ");
		print_serial( getRank());
		print_char("\n\r");
		print_char(" - values: ");
		print_serial( *this);
		print_char("\n\r");
	}

  template<typename Index,typename stde::enable_if<stde::is_integral<Index>::value>::type* = nullptr>
  T& get(const Index& index)
  {
    stde::assertt(data != nullptr);
    return (*this)[index];
  }

  template<typename Index,typename stde::enable_if<stde::is_integral<Index>::value>::type* = nullptr>
  const T& get(const Index& index) const
  {
    stde::assertt(data != nullptr);
    return (*this)[index];
  }

  template<typename First, typename Second, typename... Others>
  T& get(const First& firstIndex, const Second& secondIndex, Others&&... otherIndexes)
  {
    return get({
      firstIndex,
      secondIndex,
      stde::forward<Others>(otherIndexes)...
    });
  }

  template<typename First, typename Second, typename... Others>
  const T& get(const First& firstIndex, const Second& secondIndex, Others&&... otherIndexes) const
  {
    return get({
        firstIndex,
        secondIndex,
        stde::forward<Others>(otherIndexes)...
    });
  }

  template<typename Index>
  T& get(std::initializer_list<Index> indexes)
  {
    stde::assertt(data != nullptr);
    dimension_t offset = computeOffset(indexes);
    return (*this)[offset];
  }

  template<typename Index>
  const T& get(std::initializer_list<Index> indexes) const
  {
    stde::assertt(data != nullptr);
    dimension_t offset = computeOffset(indexes);
    return (*this)[offset];
  }

  template<typename Indexes,typename stde::enable_if<!stde::is_integral<Indexes>::value>::type* = nullptr>
	T& get(const Indexes& indexesBegin)
	{
		stde::assertt(data != nullptr);
		dimension_t offset = computeOffset(indexesBegin);
		return (*this)[offset];
	}

  template<typename Indexes,typename stde::enable_if<!stde::is_integral<Indexes>::value>::type* = nullptr>
	const T& get(const Indexes& indexes) const
	{
		stde::assertt(data != nullptr);
		dimension_t offset = computeOffset(indexes);
		return (*this)[offset];
	}

  template<typename Index>
  void set(std::initializer_list<Index> indexes, T value)
  {
    stde::assertt(data != nullptr);
    dimension_t offset = computeOffset(indexes);
    (*this)[offset] = value;
  }

  template<typename Indexes,typename stde::enable_if<!stde::is_integral<Indexes>::value>::type* = nullptr>
	void set(const Indexes& indexes, T value)
	{
		stde::assertt(data != nullptr);
		dimension_t offset = computeOffset(indexes);
		(*this)[offset] = value;
	}

	T* getData() const
	{
		return data;
	}

	rank_t getRank() const
	{
		return rank;
	}

	dimension_t getDimension(rank_t index) const
	{
		stde::assertt(index >= 0 && index < rank);
		return sizes[index];
	}

	dimension_t getNumElements() const
	{
		dimension_t result = 1;

    for (rank_t i = 0, e = getRank(); i < e; ++i) {
      result *= getDimension(i);
    }

		return result;
	}

	iterator begin()
	{
		return iterator(*this, false);
	}

	const_iterator begin() const
	{
		return const_iterator(*this, false);
	}

	iterator end()
	{
		return iterator(*this, true);
	}

	const_iterator end() const
	{
		return const_iterator(*this, true);
	}

	bool hasSameSizes() const
	{
		for (rank_t i = 1; i < rank; ++i) {
      if (sizes[i] != sizes[0]) {
        return false;
      }
    }

		return true;
	}

	private:
  template<typename Indexes>
	dimension_t computeOffset(Indexes indexes) const
	{
    auto begin = std::begin(indexes);
    auto end = std::end(indexes);

    if (begin == end) {
      return 0;
    }

    dimension_t offset = *begin;
    ++begin;

    size_t currentDimension = 1;

    for (auto it = begin; it != end; ++it) {
      dimension_t size = getDimension(currentDimension);
      stde::assertt(size > 0);
      auto index = *it;
      stde::assertt(index >= 0);
      offset = offset * size + index;
      ++currentDimension;
    }

    stde::assertt(currentDimension == rank && "Wrong number of indexes");
    return offset;
	}

	T* data;
	rank_t rank;

	// The sizes are stored as unsigned values. In fact, although arrays
	// can have dynamic sizes, when their descriptor are instantiated they
	// already have all the sizes determined, and thus a descriptor will
	// never have a size with value -1.
	dimension_t sizes[Rank];
};

namespace impl
{
	template<typename T, unsigned int Rank>
	void printArrayDescriptor(stde::ostream& stream,
														const ArrayDescriptor<T, Rank>& descriptor,
														stde::Vector<typename ArrayDescriptor<T, Rank>::dimension_t>& indexes,
														typename ArrayDescriptor<T, Rank>::rank_t dimension)
	{
		using dimension_t = typename ArrayDescriptor<T, Rank>::dimension_t;

		//stream << "[";
		print_char("[");

		for (dimension_t i = 0, e = descriptor.getDimension(dimension); i < e; ++i) {
			indexes[dimension] = i;

			if (i > 0) {
        //stream << ", "
		print_char( ", ");
      }

			if (dimension == descriptor.getRank() - 1) {
        //stream << descriptor.get(indexes);
		print_integer(descriptor.get(indexes));
      } else {
        printArrayDescriptor(stream, descriptor, indexes, dimension + 1);
      }
		}

		indexes[dimension] = 0;
		//stream << "]";
		print_char("]");
	}
}

template<typename T, unsigned int Rank>
stde::ostream& operator<<(
    stde::ostream& stream, const ArrayDescriptor<T, Rank>& descriptor)
{
	using dimension_t = typename ArrayDescriptor<T, Rank>::dimension_t;
	stde::Vector<dimension_t> indexes(descriptor.getRank(), 0);
	impl::printArrayDescriptor(stream, descriptor, indexes, 0);
	return stream;
}

/// This class allows to accept a generically sized array as input argument
/// to a function.
///
/// @tparam T data type
template<typename T>
class UnsizedArrayDescriptor;

template<typename T>
stde::ostream& operator<<(
    stde::ostream& stream, const UnsizedArrayDescriptor<T>& descriptor);

template<typename T>
class UnsizedArrayDescriptor
{
	public:
	using iterator = typename ArrayDescriptor<T, 0>::iterator;
	using const_iterator = typename ArrayDescriptor<T, 0>::const_iterator;

	using rank_t = typename ArrayDescriptor<T, 0>::rank_t;
	using dimension_t = typename ArrayDescriptor<T, 0>::dimension_t;

	template<unsigned int Rank>
	UnsizedArrayDescriptor(ArrayDescriptor<T, Rank>& descriptor)
			: rank(descriptor.getRank()), descriptor((void*) &descriptor)
	{
	}

  void dump() const
  {
    getDescriptor()->dump();
  }

  void dump(stde::ostream& os) const
  {
    getDescriptor()->dump(os);
  }

  template<typename Index,typename stde::enable_if<stde::is_integral<Index>::value>::type* = nullptr>
  T& get(const Index& index)
  {
    return getDescriptor()->get(index);
  }

  template<typename Index,typename stde::enable_if<stde::is_integral<Index>::value>::type* = nullptr>
  const T& get(const Index& index) const
  {
    return getDescriptor()->get(index);
  }

  template<typename First, typename Second, typename... Others>
  T& get(const First& firstIndex, const Second& secondIndex, Others&&... otherIndexes)
  {
    return getDescriptor()->get(firstIndex, secondIndex, stde::forward<Others>(otherIndexes)...);
  }

  template<typename First, typename Second, typename... Others>
  const T& get(const First& firstIndex, const Second& secondIndex, Others&&... otherIndexes) const
  {
    return getDescriptor()->get(firstIndex, secondIndex, stde::forward<Others>(otherIndexes)...);
  }

  template<typename Index>
  T& get(std::initializer_list<Index> indexes)
  {
    return getDescriptor()->get(indexes);
  }

  template<typename Index>
  const T& get(std::initializer_list<Index> indexes) const
  {
    return getDescriptor()->get(indexes);
  }

  template<typename Indexes,typename stde::enable_if<!stde::is_integral<Indexes>::value>::type* = nullptr>
	T& get(const Indexes& indexes)
	{
		return getDescriptor()->get(indexes);
	}

  template<typename Indexes,typename stde::enable_if<!stde::is_integral<Indexes>::value>::type* = nullptr>
  const T& get(const Indexes& indexes) const
  {
    return getDescriptor()->get(indexes);
  }

  template<typename Index>
  void set(std::initializer_list<Index> indexes, T value)
  {
    getDescriptor()->set(indexes, value);
  }

  template<typename Indexes,typename stde::enable_if<!stde::is_integral<Indexes>::value>::type* = nullptr>
	void set(const Indexes& indexes, T value)
	{
		getDescriptor()->set(indexes, value);
	}

	T* getData() const
	{
		return getDescriptor()->getData();
	}

	rank_t getRank() const
	{
		stde::assertt(getDescriptor()->getRank() == rank);
		return rank;
	}

	dimension_t getDimensionSize(dimension_t index) const
	{
		return getDescriptor()->getDimension(index);
	}

	dimension_t getNumElements() const
	{
		return getDescriptor()->getNumElements();
	}

	iterator begin()
	{
		return getDescriptor()->begin();
	}

	const_iterator begin() const
	{
		return getDescriptor()->begin();
	}

	iterator end()
	{
		return getDescriptor()->end();
	}

	const_iterator end() const
	{
		return getDescriptor()->end();
	}

	bool hasSameSizes() const
	{
		return getDescriptor()->hasSameSizes();
	}

	friend stde::ostream& operator<< <>(
      stde::ostream& stream, const UnsizedArrayDescriptor<T>& descriptor);

	private:
	ArrayDescriptor<T, 0>* getDescriptor() const
	{
		stde::assertt(descriptor != nullptr);

		// In order to keep the iterator rank-agnostic we can cast the descriptor
		// to a 0-ranked one. This works only under assumption that the
		// descriptor uses pointer arithmetics to access its elements, and doesn't
		// rely on the provided rank.

		return (ArrayDescriptor<T, 0>*) descriptor;
	}

	rank_t rank;
	void* descriptor;
};

template<typename T>
stde::ostream& operator<<(
    stde::ostream& stream, const UnsizedArrayDescriptor<T>& descriptor)
{
	//return stream << *descriptor.getDescriptor();
	print_char(*descriptor.getDescriptor);
}

/// Iterate over all the elements of a multi-dimensional array as if it was
/// a flat one.
///
/// For example, an array declared as
///   v[3][2] = {{1, 2}, {3, 4}, {5, 6}}
/// would have its elements visited in the order
///   1, 2, 3, 4, 5, 6.
///
/// @tparam T 		data type
template<typename T>
class ArrayIterator
{
	public:
	using iterator_category = stde::forward_iterator_tag;
	using value_type = T;
	using difference_type = std::ptrdiff_t;
	using pointer = T*;
	using reference = T&;

	private:
	using ArrayType = typename stde::remove_const<T>::type;

	using rank_t = typename ArrayDescriptor<T, 0>::rank_t;
	using dimension_t = typename ArrayDescriptor<T, 0>::dimension_t;

	public:
	template<unsigned int Rank>
	ArrayIterator(ArrayDescriptor<ArrayType, Rank>& descriptor, bool finished = false)
			: finished(finished),
				offset(0),
				descriptor((ArrayDescriptor<ArrayType, 0>*) &descriptor)
	{
		for (dimension_t i = 0, rank = descriptor.getRank(); i < rank; ++i)
			indexes.push_back(0);
	}

	ArrayIterator<T>& operator++()
	{
		rank_t rank = descriptor->getRank();

		if (rank == 0) {
			finished = true;
			return *this;
		}

		rank_t consumedDimensions = 0;

		for (dimension_t i = 0; i < rank && indexes[rank - 1 - i] + 1 == descriptor->getDimension(rank - 1 - i); ++i) {
			indexes[rank - 1 - i] = 0;
			++consumedDimensions;
		}

		if (consumedDimensions == rank) {
			finished = true;
			offset = 0;
			return *this;
		}

		stde::assertt(consumedDimensions < rank);
		++indexes[rank - 1 - consumedDimensions];
		offset += 1;

		return *this;
	}

	T& operator*()
	{
		return (*descriptor)[offset];
	}

	T& operator*() const
	{
		return (*descriptor)[offset];
	}

	T* operator->()
	{
		return (*descriptor)[offset];
	}

	T* operator->() const
	{
		return (*descriptor)[offset];
	}

	bool operator==(const ArrayIterator& other) const
	{
		return other.finished == finished &&
					 other.offset == offset &&
					 other.descriptor == descriptor;
	}

	bool operator!=(const ArrayIterator& other) const
	{
		return other.finished != finished ||
					 other.offset != offset ||
					 other.descriptor != descriptor;
	}

	const stde::Vector<dimension_t>& getCurrentIndexes() const
	{
		return indexes;
	}

	private:
	bool finished = false;
	dimension_t offset = 0;
	stde::Vector<dimension_t> indexes;
	ArrayDescriptor<ArrayType, 0>* descriptor;
};
