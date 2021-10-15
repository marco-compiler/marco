#pragma once

#include <iostream>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/raw_os_ostream.h>

template <typename T>
class ArrayIterator;

/**
 * This class represents how the arrays are described inside the Modelica IR.
 * Note that the Rank template argument is used only to allocate an
 * appropriately sized array of sizes. It is not used during the elements
 * access, where pointer arithmetics are used instead; this last point
 * allows to instantiate iterators on descriptor whose rank is not known
 * at compile time, and thus allows to iterate over descriptors received
 * from the IR calls.
 *
 * @tparam T 	  data type
 * @tparam Rank number of dimensions
 */
template<typename T, unsigned int Rank>
class ArrayDescriptor
{
	public:
	using iterator = ArrayIterator<T>;
	using const_iterator = ArrayIterator<const T>;

	using rank_t = uint64_t;
	using dimension_t = uint64_t;

	/**
	 * Utility constructor for tests.
	 *
	 * @param data  data pointer
	 * @param sizes sizes of the array
	 */
	template<unsigned long Size>
	ArrayDescriptor(std::array<T, Size>& data)
			: data(data.data()), rank(1), sizes{}
	{
		assert(Rank == rank);
		this->sizes[0] = Size;
	}

	/**
	 * Utility constructor for tests.
	 *
	 * @param data  data pointer
	 * @param sizes sizes of the array
	 */
	ArrayDescriptor(T* data, llvm::ArrayRef<dimension_t> sizes)
			: data(data), rank(sizes.size()), sizes{}
	{
		assert(Rank == rank);

		for (auto size : llvm::enumerate(sizes))
			this->sizes[size.index()] = size.value();
	}

	/**
	 * Get element at offset.
	 *
	 * @param offset	 index of the flat array
	 * @return value
	 */
	T& operator[](dimension_t offset)
	{
		assert(data != nullptr);
		return data[offset];
	}

	/**
	 * Get element at offset.
	 *
	 * @param offset	 index of the flat array
	 * @return value
	 */
	const T& operator[](dimension_t offset) const
	{
		assert(data != nullptr);
		return data[offset];
	}

	void dump() const
	{
		dump(llvm::outs());
	}

	void dump(llvm::raw_ostream& os) const
	{
		os << "Array descriptor";
		os.indent(2) << "address: " << getData();
		os.indent(2) << "rank: " << getRank();
		os.indent(2) << "values: " << *this;
		os << "\n";
	}

	template<typename... Index>
	T& get(Index... indexes)
	{
		llvm::SmallVector<dimension_t, 3> positions{ static_cast<dimension_t>(indexes)... };
		return get(static_cast<llvm::ArrayRef<dimension_t>>(positions));
	}

	T& get(llvm::ArrayRef<dimension_t> indexes)
	{
		assert(data != nullptr);
		dimension_t offset = computeOffset(indexes);
		return (*this)[offset];
	}

	const T& get(llvm::ArrayRef<dimension_t> indexes) const
	{
		assert(data != nullptr);
		dimension_t offset = computeOffset(indexes);
		return (*this)[offset];
	}

	void set(llvm::ArrayRef<dimension_t> indexes, T value)
	{
		assert(data != nullptr);
		dimension_t offset = computeOffset(indexes);
		(*this)[offset] = value;
	}

	[[nodiscard]] bool hasData(llvm::ArrayRef<unsigned long> indexes) const
	{
		if (indexes.size() != rank)
			return false;

		for (unsigned long i = 0; i < indexes.size(); ++i)
			if (indexes[i] >= getDimension(i))
				return false;

		return true;
	}

	[[nodiscard]] T* getData() const
	{
		return data;
	}

	[[nodiscard]] rank_t getRank() const
	{
		return rank;
	}

	[[nodiscard]] dimension_t getDimension(rank_t index) const
	{
		assert(index >= 0 && index < rank);
		return sizes[index];
	}

	[[nodiscard]] llvm::ArrayRef<dimension_t> getDimensions() const
	{
		return llvm::ArrayRef<dimension_t>(sizes, rank);
	}

	[[nodiscard]] dimension_t getNumElements() const
	{
		dimension_t result = 1;

		for (const auto& dimension : getDimensions())
			result *= dimension;

		return result;
	}

	[[nodiscard]] iterator begin()
	{
		return iterator(*this, false);
	}

	[[nodiscard]] const_iterator begin() const
	{
		return const_iterator(*this, false);
	}

	[[nodiscard]] iterator end()
	{
		return iterator(*this, true);
	}

	[[nodiscard]] const_iterator end() const
	{
		return const_iterator(*this, true);
	}

	[[nodiscard]] bool hasSameSizes() const
	{
		for (rank_t i = 1; i < rank; ++i)
			if (sizes[i] != sizes[0])
				return false;

		return true;
	}

	private:
	[[nodiscard]] dimension_t computeOffset(llvm::ArrayRef<dimension_t> indexes) const
	{
		assert(indexes.size() == rank && "Wrong amount of indexes");
		assert(llvm::all_of(indexes, [](const auto& index) { return index >= 0; }));

		dimension_t offset = indexes[0];

		for (size_t i = 1; i < indexes.size(); ++i)
		{
			dimension_t size = getDimension(i);
			assert(size > 0);
			offset = offset * size + indexes[i];
		}

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
	void printArrayDescriptor(llvm::raw_ostream& stream,
														const ArrayDescriptor<T, Rank>& descriptor,
														llvm::SmallVectorImpl<typename ArrayDescriptor<T, Rank>::dimension_t>& indexes,
														typename ArrayDescriptor<T, Rank>::rank_t dimension)
	{
		using dimension_t = typename ArrayDescriptor<T, Rank>::dimension_t;

		stream << "[";

		for (dimension_t i = 0, e = descriptor.getDimension(dimension); i < e; ++i)
		{
			indexes[dimension] = i;

			if (i > 0)
				stream << ", ";

			if (dimension == descriptor.getRank() - 1)
				stream << descriptor.get(indexes);
			else
				printArrayDescriptor(stream, descriptor, indexes, dimension + 1);
		}

		indexes[dimension] = 0;
		stream << "]";
	}
}

template<typename T, unsigned int Rank>
llvm::raw_ostream& operator<<(
		llvm::raw_ostream& stream, const ArrayDescriptor<T, Rank>& descriptor)
{
	using dimension_t = typename ArrayDescriptor<T, Rank>::dimension_t;
	llvm::SmallVector<dimension_t, Rank> indexes(descriptor.getRank(), 0);
	impl::printArrayDescriptor(stream, descriptor, indexes, 0);
	return stream;
}

template<typename T, unsigned int Rank>
std::ostream& operator<<(
		std::ostream& stream, const ArrayDescriptor<T, Rank>& descriptor)
{
	llvm::raw_os_ostream(stream) << descriptor;
	return stream;
}

/**
 * This class allows to accept a generically sized array as input argument
 * to a function.
 *
 * @tparam T data type
 */
template<typename T>
class UnsizedArrayDescriptor;

template<typename T>
llvm::raw_ostream& operator<<(
		llvm::raw_ostream& stream, const UnsizedArrayDescriptor<T>& descriptor);

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

	T& operator[](dimension_t offset)
	{
		return getDescriptor()->operator[](offset);
	}

	const T& operator[](dimension_t offset) const
	{
		return getDescriptor()->operator[](offset);
	}

	template<typename... Index>
	T& get(Index... indexes)
	{
		assert(descriptor != nullptr);
		return getDescriptor()->get(indexes...);
	}

	T& get(llvm::ArrayRef<dimension_t> indexes)
	{
		return getDescriptor()->get(indexes);
	}

	void set(llvm::ArrayRef<dimension_t> indexes, T value)
	{
		getDescriptor()->set(indexes, value);
	}

	[[nodiscard]] bool hasData(llvm::ArrayRef<unsigned long> indexes) const
	{
		return getDescriptor()->hasData(indexes);
	}

	[[nodiscard]] T* getData() const
	{
		return getDescriptor()->getData();
	}

	[[nodiscard]] rank_t getRank() const
	{
		assert(getDescriptor()->getRank() == rank);
		return rank;
	}

	[[nodiscard]] dimension_t getDimensionSize(dimension_t index) const
	{
		return getDescriptor()->getDimension(index);
	}

	[[nodiscard]] llvm::ArrayRef<dimension_t> getDimensions() const
	{
		return getDescriptor()->getDimensions();
	}

	[[nodiscard]] dimension_t getNumElements() const
	{
		return getDescriptor()->getNumElements();
	}

	[[nodiscard]] iterator begin()
	{
		return getDescriptor()->begin();
	}

	[[nodiscard]] const_iterator begin() const
	{
		return getDescriptor()->begin();
	}

	[[nodiscard]] iterator end()
	{
		return getDescriptor()->end();
	}

	[[nodiscard]] const_iterator end() const
	{
		return getDescriptor()->end();
	}

	[[nodiscard]] bool hasSameSizes() const
	{
		return getDescriptor()->hasSameSizes();
	}

	friend llvm::raw_ostream& operator<< <>(
			llvm::raw_ostream& stream, const UnsizedArrayDescriptor<T>& descriptor);

	private:
	[[nodiscard]] ArrayDescriptor<T, 0>* getDescriptor() const
	{
		assert(descriptor != nullptr);

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
llvm::raw_ostream& operator<<(
		llvm::raw_ostream& stream, const UnsizedArrayDescriptor<T>& descriptor)
{
	return stream << *descriptor.getDescriptor();
}

template<typename T>
std::ostream& operator<<(
		std::ostream& stream, const UnsizedArrayDescriptor<T>& descriptor)
{
	llvm::raw_os_ostream(stream) << descriptor;
	return stream;
}

/**
 * Iterate over all the elements of a multi-dimensional array as if it was
 * a flat one.
 *
 * For example, an array declared as
 *   v[3][2] = {{1, 2}, {3, 4}, {5, 6}}
 * would have its elements visited in the order
 *   1, 2, 3, 4, 5, 6.
 *
 * @tparam T 		data type
 */
template<typename T>
class ArrayIterator
{
	public:
	using iterator_category = std::forward_iterator_tag;
	using value_type = T;
	using difference_type = std::ptrdiff_t;
	using pointer = T*;
	using reference = T&;

	private:
	using ArrayType = typename std::remove_const<T>::type;

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

		if (rank == 0)
		{
			finished = true;
			return *this;
		}

		rank_t consumedDimensions = 0;

		for (dimension_t i = 0; i < rank && indexes[rank - 1 - i] + 1 == descriptor->getDimension(rank - 1 - i); ++i)
		{
			indexes[rank - 1 - i] = 0;
			++consumedDimensions;
		}

		if (consumedDimensions == rank)
		{
			finished = true;
			offset = 0;
			return *this;
		}

		assert(consumedDimensions < rank);
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

	[[nodiscard]] llvm::ArrayRef<dimension_t> getCurrentIndexes() const
	{
		return indexes;
	}

	private:
	bool finished = false;
	dimension_t offset = 0;
	llvm::SmallVector<dimension_t, 3> indexes;
	ArrayDescriptor<ArrayType, 0>* descriptor;
};
