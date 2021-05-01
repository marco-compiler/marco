#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

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

	/**
	 * Utility constructor for tests.
	 *
	 * @param data  data pointer
	 * @param sizes sizes of the array
	 */
	ArrayDescriptor(T* data, std::array<long, Rank>& sizes)
			: data(data), rank(sizes.size()), sizes({})
	{
		assert(Rank == rank);

		for (auto size : llvm::enumerate(sizes))
			this->sizes[size.index()] = size.value();
	}

	template<typename... Index>
	T& get(Index... indexes)
	{
		llvm::SmallVector<long, 3> positions{ indexes... };
		return get(positions);
	}

	template<typename... Index>
	T& get(llvm::ArrayRef<long> indexes)
	{
		assert(indexes.size() == rank && "Wrong amount of indexes");
		assert(llvm::all_of(indexes, [](const auto& index) { return index >= 0; }));

		long resultIndex = indexes[0];

		for (size_t i = 1; i < indexes.size(); ++i)
		{
			long size = getDimensionSize(i);
			assert(size > 0);
			resultIndex = resultIndex * size + indexes[i];
		}

		assert(data != nullptr);
		return data[resultIndex];
	}

	[[nodiscard]] T* getData() const
	{
		return data;
	}

	[[nodiscard]] long getRank() const
	{
		return rank;
	}

	[[nodiscard]] long getDimensionSize(size_t index) const
	{
		assert(index >= 0 && index < rank);
		return sizes[index];
	}

	[[nodiscard]] iterator begin()
	{
		return ArrayIterator<T>(*this, 0);
	}

	[[nodiscard]] const_iterator begin() const
	{
		return ArrayIterator<T>(*this, 0);
	}

	[[nodiscard]] iterator end()
	{
		return ArrayIterator<T>(*this, -1);
	}

	[[nodiscard]] const_iterator end() const
	{
		return ArrayIterator<T>(*this, -1);
	}

	[[nodiscard]] bool hasSameSizes() const
	{
		for (size_t i = 0; i < rank; ++i)
			if (sizes[i] != sizes[0])
				return false;

		return true;
	}

	private:
	T* data;
	long rank;
	long sizes[Rank];
};

/**
 * This class allows to accept a generically sized array as input argument
 * to a function.
 *
 * @tparam T data type
 */
template<typename T>
class UnsizedArrayDescriptor
{
	public:
	using iterator = typename ArrayDescriptor<T, 0>::iterator;
	using const_iterator = typename ArrayDescriptor<T, 0>::const_iterator;

	template<unsigned int Rank>
	UnsizedArrayDescriptor(ArrayDescriptor<T, Rank>& descriptor)
			: rank(descriptor.getRank()), descriptor((void*) &descriptor)
	{
	}

	[[nodiscard]] T* getData() const
	{
		assert(descriptor != nullptr);
		return getDescriptor()->getData();
	}

	[[nodiscard]] long getRank() const
	{
		assert(descriptor != nullptr);
		assert(getDescriptor()->getRank() == rank);
		return rank;
	}

	template<typename... Index>
	T& get(Index... indexes)
	{
		assert(descriptor != nullptr);
		return getDescriptor()->get(indexes...);
	}

	[[nodiscard]] long getDimensionSize(size_t index) const
	{
		assert(descriptor != nullptr);
		return getDescriptor()->getDimensionSize(index);
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

	private:
	[[nodiscard]] ArrayDescriptor<T, 0>* getDescriptor() const
	{
		// In order to keep the iterator rank-agnostic we can cast the descriptor
		// to a 0-ranked one. This works only under assumption that the
		// descriptor uses pointer arithmetics to access its elements, and doesn't
		// rely on the provided rank.

		return (ArrayDescriptor<T, 0>*) descriptor;
	}

	long rank;
	void* descriptor;
};

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

	template<unsigned int Rank>
	ArrayIterator(ArrayDescriptor<T, Rank>& descriptor, long offset = 0)
			: offset(offset),
				descriptor((ArrayDescriptor<T, 0>*) &descriptor)
	{
		for (long i = 0, rank = descriptor.getRank(); i < rank; ++i)
			indexes.push_back(0);
	}

	ArrayIterator<T>& operator++()
	{
		int dim = descriptor->getRank() - 1;

		while (dim >= 0 && indexes[dim] == (descriptor->getDimensionSize(dim) - 1)) {
			indexes[dim] = 0;
			--dim;
		}

		if (dim < 0) {
			offset = -1;
			return *this;
		}

		++indexes[dim];
		offset += 1;

		return *this;
	}

	T& operator*()
	{
		return descriptor->getData()[offset];
	}

	T& operator*() const
	{
		return descriptor->getData()[offset];
	}

	T* operator->()
	{
		return &descriptor->getData()[offset];
	}

	T* operator->() const
	{
		return &descriptor->getData()[offset];
	}

	bool operator==(const ArrayIterator &other) const
	{
		return other.offset == offset && other.descriptor == descriptor;
	}

	bool operator!=(const ArrayIterator &other) const
	{
		return other.offset != offset || other.descriptor != descriptor;
	}

	llvm::ArrayRef<long> getCurrentIndexes() const
	{
		return indexes;
	}

	private:
	long offset = 0;
	llvm::SmallVector<long, 3> indexes;
	const ArrayDescriptor<T, 0>* descriptor;
};

extern "C"
{
	void modelicaPrint(char* name, float value);

	void modelicaPrintFVector(char* name, float* value, int count);
	void modelicaPrintBVector(char* name, char* value, int count);
	void modelicaPrintIVector(char* name, int* value, int count);

	void fill(float* out, long* outDim, float* filler, long* dim);

	float modelicaPow(float b, float exp);
	double modelicaPowD(double b, double exp);

	void printString(char* str);
	void printI1(bool value);
	void printI32(int value);
	void printI64(long value);
	void printF32(float value);
	void printF64(double value);

	[[maybe_unused]] void _Mfill_ai1_i1(UnsizedArrayDescriptor<bool> array, bool value);
	[[maybe_unused]] void _Mfill_ai32_i32(UnsizedArrayDescriptor<int> array, int value);
	[[maybe_unused]] void _Mfill_ai64_i64(UnsizedArrayDescriptor<long> array, long value);
	[[maybe_unused]] void _Mfill_af32_f32(UnsizedArrayDescriptor<float> array, float value);
	[[maybe_unused]] void _Mfill_af64_f64(UnsizedArrayDescriptor<double> array, double value);

	[[maybe_unused]] void _mlir_ciface__Mfill_ai1_i1(UnsizedArrayDescriptor<bool> array, bool value);
	[[maybe_unused]] void _mlir_ciface__Mfill_ai32_i32(UnsizedArrayDescriptor<int> array, int value);
	[[maybe_unused]] void _mlir_ciface__Mfill_ai64_i64(UnsizedArrayDescriptor<long> array, long value);
	[[maybe_unused]] void _mlir_ciface__Mfill_af32_f32(UnsizedArrayDescriptor<float> array, float value);
	[[maybe_unused]] void _mlir_ciface__Mfill_af64_f64(UnsizedArrayDescriptor<double> array, double value);

	[[maybe_unused]] void _Midentity_ai1(UnsizedArrayDescriptor<bool> array);
	[[maybe_unused]] void _Midentity_ai32(UnsizedArrayDescriptor<int> array);
	[[maybe_unused]] void _Midentity_ai64(UnsizedArrayDescriptor<long> array);
	[[maybe_unused]] void _Midentity_af32(UnsizedArrayDescriptor<float> array);
	[[maybe_unused]] void _Midentity_af64(UnsizedArrayDescriptor<double> array);

	[[maybe_unused]] void _mlir_ciface__Midentity_ai1(UnsizedArrayDescriptor<bool> array);
	[[maybe_unused]] void _mlir_ciface__Midentity_ai32(UnsizedArrayDescriptor<int> array);
	[[maybe_unused]] void _mlir_ciface__Midentity_ai64(UnsizedArrayDescriptor<long> array);
	[[maybe_unused]] void _mlir_ciface__Midentity_af32(UnsizedArrayDescriptor<float> array);
	[[maybe_unused]] void _mlir_ciface__Midentity_af64(UnsizedArrayDescriptor<double> array);
}
