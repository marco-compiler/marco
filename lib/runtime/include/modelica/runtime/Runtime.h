#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <iostream>

template <typename T, unsigned int Rank>
class ArrayIterator;

template<typename T, unsigned int Rank>
struct ArrayDescriptor
{
	public:
	using iterator = ArrayIterator<T, Rank>;
	using const_iterator = ArrayIterator<const T, Rank>;

	ArrayDescriptor(T* data, std::array<long, Rank>& sizes)
			: data(data), rank(sizes.size())
	{
		for (size_t i = 0, e = sizes.size(); i < e; ++i)
			this->sizes[i] = sizes[i];
	}

	template<typename... Index>
	T& get(Index... indexes)
	{
		llvm::SmallVector<long, 3> positions{ indexes... };
		assert(positions.size() == rank && "Wrong amount of indexes");
		assert(llvm::all_of(positions, [](const auto& index) { return index >= 0; }));

		long resultIndex = positions[0];

		for (size_t i = 1; i < positions.size(); ++i)
		{
			long size = getDimensionSize(i);
			assert(size > 0);
			resultIndex = resultIndex * size + positions[i];
		}

		assert(data != nullptr);
		return data[resultIndex];
	}

	T* getData()
	{
		return data;
	}

	long getRank()
	{
		return rank;
	}

	long getDimensionSize(size_t index)
	{
		assert(index < rank);
		return sizes[index];
	}

	iterator begin()
	{
		return ArrayIterator<T, Rank>(*this, 0);
	}

	const_iterator begin() const
	{
		return ArrayIterator<T, Rank>(*this, 0);
	}

	iterator end()
	{
		return ArrayIterator<T, Rank>(*this, -1);
	}

	const_iterator end() const
	{
		return ArrayIterator<T, Rank>(*this, -1);
	}

	private:
	T* data;
	long rank;
	long sizes[Rank];
};

template<typename T>
struct UnsizedArrayDescriptor
{
	long rank;
	void* descriptor;

	using iterator = typename ArrayDescriptor<T, 0>::iterator;
	using const_iterator = typename ArrayDescriptor<T, 0>::const_iterator;

	template<unsigned int Rank>
	UnsizedArrayDescriptor(ArrayDescriptor<T, Rank>& descriptor)
			: rank(descriptor.getRank()), descriptor((void*) &descriptor)
	{
	}

	template<typename... Index>
	T& get(Index... indexes)
	{
		assert(descriptor != nullptr);
		return getDescriptor()->get(indexes...);
	}

	long getDimensionSize(size_t index)
	{
		assert(descriptor != nullptr);
		return getDescriptor()->getDimensionSize(index);
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

	ArrayDescriptor<T, 0>* getDescriptor()
	{
		return (ArrayDescriptor<T, 0>*) descriptor;
	}
};

template <typename T>
class DynamicArrayDescriptor {
	public:
	template<unsigned int Rank>
	explicit DynamicArrayDescriptor(const ArrayDescriptor<T, Rank>& descriptor)
			: rank(descriptor.rank),
				data(descriptor.data),
				sizes(descriptor.sizes)
	{
	}

	explicit DynamicArrayDescriptor(const UnsizedArrayDescriptor<T>& descriptor)
			: rank(descriptor.rank),
				data(descriptor.getDescriptor()->data),
				sizes(rank == 0 ? nullptr : descriptor.getDescriptor()->sizes)
	{
	}

	unsigned int rank;
	T* data;
	const long* sizes;
};

template<typename T, unsigned int Rank>
class ArrayIterator
{
	public:
	using iterator_category = std::forward_iterator_tag;
	using value_type = T;
	using difference_type = std::ptrdiff_t;
	using pointer = T*;
	using reference = T&;

	ArrayIterator(ArrayDescriptor<T, Rank>& descriptor, long offset = 0)
			: offset(offset),
				descriptor(&descriptor)
	{
		for (long i = 0; i < descriptor.getRank(); ++i)
			indices.push_back(0);
	}

	ArrayIterator<T, Rank>& operator++() {
		int dim = descriptor->getRank() - 1;

		while (dim >= 0 && indices[dim] == (descriptor->getDimensionSize(dim) - 1)) {
			indices[dim] = 0;
			--dim;
		}

		if (dim < 0) {
			offset = -1;
			return *this;
		}

		++indices[dim];
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

	bool operator==(const ArrayIterator &other) const {
		return other.offset == offset && other.descriptor == descriptor;
	}

	bool operator!=(const ArrayIterator &other) const {
		return other.offset != offset || other.descriptor != descriptor;
	}

	private:
	long offset = 0;
	llvm::SmallVector<long, 3> indices;
	ArrayDescriptor<T, Rank>* descriptor;
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

	void _Mfill_i1(UnsizedArrayDescriptor<bool> descriptor, bool value);
	void _Mfill_i32(UnsizedArrayDescriptor<int> descriptor, int value);
	void _Mfill_i64(UnsizedArrayDescriptor<long> descriptor, long value);
	void _Mfill_f32(UnsizedArrayDescriptor<float> descriptor, float value);
	void _Mfill_f64(UnsizedArrayDescriptor<double> descriptor, double value);
}
