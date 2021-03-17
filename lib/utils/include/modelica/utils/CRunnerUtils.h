#pragma once

namespace modelica
{
	template <typename T, int Rank>
	class ArrayDescriptorIterator;

	template<typename T, int Rank>
	class ArrayDescriptor
	{
		friend ArrayDescriptorIterator<T, Rank>;

		public:
		using iterator = ArrayDescriptorIterator<T, Rank>;
		using const_iterator = ArrayDescriptorIterator<const T, Rank>;

		ArrayDescriptor(T* data, llvm::ArrayRef<long> sizes) : data(data), rank(Rank)
		{
			for (size_t i = 0, e = sizes.size(); i < e; ++i)
				this->sizes[i] = sizes[i];
		}

		// 1-D case
		T operator[](unsigned int index)
		{
			assert(Rank == 1);
			return get(index);
		}

		// [] operator can't be overloaded with multiple arguments, so we must
		// use a dedicated method to access multidimensional arrays.
		template<typename... Indexes>
		T get(Indexes... indexes)
		{
			llvm::SmallVector<long, 3> positions{ indexes... };
			assert(positions.size() == Rank && "Wrong amount of indexes");

			unsigned int resultIndex = positions[0];

			for (size_t i = 1; i < positions.size(); i++)
				resultIndex = resultIndex * sizes[i] + positions[i];

			return data[resultIndex];
		}

		long getRank()
		{
			return rank;
		}

		long getSize(size_t index)
		{
			return sizes[index];
		}

		iterator begin() { return { *this }; }
		const_iterator begin() const { return { *this }; }

		iterator end() { return { *this, -1 }; }
		const_iterator end() const { return { *this, -1 }; }

		private:
		T* data;
		long rank;
		long sizes[Rank];
	};

	struct UnrankedArrayDescriptor {
		long rank;
		void* descriptor;
	};

	template <typename T, int Rank>
	class ArrayDescriptorIterator {
		public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using pointer = T*;
		using reference = T&;

		ArrayDescriptorIterator(ArrayDescriptor<T, Rank>& descriptor, long offset = 0)
				: offset(offset),
					descriptor(&descriptor)
		{
		}

		ArrayDescriptorIterator<T, Rank>& operator++() {
			int dim = Rank - 1;

			while (dim >= 0 && indices[dim] == (descriptor->sizes[dim] - 1)) {
				offset -= indices[dim];
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
			return descriptor->data[offset];
		}

		T& operator*() const
		{
			return descriptor->data[offset];
		}

		T* operator->()
		{
			return &descriptor->data[offset];
		}

		T* operator->() const
		{
			return &descriptor->data[offset];
		}

		bool operator==(const ArrayDescriptorIterator &other) const {
			return other.offset == offset && other.descriptor == descriptor;
		}

		bool operator!=(const ArrayDescriptorIterator &other) const {
			return other.offset != offset || other.descriptor != descriptor;
		}

		private:
		long offset = 0;
		std::array<long, Rank> indices = {};
		ArrayDescriptor<T, Rank>* descriptor;
	};
}
