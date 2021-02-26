#pragma once

namespace modelica
{
	template <typename T, int Rank>
	class ArrayDescriptorIterator;

	template<typename T, long N>
	struct ArrayDescriptor
	{
		T* data;
		long rank;
		long sizes[N];

		template <typename Range,
				typename sfinae = decltype(std::declval<Range>().begin())>
		T& operator[](Range&& indexes) {
			assert(indexes.size() == N && "Indexes should match the rank of the array");
			long curOffset = 0;

			for (int dim = N - 1; dim >= 0; --dim) {
				long currentIndex = *(indexes.begin() + dim);
				assert(currentIndex < sizes[dim] && "Index overflow");
				curOffset += currentIndex;
			}

			return data[curOffset];
		}

		ArrayDescriptorIterator<T, N> begin() { return {*this}; }
		ArrayDescriptorIterator<T, N> end() { return {*this, -1}; }
	};

	template <typename T, int Rank>
	class ArrayDescriptorIterator {
		public:
		ArrayDescriptorIterator(ArrayDescriptor<T, Rank>& descriptor) : descriptor(descriptor)
		{
		}

		ArrayDescriptorIterator<T, Rank>& operator++()
		{
			int dim = Rank - 1;

			while (dim >= 0 && indices[dim] == (descriptor.sizes[dim] - 1)) {
				indices[dim] = 0;
				--dim;
			}

			if (dim < 0) {
				return *this;
			}

			++indices[dim];
			return *this;
		}

		T &operator*() { return descriptor.data[0]; }
		T *operator->() { return &descriptor.data[0]; }

		const std::array<int64_t, Rank> &getIndices() { return indices; }

		bool operator==(const ArrayDescriptorIterator& other) const {
			return &other.descriptor == &descriptor;
		}

		bool operator!=(const ArrayDescriptorIterator& other) const {
			return *this != other;
		}

		private:
		std::array<long, Rank> indices = {};
		ArrayDescriptor<T, Rank>& descriptor;
	};
}

