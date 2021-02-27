#pragma once

namespace modelica
{
	template <typename T, int Rank>
	class ArrayDescriptorIterator;

	template<typename T, int Rank>
	struct ArrayDescriptor
	{
		using iterator = ArrayDescriptorIterator<T, Rank>;
		using const_iterator = ArrayDescriptorIterator<const T, Rank>;

		long* data;
		long rank;
		long sizes[3];

		iterator begin() { return { *this }; }
		const_iterator begin() const { return { *this }; }

		iterator end() { return { *this, -1 }; }
		const_iterator end() const { return { *this, -1 }; }
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

		//const std::array<long, Rank> &getIndices() { return indices; }

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