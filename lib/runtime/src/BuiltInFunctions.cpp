#include <modelica/runtime/Runtime.h>

/**
 * Set all the elements of an array to a given value.
 *
 * @tparam T 		 data type
 * @param array  array to be populated
 * @param value  value to be set
 */
template<typename T>
inline void fill(UnsizedArrayDescriptor<T> array, T value)
{
	for (auto& element : array)
		element = value;
}

void _Mfill_ai1_i1(UnsizedArrayDescriptor<bool> array, bool value)
{
	fill(array, value);
}

void _Mfill_ai32_i32(UnsizedArrayDescriptor<int> array, int value)
{
	fill(array, value);
}

void _Mfill_ai64_i64(UnsizedArrayDescriptor<long> array, long value)
{
	fill(array, value);
}

void _Mfill_af32_f32(UnsizedArrayDescriptor<float> array, float value)
{
	fill(array, value);
}

void _Mfill_af64_f64(UnsizedArrayDescriptor<double> array, double value)
{
	fill(array, value);
}

void _mlir_ciface__Mfill_ai1_i1(UnsizedArrayDescriptor<bool> array, bool value)
{
	_Mfill_ai1_i1(array, value);
}

void _mlir_ciface__Mfill_ai32_i32(UnsizedArrayDescriptor<int> array, int value)
{
	_Mfill_ai32_i32(array, value);
}

void _mlir_ciface__Mfill_ai64_i64(UnsizedArrayDescriptor<long> array, long value)
{
	_Mfill_ai64_i64(array, value);
}

void _mlir_ciface__Mfill_af32_f32(UnsizedArrayDescriptor<float> array, float value)
{
	_Mfill_af32_f32(array, value);
}

void _mlir_ciface__Mfill_af64_f64(UnsizedArrayDescriptor<double> array, double value)
{
	_Mfill_af64_f64(array, value);
}

/**
 * Set a multi-dimensional array to an identity like matrix.
 *
 * @tparam T 	   data type
 * @param array  array to be populated
 */
template<typename T>
inline void identity(UnsizedArrayDescriptor<T> array)
{
	// Check that the array is square-like (all the dimensions have the same
	// size). Note that the implementation is generalized to n-D dimensions,
	// while the "identity" Modelica function is defined only for 2-D arrays.
	// Still, the implementation complexity would be the same.

	assert(array.hasSameSizes());

	// Directly use the iterators, as we need to determine the current indexes
	// so that we can place a 1 if the access is on the matrix diagonal.

	for (auto it = array.begin(), end = array.end(); it != end; ++it)
	{
		auto indexes = it.getCurrentIndexes();
		assert(!indexes.empty());

		bool isIdentityAccess = llvm::all_of(indexes, [&indexes](const auto& i) {
			return i == indexes[0];
		});

		*it = isIdentityAccess ? 1 : 0;
	}
}

void _Midentity_ai1(UnsizedArrayDescriptor<bool> array)
{
	identity(array);
}

void _Midentity_ai32(UnsizedArrayDescriptor<int> array)
{
	identity(array);
}

void _Midentity_ai64(UnsizedArrayDescriptor<long> array)
{
	identity(array);
}

void _Midentity_af32(UnsizedArrayDescriptor<float> array)
{
	identity(array);
}

void _Midentity_af64(UnsizedArrayDescriptor<double> array)
{
	identity(array);
}

void _mlir_ciface__Midentity_ai1(UnsizedArrayDescriptor<bool> array)
{
	_Midentity_ai1(array);
}

void _mlir_ciface__Midentity_ai32(UnsizedArrayDescriptor<int> array)
{
	_Midentity_ai32(array);
}

void _mlir_ciface__Midentity_ai64(UnsizedArrayDescriptor<long> array)
{
	_Midentity_ai64(array);
}

void _mlir_ciface__Midentity_af32(UnsizedArrayDescriptor<float> array)
{
	_Midentity_af32(array);
}

void _mlir_ciface__Midentity_af64(UnsizedArrayDescriptor<double> array)
{
	_Midentity_af64(array);
}
