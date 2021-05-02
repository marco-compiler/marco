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

RUNTIME_FUNC_DEF(fill, void, array(bool), bool);
RUNTIME_FUNC_DEF(fill, void, array(int), int);
RUNTIME_FUNC_DEF(fill, void, array(long), long);
RUNTIME_FUNC_DEF(fill, void, array(float), float);
RUNTIME_FUNC_DEF(fill, void, array(double), double);

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

RUNTIME_FUNC_DEF(identity, void, array(bool))
RUNTIME_FUNC_DEF(identity, void, array(int))
RUNTIME_FUNC_DEF(identity, void, array(long))
RUNTIME_FUNC_DEF(identity, void, array(float))
RUNTIME_FUNC_DEF(identity, void, array(double))

/**
 * Place some values on the diagonal of a matrix, and set all the other
 * elements to zero.
 *
 * @tparam T 					destination matrix type
 * @tparam U 					source values type
 * @param destination destination matrix
 * @param values 			source values
 */
template<typename T, typename U>
void diagonal(UnsizedArrayDescriptor<T> destination, UnsizedArrayDescriptor<U> values)
{
	// Check that the array is square-like (all the dimensions have the same
	// size). Note that the implementation is generalized to n-D dimensions,
	// while the "identity" Modelica function is defined only for 2-D arrays.
	// Still, the implementation complexity would be the same.

	assert(destination.hasSameSizes());

	// Check that the sizes of the matrix dimensions match with the amount of
	// values to be set.

	assert(destination.getRank() > 0);
	assert(values.getRank() == 1);
	assert(destination.getDimensionSize(0) == values.getDimensionSize(0));

	// Directly use the iterators, as we need to determine the current indexes
	// so that we can place a 1 if the access is on the matrix diagonal.

	for (auto it = destination.begin(), end = destination.end(); it != end; ++it)
	{
		auto indexes = it.getCurrentIndexes();
		assert(!indexes.empty());

		bool isIdentityAccess = llvm::all_of(indexes, [&indexes](const auto& i) {
			return i == indexes[0];
		});

		*it = isIdentityAccess ? values.get(indexes[0]) : 0;
	}
}

RUNTIME_FUNC_DEF(diagonal, void, array(bool), array(bool))
RUNTIME_FUNC_DEF(diagonal, void, array(bool), array(int))
RUNTIME_FUNC_DEF(diagonal, void, array(bool), array(long))
RUNTIME_FUNC_DEF(diagonal, void, array(bool), array(float))
RUNTIME_FUNC_DEF(diagonal, void, array(bool), array(double))

RUNTIME_FUNC_DEF(diagonal, void, array(int), array(bool))
RUNTIME_FUNC_DEF(diagonal, void, array(int), array(int))
RUNTIME_FUNC_DEF(diagonal, void, array(int), array(long))
RUNTIME_FUNC_DEF(diagonal, void, array(int), array(float))
RUNTIME_FUNC_DEF(diagonal, void, array(int), array(double))

RUNTIME_FUNC_DEF(diagonal, void, array(long), array(bool))
RUNTIME_FUNC_DEF(diagonal, void, array(long), array(int))
RUNTIME_FUNC_DEF(diagonal, void, array(long), array(long))
RUNTIME_FUNC_DEF(diagonal, void, array(long), array(float))
RUNTIME_FUNC_DEF(diagonal, void, array(long), array(double))

RUNTIME_FUNC_DEF(diagonal, void, array(float), array(bool))
RUNTIME_FUNC_DEF(diagonal, void, array(float), array(int))
RUNTIME_FUNC_DEF(diagonal, void, array(float), array(long))
RUNTIME_FUNC_DEF(diagonal, void, array(float), array(float))
RUNTIME_FUNC_DEF(diagonal, void, array(float), array(double))

RUNTIME_FUNC_DEF(diagonal, void, array(double), array(bool))
RUNTIME_FUNC_DEF(diagonal, void, array(double), array(int))
RUNTIME_FUNC_DEF(diagonal, void, array(double), array(long))
RUNTIME_FUNC_DEF(diagonal, void, array(double), array(float))
RUNTIME_FUNC_DEF(diagonal, void, array(double), array(double))

/**
 * Populate a 1-D array with equally spaced elements.
 *
 * @tparam T 		 data type
 * @param array  array to be populated
 * @param start  start value
 * @param end 	 end value
 */
template<typename T>
void linspace(UnsizedArrayDescriptor<T> array, double start, double end)
{
	assert(array.getRank() == 1);

	size_t n = array.getDimensionSize(0);
	double step = (end - start) / ((double) n - 1);

	for (size_t i = 0; i < n; ++i)
		array.get(i) = start + i * step;
}

RUNTIME_FUNC_DEF(linspace, void, array(bool), float, float)
RUNTIME_FUNC_DEF(linspace, void, array(bool), double, double)
RUNTIME_FUNC_DEF(linspace, void, array(int), float, float)
RUNTIME_FUNC_DEF(linspace, void, array(int), double, double)
RUNTIME_FUNC_DEF(linspace, void, array(long), float, float)
RUNTIME_FUNC_DEF(linspace, void, array(long), double, double)
RUNTIME_FUNC_DEF(linspace, void, array(float), float, float)
RUNTIME_FUNC_DEF(linspace, void, array(float), double, double)
RUNTIME_FUNC_DEF(linspace, void, array(double), float, float)
RUNTIME_FUNC_DEF(linspace, void, array(double), double, double)
