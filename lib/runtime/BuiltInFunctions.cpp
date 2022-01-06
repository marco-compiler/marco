#include <cmath>
#include <marco/runtime/BuiltInFunctions.h>
#include <numeric>

/**
 * Get the absolute value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return absolute value
 */
template<typename T>
inline T abs(T value)
{
	if (value >= 0)
		return value;

	return -1 * value;
}

inline bool abs(bool value)
{
	return value;
}

RUNTIME_FUNC_DEF(abs, bool, bool)
RUNTIME_FUNC_DEF(abs, int32_t, int32_t)
RUNTIME_FUNC_DEF(abs, int64_t, int64_t)
RUNTIME_FUNC_DEF(abs, float, float)
RUNTIME_FUNC_DEF(abs, double, double)

/**
 * Get the inverse cosine of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return acos
 */
template<typename T>
inline double acos(T value)
{
	return std::acos(value);
}

RUNTIME_FUNC_DEF(acos, float, float)
RUNTIME_FUNC_DEF(acos, double, double)

/**
 * Get the inverse sine of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return asin
 */
template<typename T>
inline double asin(T value)
{
	return std::asin(value);
}

RUNTIME_FUNC_DEF(asin, float, float)
RUNTIME_FUNC_DEF(asin, double, double)

/**
 * Get the inverse tangent of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return atan
 */
template<typename T>
inline double atan(T value)
{
	return std::atan(value);
}

RUNTIME_FUNC_DEF(atan, float, float)
RUNTIME_FUNC_DEF(atan, double, double)

/**
 * Get the inverse tangent of a value.
 *
 * @tparam T	data type
 * @param y		y value
 * @param x		x value
 * @return atan2
 */
template<typename T>
inline double atan2(T y, T x)
{
	return std::atan2(y, x);
}

RUNTIME_FUNC_DEF(atan2, float, float, float)
RUNTIME_FUNC_DEF(atan2, double, double, double)

/**
 * Get the cosine of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return cos
 */
template<typename T>
inline double cos(T value)
{
	return std::cos(value);
}

RUNTIME_FUNC_DEF(cos, float, float)
RUNTIME_FUNC_DEF(cos, double, double)

/**
 * Get the hyperbolic cosine of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return cosh
 */
template<typename T>
inline double cosh(T value)
{
	return std::cosh(value);
}

RUNTIME_FUNC_DEF(cosh, float, float)
RUNTIME_FUNC_DEF(cosh, double, double)

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
inline void diagonal(UnsizedArrayDescriptor<T> destination, UnsizedArrayDescriptor<U> values)
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

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(double))

/**
 * Get the value e at the power of a value.
 *
 * @tparam T		 		data type
 * @param exponent	exponent
 * @return exp
 */
template<typename T>
inline double exp(T value)
{
	return std::exp(value);
}

RUNTIME_FUNC_DEF(exp, float, float)
RUNTIME_FUNC_DEF(exp, double, double)

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

RUNTIME_FUNC_DEF(fill, void, ARRAY(bool), bool)
RUNTIME_FUNC_DEF(fill, void, ARRAY(int32_t), int32_t)
RUNTIME_FUNC_DEF(fill, void, ARRAY(int64_t), int64_t)
RUNTIME_FUNC_DEF(fill, void, ARRAY(float), float)
RUNTIME_FUNC_DEF(fill, void, ARRAY(double), double)

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

RUNTIME_FUNC_DEF(identity, void, ARRAY(bool))
RUNTIME_FUNC_DEF(identity, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(identity, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(identity, void, ARRAY(float))
RUNTIME_FUNC_DEF(identity, void, ARRAY(double))

/**
 * Populate a 1-D array with equally spaced elements.
 *
 * @tparam T 		 data type
 * @param array  array to be populated
 * @param start  start value
 * @param end 	 end value
 */
template<typename T>
inline void linspace(UnsizedArrayDescriptor<T> array, double start, double end)
{
	using dimension_t = typename UnsizedArrayDescriptor<T>::dimension_t;
	assert(array.getRank() == 1);

	auto n = array.getDimensionSize(0);
	double step = (end - start) / ((double) n - 1);

	for (dimension_t i = 0; i < n; ++i)
		array.get(i) = start + static_cast<double>(i) * step;
}

RUNTIME_FUNC_DEF(linspace, void, ARRAY(bool), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(bool), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int32_t), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int32_t), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int64_t), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int64_t), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(float), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(float), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(double), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(double), double, double)

/**
 * Get the natural logarithm of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return log
 */
template<typename T>
inline double log(T value)
{
	assert(value > 0);
	return std::log(value);
}

RUNTIME_FUNC_DEF(log, float, float)
RUNTIME_FUNC_DEF(log, double, double)

/**
 * Get the base 10 logarithm of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return log10
 */
template<typename T>
inline double log10(T value)
{
	assert(value > 0);
	return std::log10(value);
}

RUNTIME_FUNC_DEF(log10, float, float)
RUNTIME_FUNC_DEF(log10, double, double)

/**
 * Get the max value of an array.
 *
 * @tparam T 		 data type
 * @param array	 array
 * @return maximum value
 */
template<typename T>
inline T max(UnsizedArrayDescriptor<T> array)
{
	return *std::max_element(array.begin(), array.end());
}

RUNTIME_FUNC_DEF(max, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(max, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(max, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(max, float, ARRAY(float))
RUNTIME_FUNC_DEF(max, double, ARRAY(double))

/**
 * Get the max among two values.
 *
 * @tparam T data type
 * @param x  left-hand side value
 * @param y  right-hand side value
 * @return maximum value
 */
template<typename T>
inline T max(T x, T y)
{
	return std::max(x, y);
}

RUNTIME_FUNC_DEF(max, bool, bool, bool)
RUNTIME_FUNC_DEF(max, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(max, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(max, float, float, float)
RUNTIME_FUNC_DEF(max, double, double, double)

/**
 * Get the min value of an array.
 *
 * @tparam T 		 data type
 * @param array	 array
 * @return minimum value
 */
template<typename T>
inline T min(UnsizedArrayDescriptor<T> array)
{
	return *std::min_element(array.begin(), array.end());
}

RUNTIME_FUNC_DEF(min, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(min, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(min, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(min, float, ARRAY(float))
RUNTIME_FUNC_DEF(min, double, ARRAY(double))

/**
 * Get the min among two values.
 *
 * @tparam T  data type
 * @param x   left-hand side value
 * @param y   right-hand side value
 * @return minimum value
 */
template<typename T>
inline T min(T x, T y)
{
	return std::min(x, y);
}

RUNTIME_FUNC_DEF(min, bool, bool, bool)
RUNTIME_FUNC_DEF(min, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(min, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(min, float, float, float)
RUNTIME_FUNC_DEF(min, double, double, double)

/**
 * Set all the elements of an array to ones.
 *
 * @tparam T data type
 * @param array   array to be populated
 */
template<typename T>
inline void ones(UnsizedArrayDescriptor<T> array)
{
	for (auto& element : array)
		element = 1;
}

RUNTIME_FUNC_DEF(ones, void, ARRAY(bool))
RUNTIME_FUNC_DEF(ones, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(ones, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(ones, void, ARRAY(float))
RUNTIME_FUNC_DEF(ones, void, ARRAY(double))

/**
 * Multiply all the elements of an array.
 *
 * @tparam T 		 data type
 * @param array  array
 * @return product of all the values
 */
template<typename T>
inline T product(UnsizedArrayDescriptor<T> array)
{
	return std::accumulate(array.begin(), array.end(), 1, std::multiplies<T>());
}

RUNTIME_FUNC_DEF(product, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(product, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(product, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(product, float, ARRAY(float))
RUNTIME_FUNC_DEF(product, double, ARRAY(double))

/**
 * Get the sign of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return 1 if value is > 0, -1 if < 0, 0 if = 0
 */
template<typename T>
inline long sign(T value)
{
	if (value == 0)
		return 0;

	if (value > 0)
		return 1;

	return -1;
}

RUNTIME_FUNC_DEF(sign, int32_t, bool)
RUNTIME_FUNC_DEF(sign, int32_t, int32_t)
RUNTIME_FUNC_DEF(sign, int32_t, int64_t)
RUNTIME_FUNC_DEF(sign, int32_t, float)
RUNTIME_FUNC_DEF(sign, int32_t, double)

RUNTIME_FUNC_DEF(sign, int64_t, bool)
RUNTIME_FUNC_DEF(sign, int64_t, int32_t)
RUNTIME_FUNC_DEF(sign, int64_t, int64_t)
RUNTIME_FUNC_DEF(sign, int64_t, float)
RUNTIME_FUNC_DEF(sign, int64_t, double)

/**
 * Get the sine of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return sin
 */
template<typename T>
inline double sin(T value)
{
	return std::sin(value);
}

RUNTIME_FUNC_DEF(sin, float, float)
RUNTIME_FUNC_DEF(sin, double, double)

/**
 * Get the hyperbolic sine of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return sinh
 */
template<typename T>
inline double sinh(T value)
{
	return std::sinh(value);
}

RUNTIME_FUNC_DEF(sinh, float, float)
RUNTIME_FUNC_DEF(sinh, double, double)

/**
 * Get the square root of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return square root
 */
template<typename T>
inline double sqrt(T value)
{
	assert(value >= 0);
	return std::sqrt(value);
}

RUNTIME_FUNC_DEF(sqrt, float, float)
RUNTIME_FUNC_DEF(sqrt, double, double)

/**
 * Sum all the elements of an array.
 *
 * @tparam T 		 data type
 * @param array  array
 * @return sum of all the values
 */
template<typename T>
inline T sum(UnsizedArrayDescriptor<T> array)
{
	return std::accumulate(array.begin(), array.end(), 0, std::plus<T>());
}

RUNTIME_FUNC_DEF(sum, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(sum, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(sum, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(sum, float, ARRAY(float))
RUNTIME_FUNC_DEF(sum, double, ARRAY(double))

/**
 * Populate the destination matrix so that it becomes the symmetric version
 * of the source one, thus discarding the elements below the source diagonal.
 *
 * @tparam Destination	destination type
 * @tparam Source				source type
 * @param destination		destination matrix
 * @param source				source matrix
 */
template<typename Destination, typename Source>
void symmetric(UnsizedArrayDescriptor<Destination> destination, UnsizedArrayDescriptor<Source> source)
{
	using dimension_t = typename UnsizedArrayDescriptor<Destination>::dimension_t;

	// The two arrays must have exactly two dimensions
	assert(destination.getRank() == 2);
	assert(source.getRank() == 2);

	// The two matrixes must have the same dimensions
	assert(destination.getDimensionSize(0) == source.getDimensionSize(0));
	assert(destination.getDimensionSize(1) == source.getDimensionSize(1));

	auto size = destination.getDimensionSize(0);

	// Manually iterate on the dimensions, so that we can explore just half
	// of the source matrix.

	for (dimension_t i = 0; i < size; ++i)
	{
		for (dimension_t j = i; j < size; ++j)
		{
			destination.set({ i, j }, source.get(i, j));

			if (i != j)
				destination.set({ j, i }, source.get(i, j));
		}
	}
}

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(double))

/**
 * Get the tangent of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return tan
 */
template<typename T>
inline double tan(T value)
{
	return std::tan(value);
}

RUNTIME_FUNC_DEF(tan, float, float)
RUNTIME_FUNC_DEF(tan, double, double)

/**
 * Get the hyperbolic tangent of a value.
 *
 * @tparam T		 data type
 * @param value  value
 * @return tanh
 */
template<typename T>
inline double tanh(T value)
{
	return std::tanh(value);
}

RUNTIME_FUNC_DEF(tanh, float, float)
RUNTIME_FUNC_DEF(tanh, double, double)

/**
 * Transpose a matrix.
 *
 * @tparam Destination destination type
 * @tparam Source 		 source type
 * @param destination  destination matrix
 * @param source  		 source matrix
 */
template<typename Destination, typename Source>
void transpose(UnsizedArrayDescriptor<Destination> destination, UnsizedArrayDescriptor<Source> source)
{
	using dimension_t = typename UnsizedArrayDescriptor<Source>::dimension_t;

	// The two arrays must have exactly two dimensions
	assert(destination.getRank() == 2);
	assert(source.getRank() == 2);

	// The two matrixes must have transposed dimensions
	assert(destination.getDimensionSize(0) == source.getDimensionSize(1));
	assert(destination.getDimensionSize(1) == source.getDimensionSize(0));

	// Directly use the iterators, as we need to determine the current
	// indexes and transpose them to access the other matrix.

	for (auto it = source.begin(), end = source.end(); it != end; ++it)
	{
		auto indexes = it.getCurrentIndexes();
		assert(indexes.size() == 2);

		llvm::SmallVector<dimension_t, 2> transposedIndexes;

		for (auto revIt = indexes.rbegin(), revEnd = indexes.rend(); revIt != revEnd; ++revIt)
			transposedIndexes.push_back(*revIt);

		destination.set(transposedIndexes, *it);
	}
}

RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(double))

/**
 * Set all the elements of an array to zero.
 *
 * @tparam T data type
 * @param array   array to be populated
 */
template<typename T>
inline void zeros(UnsizedArrayDescriptor<T> array)
{
	for (auto& element : array)
		element = 0;
}

RUNTIME_FUNC_DEF(zeros, void, ARRAY(bool))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(float))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(double))
