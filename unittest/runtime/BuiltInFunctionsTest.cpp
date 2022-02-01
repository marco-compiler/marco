#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"
#include "marco/runtime/BuiltInFunctions.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cmath>
#include <numeric>

template<typename T, unsigned int N> using ArraySizes =
		std::array<typename ArrayDescriptor<T, N>::dimension_t, N>;

TEST(Runtime, abs_i1)
{
	std::array<bool, 2> data = { false, true };

	for (const auto& element : data)
	{
		auto result = NAME_MANGLED(abs, bool, bool)(element);
		EXPECT_EQ(result, element);
	}
}

TEST(Runtime, abs_i32)
{
	std::array<int32_t, 3> data = { -1, 0, 1 };

	for (const auto& element : data)
	{
		auto result = NAME_MANGLED(abs, int32_t, int32_t)(element);
		EXPECT_EQ(result, std::abs(element));
	}
}

TEST(Runtime, abs_i64)
{
	std::array<int64_t, 3> data = { -1, 0, 1 };

	for (const auto& element : data)
	{
		auto result = NAME_MANGLED(abs, int64_t, int64_t)(element);
		EXPECT_EQ(result, std::abs(element));
	}
}

TEST(Runtime, abs_f32)
{
	std::array<float, 3> data = { -1, 0, 1 };

	for (const auto& element : data)
	{
		auto result = NAME_MANGLED(abs, float, float)(element);
		EXPECT_EQ(result, std::abs(element));
	}
}

TEST(Runtime, abs_f64)
{
	std::array<double, 3> data = { -1, 0, 1 };

	EXPECT_EQ(NAME_MANGLED(abs, double, double)(data[0]), 1);
	EXPECT_EQ(NAME_MANGLED(abs, double, double)(data[1]), 0);
	EXPECT_EQ(NAME_MANGLED(abs, double, double)(data[2]), 1);
}

TEST(Runtime, acos_f32)
{
	EXPECT_NEAR(NAME_MANGLED(acos, float, float)(1), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(acos, float, float)(0.866025403403), M_PI / 6, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(acos, float, float)(0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(acos, float, float)(0), M_PI / 2, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(acos, float, float)(-0.707106781), M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(acos, float, float)(-0.866025403), M_PI * 5 / 6, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(acos, float, float)(-1), M_PI, 0.000001);
}

TEST(Runtime, acos_f64)
{
	EXPECT_NEAR(NAME_MANGLED(acos, double, double)(1), 0, 0.00001);
	EXPECT_NEAR(NAME_MANGLED(acos, double, double)(0.866025403), M_PI / 6, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(acos, double, double)(0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(acos, double, double)(0), M_PI / 2, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(acos, double, double)(-0.707106781), M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(acos, double, double)(-0.866025403), M_PI * 5 / 6, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(acos, double, double)(-1), M_PI, 0.000001);
}

TEST(Runtime, asin_f32)
{
	EXPECT_NEAR(NAME_MANGLED(asin, float, float)(1), M_PI / 2, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, float, float)(0.866025403), M_PI / 3, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, float, float)(0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, float, float)(0), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, float, float)(-0.707106781), -1 * M_PI / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, float, float)(-0.866025403), -1 * M_PI / 3, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, float, float)(-1), -1 * M_PI / 2, 0.000001);
}

TEST(Runtime, asin_f64)
{
	EXPECT_NEAR(NAME_MANGLED(asin, double, double)(1), M_PI / 2, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, double, double)(0.866025403), M_PI / 3, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, double, double)(0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, double, double)(0), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, double, double)(-0.707106781), -1 * M_PI / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, double, double)(-0.866025403), -1 * M_PI / 3, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(asin, double, double)(-1), -1 * M_PI / 2, 0.000001);
}

TEST(Runtime, atan_f32)
{
	EXPECT_NEAR(NAME_MANGLED(atan, float, float)(1), M_PI / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan, float, float)(0.577350269), M_PI / 6, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan, float, float)(0), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan, float, float)(-0.577350269), -1 * M_PI / 6, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan, float, float)(-1), -1 * M_PI / 4, 0.000001);
}

TEST(Runtime, atan_f64)
{
	EXPECT_NEAR(NAME_MANGLED(atan, double, double)(1), M_PI / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan, double, double)(0.577350269), M_PI / 6, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan, double, double)(0), 0, 0.0000001);
	EXPECT_NEAR(NAME_MANGLED(atan, double, double)(-0.577350269), -1 * M_PI / 6, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan, double, double)(-1), -1 * M_PI / 4, 0.000001);
}

TEST(Runtime, atan2_f32)
{
	EXPECT_NEAR(NAME_MANGLED(atan2, float, float, float)(0.707106781, 0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan2, float, float, float)(0.707106781, -0.707106781), M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan2, float, float, float)(-0.707106781, -0.707106781), -1 * M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan2, float, float, float)(-0.707106781, 0.707106781), -1 * M_PI / 4, 0.000001);
}

TEST(Runtime, atan2_f64)
{
	EXPECT_NEAR(NAME_MANGLED(atan2, double, double, double)(0.707106781, 0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan2, double, double, double)(0.707106781, -0.707106781), M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan2, double, double, double)(-0.707106781, -0.707106781), -1 * M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(atan2, double, double, double)(-0.707106781, 0.707106781), -1 * M_PI / 4, 0.000001);
}

TEST(Runtime, cos_f32)
{
	EXPECT_NEAR(NAME_MANGLED(cos, float, float)(0), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cos, float, float)(M_PI / 6), 0.866025403, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cos, float, float)(M_PI / 4), 0.707106781, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cos, float, float)(M_PI / 2), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cos, float, float)(M_PI), -1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cos, float, float)(2 * M_PI), 1, 0.000001);
}

TEST(Runtime, cos_f64)
{
	EXPECT_NEAR(NAME_MANGLED(cos, double, double)(0), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cos, double, double)(M_PI / 6), 0.866025403, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cos, double, double)(M_PI / 4), 0.707106781, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cos, double, double)(M_PI / 2), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cos, double, double)(M_PI), -1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cos, double, double)(2 * M_PI), 1, 0.000001);
}

TEST(Runtime, cosh_f32)
{
	EXPECT_NEAR(NAME_MANGLED(cosh, float, float)(0), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cosh, float, float)(1), 1.543080634, 0.000001);
}

TEST(Runtime, cosh_f64)
{
	EXPECT_NEAR(NAME_MANGLED(cosh, double, double)(0), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(cosh, double, double)(1), 1.543080634, 0.000001);
}

TEST(Runtime, diagonalSquareMatrix_i1_i1)
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(bool))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : false);
}

TEST(Runtime, diagonalSquareMatrix_i1_i32)	 // NOLINT
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<int32_t, 3> values = { 2, 2, 2 };
	ArrayDescriptor<int32_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int32_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(int32_t))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] > 0 : false);
}

TEST(Runtime, diagonalSquareMatrix_i1_i64)	 // NOLINT
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<int64_t, 3> values = { 2, 2, 2 };
	ArrayDescriptor<int64_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int64_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(int64_t))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] > 0 : false);
}

TEST(Runtime, diagonalSquareMatrix_i1_f32)	 // NOLINT
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 2, 2, 2 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(float))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] > 0 : false);
}

TEST(Runtime, diagonalSquareMatrix_i1_f64)	 // NOLINT
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 2, 2, 2 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(double))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] > 0 : false);
}

TEST(Runtime, diagonalSquareMatrix_i32_i1)	 // NOLINT
{
	std::array<int32_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(bool))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? (values[i] ? 1 : 0) : 0);
}

TEST(Runtime, diagonalSquareMatrix_i32_i32)	 // NOLINT
{
	std::array<int, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	std::array<int32_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int32_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(int32_t))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i32_i64)	 // NOLINT
{
	std::array<int32_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	std::array<int64_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int64_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(int64_t))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i32_f32)	 // NOLINT
{
	std::array<int32_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 1, 2, 3 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(float))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i32_f64)	 // NOLINT
{
	std::array<int32_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 1, 2, 3 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(double))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i64_i1)	 // NOLINT
{
	std::array<int64_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(bool))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? (values[i] ? 1 : 0) : 0);
}

TEST(Runtime, diagonalSquareMatrix_i64_i32)	 // NOLINT
{
	std::array<int64_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	std::array<int32_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int32_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(int32_t))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i64_i64)	 // NOLINT
{
	std::array<int64_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	std::array<int64_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int64_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(int64_t))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i64_f32)	 // NOLINT
{
	std::array<int64_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 1, 2, 3 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(float))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i64_f64)	 // NOLINT
{
	std::array<int64_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 1, 2, 3 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(double))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f32_i1)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(bool))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? (values[i] ? 1 : 0) : 0);
}

TEST(Runtime, diagonalSquareMatrix_f32_i32)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<int32_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int32_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(int32_t))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f32_i64)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<int64_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int64_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(int64_t))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f32_f32)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 1, 2, 3 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(float))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f32_f64)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 1, 2, 3 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(double))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f64_i1)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(bool))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? (values[i] ? 1 : 0) : 0);
}

TEST(Runtime, diagonalSquareMatrix_f64_i32)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<int32_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int32_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(int32_t))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f64_i64)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<int64_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int64_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(int64_t))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f64_f32)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 1, 2, 3 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(float))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f64_f64)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 1, 2, 3 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(double))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, exp_f32)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(exp, float, float)(0), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(exp, float, float)(1), 2.718281, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(exp, float, float)(2), 7.389056, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(exp, float, float)(-2), 0.135335, 0.000001);
}

TEST(Runtime, exp_f64)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(exp, double, double)(0), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(exp, double, double)(1), 2.718281, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(exp, double, double)(2), 7.389056, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(exp, double, double)(-2), 0.135335, 0.000001);
}

TEST(Runtime, fill_i1)	 // NOLINT
{
	std::array<bool, 3> data = { false, false, false };
	ArrayDescriptor<bool, 1> descriptor(data);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	bool value = true;
	NAME_MANGLED(fill, void, ARRAY(bool), bool)(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_i32)	 // NOLINT
{
	std::array<int32_t, 3> data = { 0, 0, 0 };
	ArrayDescriptor<int32_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	int value = 1;
	NAME_MANGLED(fill, void, ARRAY(int32_t), int32_t)(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_i64)	 // NOLINT
{
	std::array<int64_t, 3> data = { 0, 0, 0 };
	ArrayDescriptor<int64_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	long value = 1;
	NAME_MANGLED(fill, void, ARRAY(int64_t), int64_t)(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_f32)	 // NOLINT
{
	std::array<float, 3> data = { 0, 0, 0 };
	ArrayDescriptor<float, 1> descriptor(data);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	float value = 1;
	NAME_MANGLED(fill, void, ARRAY(float), float)(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_f64)	 // NOLINT
{
	std::array<double, 3> data = { 0, 0, 0 };
	ArrayDescriptor<double, 1> descriptor(data);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	double value = 1;
	NAME_MANGLED(fill, void, ARRAY(double), double)(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, identitySquareMatrix_i1)	 // NOLINT
{
	std::array<bool, 9> data = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> sizes = { 3, 3 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(bool))(unsized);

	for (size_t i = 0; i < sizes[0]; ++i)
		for (size_t j = 0; j < sizes[1]; j++)
			EXPECT_EQ(data[3 * i + j], i == j);
}

TEST(Runtime, identitySquareMatrix_i32)	 // NOLINT
{
	std::array<int32_t, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> sizes = { 3, 3 };

	ArrayDescriptor<int32_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(int32_t))(unsized);

	for (size_t i = 0; i < sizes[0]; ++i)
		for (size_t j = 0; j < sizes[1]; j++)
			EXPECT_EQ(data[3 * i + j], i == j ? 1 : 0);
}

TEST(Runtime, identitySquareMatrix_i64)	 // NOLINT
{
	std::array<int64_t, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> sizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(int64_t))(unsized);

	for (size_t i = 0; i < sizes[0]; ++i)
		for (size_t j = 0; j < sizes[1]; j++)
			EXPECT_EQ(data[3 * i + j], i == j ? 1 : 0);
}

TEST(Runtime, identitySquareMatrix_f32)	 // NOLINT
{
	std::array<float, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> sizes = { 3, 3 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(float))(unsized);

	for (size_t i = 0; i < sizes[0]; ++i)
		for (size_t j = 0; j < sizes[1]; j++)
			EXPECT_FLOAT_EQ(data[3 * i + j], i == j ? 1 : 0);
}

TEST(Runtime, identitySquareMatrix_f64)	 // NOLINT
{
	std::array<double, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> sizes = { 3, 3 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(double))(unsized);

	for (size_t i = 0; i < sizes[0]; ++i)
		for (size_t j = 0; j < sizes[1]; j++)
			EXPECT_DOUBLE_EQ(data[3 * i + j], i == j ? 1 : 0);
}

TEST(Runtime, linspace_i1)	 // NOLINT
{
	std::array<bool, 4> data = { true, false, false, false };
	ArrayDescriptor<bool, 1> descriptor(data);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	double start = 0;
	double end = 1;

	NAME_MANGLED(linspace, void, ARRAY(bool), double, double)(unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i)
		EXPECT_EQ(data[i], (start + i * (end - start) / (data.size() - 1)) > 0);
}

TEST(Runtime, linspace_i32)	 // NOLINT
{
	std::array<int32_t, 4> data = { -1, -1, -1, -1 };
	ArrayDescriptor<int32_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	double start = 0;
	double end = 2;

	NAME_MANGLED(linspace, void, ARRAY(int32_t), double, double)(unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i)
		EXPECT_EQ(data[i], (int32_t) (start + i * (end - start) / (data.size() - 1)));
}

TEST(Runtime, linspace_i64)	 // NOLINT
{
	std::array<int64_t, 4> data = { -1, -1, -1, -1 };
	ArrayDescriptor<int64_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	double start = 0;
	double end = 2;

	NAME_MANGLED(linspace, void, ARRAY(int64_t), double, double)(unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i)
		EXPECT_EQ(data[i], (int64_t) (start + i * (end - start) / (data.size() - 1)));
}

TEST(Runtime, linspace_f32)	 // NOLINT
{
	std::array<float, 4> data = { -1, -1, -1, -1 };
	ArrayDescriptor<float, 1> descriptor(data);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	double start = 0;
	double end = 2;

	NAME_MANGLED(linspace, void, ARRAY(float), double, double)(unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i)
		EXPECT_FLOAT_EQ(data[i], start +  i * (end - start) / (data.size() - 1));
}

TEST(Runtime, linspace_f64)	 // NOLINT
{
	std::array<double, 4> data = { -1, -1, -1, -1 };
	ArrayDescriptor<double, 1> descriptor(data);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	double start = 0;
	double end = 2;

	NAME_MANGLED(linspace, void, ARRAY(double), double, double)(unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i)
		EXPECT_FLOAT_EQ(data[i], start +  i * (end - start) / (data.size() - 1));
}

TEST(Runtime, log_f32)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(log, float, float)(1), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log, float, float)(2.718281828), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log, float, float)(7.389056099), 2, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log, float, float)(0.367879441), -1, 0.000001);
}

TEST(Runtime, log_f64)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(log, double, double)(1), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log, double, double)(2.718281828), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log, double, double)(7.389056099), 2, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log, double, double)(0.367879441), -1, 0.000001);
}

TEST(Runtime, log10_f32)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(log10, float, float)(1), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log10, float, float)(10), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log10, float, float)(100), 2, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log10, float, float)(0.1), -1, 0.000001);
}

TEST(Runtime, log10_f64)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(log10, double, double)(1), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log10, double, double)(10), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log10, double, double)(100), 2, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(log10, double, double)(0.1), -1, 0.000001);
}

TEST(Runtime, max_ai1)	 // NOLINT
{
	std::array<bool, 4> data = { false, true, true, false };
	ArraySizes<bool, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	auto result = NAME_MANGLED(max, bool, ARRAY(bool))(unsized);
	EXPECT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_ai32)	 // NOLINT
{
	std::array<int32_t, 4> data = { 5, 0, -3, 2 };
	ArraySizes<int32_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int32_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	auto result = NAME_MANGLED(max, int32_t, ARRAY(int32_t))(unsized);
	EXPECT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_ai64)	 // NOLINT
{
	std::array<int64_t, 4> data = { 5, 0, -3, 2 };
	ArraySizes<int64_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int64_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	auto result = NAME_MANGLED(max, int64_t, ARRAY(int64_t))(unsized);
	EXPECT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_af32)	 // NOLINT
{
	std::array<float, 4> data = { 5, 0, -3, 2 };
	ArraySizes<float, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	auto result = NAME_MANGLED(max, float, ARRAY(float))(unsized);
	EXPECT_FLOAT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_af64)	 // NOLINT
{
	std::array<double, 4> data = { 5, 0, -3, 2 };
	ArraySizes<double, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	auto result = NAME_MANGLED(max, double, ARRAY(double))(unsized);
	EXPECT_DOUBLE_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_i1_i1)	 // NOLINT
{
	std::array<bool, 4> x = { false, false, true, true };
	std::array<bool, 4> y = { false, true, false, true };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(max, bool, bool, bool)(x, y);
		ASSERT_EQ(result, std::max(x, y));
	}
}

TEST(Runtime, max_i32_i32)	 // NOLINT
{
	std::array<int32_t, 3> x = { 0, 1, 2 };
	std::array<int32_t, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(max, int32_t, int32_t, int32_t)(x, y);
		ASSERT_EQ(result, std::max(x, y));
	}
}

TEST(Runtime, max_i64_i64)	 // NOLINT
{
	std::array<int64_t, 3> x = { 0, 1, 2 };
	std::array<int64_t, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(max, int64_t, int64_t, int64_t)(x, y);
		ASSERT_EQ(result, std::max(x, y));
	}
}

TEST(Runtime, max_f32_f32)	 // NOLINT
{
	std::array<float, 3> x = { 0, 1, 2 };
	std::array<float, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(max, float, float, float)(x, y);
		ASSERT_EQ(result, std::max(x, y));
	}
}

TEST(Runtime, max_f64_f64)	 // NOLINT
{
	std::array<double, 3> x = { 0, 1, 2 };
	std::array<double, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(max, double, double, double)(x, y);
		ASSERT_EQ(result, std::max(x, y));
	}
}

TEST(Runtime, min_ai1)	 // NOLINT
{
	std::array<bool, 4> data = { false, true, true, false };
	ArraySizes<bool, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	auto result = NAME_MANGLED(min, bool, ARRAY(bool))(unsized);
	EXPECT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_ai32)	 // NOLINT
{
	std::array<int32_t, 4> data = { 5, 0, -3, 2 };
	ArraySizes<int32_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int32_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	auto result = NAME_MANGLED(min, int32_t, ARRAY(int32_t))(unsized);
	EXPECT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_ai64)	 // NOLINT
{
	std::array<int64_t, 4> data = { 5, 0, -3, 2 };
	ArraySizes<int64_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int64_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	auto result = NAME_MANGLED(min, int64_t, ARRAY(int64_t))(unsized);
	EXPECT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_af32)	 // NOLINT
{
	std::array<float, 4> data = { 5, 0, -3, 2 };
	ArraySizes<float, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	auto result = NAME_MANGLED(min, float, ARRAY(float))(unsized);
	EXPECT_FLOAT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_af64)	 // NOLINT
{
	std::array<double, 4> data = { 5, 0, -3, 2 };
	ArraySizes<double, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	auto result = NAME_MANGLED(min, double, ARRAY(double))(unsized);
	EXPECT_DOUBLE_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_i1_i1)	 // NOLINT
{
	std::array<bool, 4> x = { false, false, true, true };
	std::array<bool, 4> y = { false, true, false, true };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(min, bool, bool, bool)(x, y);
		ASSERT_EQ(result, std::min(x, y));
	}
}

TEST(Runtime, min_i32_i32)	 // NOLINT
{
	std::array<int32_t, 3> x = { 0, 1, 2 };
	std::array<int32_t, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(min, int32_t, int32_t, int32_t)(x, y);
		ASSERT_EQ(result, std::min(x, y));
	}
}

TEST(Runtime, min_i64_i64)	 // NOLINT
{
	std::array<int64_t, 3> x = { 0, 1, 2 };
	std::array<int64_t, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(min, int64_t, int64_t, int64_t)(x, y);
		ASSERT_EQ(result, std::min(x, y));
	}
}

TEST(Runtime, min_f32_f32)	 // NOLINT
{
	std::array<float, 3> x = { 0, 1, 2 };
	std::array<float, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(min, float, float, float)(x, y);
		ASSERT_EQ(result, std::min(x, y));
	}
}

TEST(Runtime, min_f64_f64)	 // NOLINT
{
	std::array<double, 3> x = { 0, 1, 2 };
	std::array<double, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(min, double, double, double)(x, y);
		ASSERT_EQ(result, std::min(x, y));
	}
}

TEST(Runtime, ones_i1)	 // NOLINT
{
	std::array<bool, 4> data = { false, false, false, false };
	ArraySizes<bool, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(bool))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, true);
}

TEST(Runtime, ones_i32)	 // NOLINT
{
	std::array<int32_t, 4> data = { 0, 0, 0, 0 };
	ArraySizes<int32_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int32_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(int32_t))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 1);
}

TEST(Runtime, ones_i64)	 // NOLINT
{
	std::array<int64_t, 4> data = { 0, 0, 0, 0 };
	ArraySizes<int64_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int64_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(int64_t))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 1);
}

TEST(Runtime, ones_f32)	 // NOLINT
{
	std::array<float, 4> data = { 0, 0, 0, 0 };
	ArraySizes<float, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(float))(unsized);

	for (const auto& element : data)
		EXPECT_FLOAT_EQ(element, 1);
}

TEST(Runtime, ones_f64)	 // NOLINT
{
	std::array<double, 4> data = { 0, 0, 0, 0 };
	ArraySizes<double, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(double))(unsized);

	for (const auto& element : data)
		EXPECT_DOUBLE_EQ(element, 1);
}

TEST(Runtime, product_ai1)	 // NOLINT
{
	std::array<bool, 3> data = { false, true, true };
	ArrayDescriptor<bool, 1> descriptor(data);
	UnsizedArrayDescriptor<bool> unsized(descriptor);
	auto result = NAME_MANGLED(product, bool, ARRAY(bool))(unsized);
	EXPECT_EQ(result, (bool) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_ai32)	 // NOLINT
{
	std::array<int32_t, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);
	auto result = NAME_MANGLED(product, int32_t, ARRAY(int32_t))(unsized);
	EXPECT_EQ(result, (int) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_ai64)	 // NOLINT
{
	std::array<int64_t, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);
	auto result = NAME_MANGLED(product, int64_t, ARRAY(int64_t))(unsized);
	EXPECT_EQ(result, (long) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_af32)	 // NOLINT
{
	std::array<float, 3> data = { 1, 2, 3 };
	ArrayDescriptor<float, 1> descriptor(data);
	UnsizedArrayDescriptor<float> unsized(descriptor);
	auto result = NAME_MANGLED(product, float, ARRAY(float))(unsized);
	EXPECT_FLOAT_EQ(result, (float) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_af64)	 // NOLINT
{
	std::array<double, 3> data = { 1, 2, 3 };
	ArrayDescriptor<double, 1> descriptor(data);
	UnsizedArrayDescriptor<double> unsized(descriptor);
	auto result = NAME_MANGLED(product, double, ARRAY(double))(unsized);
	EXPECT_DOUBLE_EQ(result, (double) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, sign_i1)	 // NOLINT
{
	std::array<bool, 2> data = { false, true };

	EXPECT_EQ(NAME_MANGLED(sign, int32_t, bool)(data[0]), 0);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, bool)(data[0]), 0);

	EXPECT_EQ(NAME_MANGLED(sign, int32_t, bool)(data[1]), 1);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, bool)(data[1]), 1);
}

TEST(Runtime, sign_i32)	 // NOLINT
{
	EXPECT_EQ(NAME_MANGLED(sign, int32_t, int32_t)(-2), -1);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, int32_t)(-2), -1);

	EXPECT_EQ(NAME_MANGLED(sign, int32_t, int32_t)(0), 0);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, int32_t)(0), 0);

	EXPECT_EQ(NAME_MANGLED(sign, int32_t, int32_t)(2), 1);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, int32_t)(2), 1);
}

TEST(Runtime, sign_i64)	 // NOLINT
{
	EXPECT_EQ(NAME_MANGLED(sign, int32_t, int64_t)(-2), -1);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, int64_t)(-2), -1);

	EXPECT_EQ(NAME_MANGLED(sign, int32_t, int64_t)(0), 0);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, int64_t)(0), 0);

	EXPECT_EQ(NAME_MANGLED(sign, int32_t, int64_t)(2), 1);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, int64_t)(2), 1);
}

TEST(Runtime, sign_f32)	 // NOLINT
{
	EXPECT_EQ(NAME_MANGLED(sign, int32_t, float)(-2), -1);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, float)(-2), -1);

	EXPECT_EQ(NAME_MANGLED(sign, int32_t, float)(0), 0);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, float)(0), 0);

	EXPECT_EQ(NAME_MANGLED(sign, int32_t, float)(2), 1);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, float)(2), 1);
}

TEST(Runtime, sign_f64)	 // NOLINT
{
	EXPECT_EQ(NAME_MANGLED(sign, int32_t, double)(-2), -1);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, double)(-2), -1);

	EXPECT_EQ(NAME_MANGLED(sign, int32_t, double)(0), 0);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, double)(0), 0);

	EXPECT_EQ(NAME_MANGLED(sign, int32_t, double)(2), 1);
	EXPECT_EQ(NAME_MANGLED(sign, int64_t, double)(2), 1);
}

TEST(Runtime, sin_f32)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(sin, float, float)(0), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sin, float, float)(M_PI / 6), 0.5, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sin, float, float)(M_PI / 4), 0.707106781, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sin, float, float)(M_PI / 2), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sin, float, float)(M_PI), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sin, float, float)(2 * M_PI), 0, 0.000001);
}

TEST(Runtime, sin_f64)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(sin, double, double)(0), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sin, double, double)(M_PI / 6), 0.5, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sin, double, double)(M_PI / 4), 0.707106781, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sin, double, double)(M_PI / 2), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sin, double, double)(M_PI), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sin, double, double)(2 * M_PI), 0, 0.000001);
}

TEST(Runtime, sqrt_f32)	 // NOLINT
{
	EXPECT_FLOAT_EQ(NAME_MANGLED(sqrt, float, float)(0), 0);
	EXPECT_FLOAT_EQ(NAME_MANGLED(sqrt, float, float)(1), 1);
	EXPECT_FLOAT_EQ(NAME_MANGLED(sqrt, float, float)(4), 2);
}

TEST(Runtime, sqrt_f64)	 // NOLINT
{
	EXPECT_DOUBLE_EQ(NAME_MANGLED(sqrt, double, double)(0), 0);
	EXPECT_DOUBLE_EQ(NAME_MANGLED(sqrt, double, double)(1), 1);
	EXPECT_DOUBLE_EQ(NAME_MANGLED(sqrt, double, double)(4), 2);
}

TEST(Runtime, sinh_f32)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(sinh, float, float)(0), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sinh, float, float)(1), 1.175201193, 0.000001);
}

TEST(Runtime, sinh_f64)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(sinh, double, double)(0), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(sinh, double, double)(1), 1.175201193, 0.000001);
}

TEST(Runtime, sum_ai1)	 // NOLINT
{
	std::array<bool, 3> data = { false, true, true };
	ArrayDescriptor<bool, 1> descriptor(data);
	UnsizedArrayDescriptor<bool> unsized(descriptor);
	auto result = NAME_MANGLED(sum, bool, ARRAY(bool))(unsized);
	EXPECT_EQ(result, (bool) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_ai32)	 // NOLINT
{
	std::array<int32_t, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);
	auto result = NAME_MANGLED(sum, int32_t, ARRAY(int32_t))(unsized);
	EXPECT_EQ(result, (int32_t) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_ai64)	 // NOLINT
{
	std::array<int64_t, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);
	auto result = NAME_MANGLED(sum, int64_t, ARRAY(int64_t))(unsized);
	EXPECT_EQ(result, (int64_t) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_af32)	 // NOLINT
{
	std::array<float, 3> data = { 1, 2, 3 };
	ArrayDescriptor<float, 1> descriptor(data);
	UnsizedArrayDescriptor<float> unsized(descriptor);
	auto result = NAME_MANGLED(sum, float, ARRAY(float))(unsized);
	EXPECT_FLOAT_EQ(result, (float) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_af64)	 // NOLINT
{
	std::array<double, 3> data = { 1, 2, 3 };
	ArrayDescriptor<double, 1> descriptor(data);
	UnsizedArrayDescriptor<double> unsized(descriptor);
	auto result = NAME_MANGLED(sum, double, ARRAY(double))(unsized);
	EXPECT_DOUBLE_EQ(result, (double) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, symmetric_ai1_ai1)	 // NOLINT
{
	std::array<bool, 9> source = { true, false, true, true, false, true, true, false, true };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (bool) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai1_ai32)	 // NOLINT
{
	std::array<int32_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(int32_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (bool) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai1_ai64)	 // NOLINT
{
	std::array<int64_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(int64_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (bool) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai1_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (bool) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai1_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (bool) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai32_ai1)	 // NOLINT
{
	std::array<bool, 9> source = { true, false, true, true, false, true, true, false, true };
	std::array<int32_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai32_ai32)	 // NOLINT
{
	std::array<int32_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int32_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(int32_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai32_ai64)	 // NOLINT
{
	std::array<int64_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int32_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(int64_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai32_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int32_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai32_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int32_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai64_ai1)	 // NOLINT
{
	std::array<bool, 9> source = { true, false, true, true, false, true, true, false, true };
	std::array<int64_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int64_t) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai64_ai32)	 // NOLINT
{
	std::array<int32_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int64_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(int32_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai64_ai64)	 // NOLINT
{
	std::array<int64_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int64_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(int64_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int64_t) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai64_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int64_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int64_t) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai64_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int64_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int64_t) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af32_ai1)	 // NOLINT
{
	std::array<bool, 9> source = { true, false, true, true, false, true, true, false, true };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (float) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af32_ai32)	 // NOLINT
{
	std::array<int32_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(int32_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (float) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af32_ai64)	 // NOLINT
{
	std::array<int64_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(int64_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (float) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af32_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (float) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af32_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (float) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af64_ai1)	 // NOLINT
{
	std::array<bool, 9> source = { true, false, true, true, false, true, true, false, true };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af64_ai32)	 // NOLINT
{
	std::array<int32_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(int32_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af64_ai64)	 // NOLINT
{
	std::array<int64_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(int64_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af64_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af64_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, tan_f32)	 // NOLINT
{
	std::array<float, 5> data = { 0, M_PI / 6, M_PI / 4, M_PI, 2 * M_PI };

	EXPECT_NEAR(NAME_MANGLED(tan, float, float)(data[0]), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(tan, float, float)(data[1]), 0.577350269, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(tan, float, float)(data[2]), 1, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(tan, float, float)(data[3]), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(tan, float, float)(data[4]), 0, 0.000001);
}

TEST(Runtime, tan_f64)	 // NOLINT
{
	std::array<double, 5> data = { 0, M_PI / 6, M_PI / 4, M_PI, 2 * M_PI };

	EXPECT_NEAR(NAME_MANGLED(tan, double, double)(data[0]), 0, 0.000000001);
	EXPECT_NEAR(NAME_MANGLED(tan, double, double)(data[1]), 0.577350269, 0.000000001);
	EXPECT_NEAR(NAME_MANGLED(tan, double, double)(data[2]), 1, 0.000000001);
	EXPECT_NEAR(NAME_MANGLED(tan, double, double)(data[3]), 0, 0.000000001);
	EXPECT_NEAR(NAME_MANGLED(tan, double, double)(data[4]), 0, 0.000000001);
}

TEST(Runtime, tanh_f32)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(tanh, float, float)(0), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(tanh, float, float)(1), 0.761594155, 0.000001);
}

TEST(Runtime, tanh_f64)	 // NOLINT
{
	EXPECT_NEAR(NAME_MANGLED(tanh, double, double)(0), 0, 0.000001);
	EXPECT_NEAR(NAME_MANGLED(tanh, double, double)(1), 0.761594155, 0.000001);
}

TEST(Runtime, transpose_ai1_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai1_ai32)	 // NOLINT
{
	std::array<int32_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(int32_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai1_ai64)	 // NOLINT
{
	std::array<int64_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(int64_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai1_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai1_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai32_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<int32_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai32_ai32)	 // NOLINT
{
	std::array<int32_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int32_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(int32_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai32_ai64)	 // NOLINT
{
	std::array<int64_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int32_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(int64_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai32_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int32_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai32_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int32_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai64_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<int64_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai64_ai32)	 // NOLINT
{
	std::array<int32_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int64_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(int32_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai64_ai64)	 // NOLINT
{
	std::array<int64_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int64_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(int64_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai64_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int64_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai64_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int64_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af32_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af32_ai32)	 // NOLINT
{
	std::array<int32_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(int32_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af32_ai64)	 // NOLINT
{
	std::array<int64_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(int64_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af32_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af32_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af64_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af64_ai32)	 // NOLINT
{
	std::array<int32_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(int32_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af64_ai64)	 // NOLINT
{
	std::array<int64_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(int64_t))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af64_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af64_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
}

TEST(Runtime, zeros_i1)	 // NOLINT
{
	std::array<bool, 4> data = { true, true, true, true };
	ArraySizes<bool, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(bool))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, false);
}

TEST(Runtime, zeros_i32)	 // NOLINT
{
	std::array<int32_t, 4> data = { 1, 1, 1, 1 };
	ArraySizes<int32_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int32_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(int32_t))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}

TEST(Runtime, zeros_i64)	 // NOLINT
{
	std::array<int64_t, 4> data = { 1, 1, 1, 1 };
	ArraySizes<int64_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int64_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(int64_t))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}

TEST(Runtime, zeros_f32)	 // NOLINT
{
	std::array<float, 4> data = { 1, 1, 1, 1 };
	ArraySizes<float, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(float))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}

TEST(Runtime, zeros_f64)	 // NOLINT
{
	std::array<double, 4> data = { 1, 1, 1, 1 };
	ArraySizes<double, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(double))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}
