#include <gtest/gtest.h>
#include <modelica/runtime/Runtime.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <numeric>

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
	std::array<int, 3> data = { 0, 0, 0 };
	ArrayDescriptor<int, 1> descriptor(data);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	int value = 1;
	NAME_MANGLED(fill, void, ARRAY(int), int)(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_i64)	 // NOLINT
{
	std::array<long, 3> data = { 0, 0, 0 };
	ArrayDescriptor<long, 1> descriptor(data);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	long value = 1;
	NAME_MANGLED(fill, void, ARRAY(long), long)(unsized, value);

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
	std::array<unsigned long, 2> sizes = { 3, 3 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(bool))(unsized);

	for (size_t i = 0; i < sizes[0]; ++i)
		for (size_t j = 0; j < sizes[1]; j++)
			EXPECT_EQ(data[3 * i + j], i == j);
}

TEST(Runtime, identitySquareMatrix_i32)	 // NOLINT
{
	std::array<int, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> sizes = { 3, 3 };

	ArrayDescriptor<int, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(int))(unsized);

	for (size_t i = 0; i < sizes[0]; ++i)
		for (size_t j = 0; j < sizes[1]; j++)
			EXPECT_EQ(data[3 * i + j], i == j ? 1 : 0);
}

TEST(Runtime, identitySquareMatrix_i64)	 // NOLINT
{
	std::array<long, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> sizes = { 3, 3 };

	ArrayDescriptor<long, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(long))(unsized);

	for (size_t i = 0; i < sizes[0]; ++i)
		for (size_t j = 0; j < sizes[1]; j++)
			EXPECT_EQ(data[3 * i + j], i == j ? 1 : 0);
}

TEST(Runtime, identitySquareMatrix_f32)	 // NOLINT
{
	std::array<float, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> sizes = { 3, 3 };

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
	std::array<unsigned long, 2> sizes = { 3, 3 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(double))(unsized);

	for (size_t i = 0; i < sizes[0]; ++i)
		for (size_t j = 0; j < sizes[1]; j++)
			EXPECT_DOUBLE_EQ(data[3 * i + j], i == j ? 1 : 0);
}

TEST(Runtime, diagonalSquareMatrix_i1_i1)	 // NOLINT
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

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
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<int, 3> values = { 2, 2, 2 };
	ArrayDescriptor<int, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(int))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] > 0 : false);
}

TEST(Runtime, diagonalSquareMatrix_i1_i64)	 // NOLINT
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<long, 3> values = { 2, 2, 2 };
	ArrayDescriptor<long, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<long> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(long))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] > 0 : false);
}

TEST(Runtime, diagonalSquareMatrix_i1_f32)	 // NOLINT
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

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
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

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
	std::array<int, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int), ARRAY(bool))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? (values[i] ? 1 : 0) : 0);
}

TEST(Runtime, diagonalSquareMatrix_i32_i32)	 // NOLINT
{
	std::array<int, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	std::array<int, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int), ARRAY(int))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i32_i64)	 // NOLINT
{
	std::array<int, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	std::array<long, 3> values = { 1, 2, 3 };
	ArrayDescriptor<long, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<long> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int), ARRAY(long))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i32_f32)	 // NOLINT
{
	std::array<int, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 1, 2, 3 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int), ARRAY(float))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i32_f64)	 // NOLINT
{
	std::array<int, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 1, 2, 3 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int), ARRAY(double))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i64_i1)	 // NOLINT
{
	std::array<long, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(long), ARRAY(bool))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? (values[i] ? 1 : 0) : 0);
}

TEST(Runtime, diagonalSquareMatrix_i64_i32)	 // NOLINT
{
	std::array<long, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	std::array<int, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(long), ARRAY(int))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i64_i64)	 // NOLINT
{
	std::array<long, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	std::array<long, 3> values = { 1, 2, 3 };
	ArrayDescriptor<long, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<long> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(long), ARRAY(long))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i64_f32)	 // NOLINT
{
	std::array<long, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 1, 2, 3 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(long), ARRAY(float))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_i64_f64)	 // NOLINT
{
	std::array<long, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 1, 2, 3 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(long), ARRAY(double))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f32_i1)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

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
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<int, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(int))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f32_i64)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<long, 3> values = { 1, 2, 3 };
	ArrayDescriptor<long, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<long> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(long))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f32_f32)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

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
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

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
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

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
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<int, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(int))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f64_i64)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<long, 3> values = { 1, 2, 3 };
	ArrayDescriptor<long, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<long> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(long))(unsizedDestination, unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i)
		for (size_t j = 0; j < destinationSizes[1]; j++)
			EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? values[i] : 0);
}

TEST(Runtime, diagonalSquareMatrix_f64_f32)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

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
	std::array<unsigned long, 2> destinationSizes = { 3, 3 };

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

TEST(Runtime, zeros_i1)	 // NOLINT
{
	std::array<bool, 4> data = { true, true, true, true };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(bool))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, false);
}

TEST(Runtime, zeros_i32)	 // NOLINT
{
	std::array<int, 4> data = { 1, 1, 1, 1 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<int, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(int))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}

TEST(Runtime, zeros_i64)	 // NOLINT
{
	std::array<long, 4> data = { 1, 1, 1, 1 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<long, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(long))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}

TEST(Runtime, zeros_f32)	 // NOLINT
{
	std::array<float, 4> data = { 1, 1, 1, 1 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(float))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}

TEST(Runtime, zeros_f64)	 // NOLINT
{
	std::array<double, 4> data = { 1, 1, 1, 1 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(double))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}

TEST(Runtime, ones_i1)	 // NOLINT
{
	std::array<bool, 4> data = { false, false, false, false };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(bool))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, true);
}

TEST(Runtime, ones_i32)	 // NOLINT
{
	std::array<int, 4> data = { 0, 0, 0, 0 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<int, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(int))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 1);
}

TEST(Runtime, ones_i64)	 // NOLINT
{
	std::array<long, 4> data = { 0, 0, 0, 0 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<long, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(long))(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 1);
}

TEST(Runtime, ones_f32)	 // NOLINT
{
	std::array<float, 4> data = { 0, 0, 0, 0 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(float))(unsized);

	for (const auto& element : data)
		EXPECT_FLOAT_EQ(element, 1);
}

TEST(Runtime, ones_f64)	 // NOLINT
{
	std::array<double, 4> data = { 0, 0, 0, 0 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(double))(unsized);

	for (const auto& element : data)
		EXPECT_DOUBLE_EQ(element, 1);
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
	std::array<int, 4> data = { -1, -1, -1, -1 };
	ArrayDescriptor<int, 1> descriptor(data);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	double start = 0;
	double end = 2;

	NAME_MANGLED(linspace, void, ARRAY(int), double, double)(unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i)
		EXPECT_EQ(data[i], (int) (start + i * (end - start) / (data.size() - 1)));
}

TEST(Runtime, linspace_i64)	 // NOLINT
{
	std::array<long, 4> data = { -1, -1, -1, -1 };
	ArrayDescriptor<long, 1> descriptor(data);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	double start = 0;
	double end = 2;

	NAME_MANGLED(linspace, void, ARRAY(long), double, double)(unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i)
		EXPECT_EQ(data[i], (long) (start + i * (end - start) / (data.size() - 1)));
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

TEST(Runtime, min_ai1)	 // NOLINT
{
	std::array<bool, 4> data = { false, true, true, false };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	auto result = NAME_MANGLED(min, bool, ARRAY(bool))(unsized);
	EXPECT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_ai32)	 // NOLINT
{
	std::array<int, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<int, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	auto result = NAME_MANGLED(min, int, ARRAY(int))(unsized);
	EXPECT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_ai64)	 // NOLINT
{
	std::array<long, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<long, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	auto result = NAME_MANGLED(min, long, ARRAY(long))(unsized);
	EXPECT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_af32)	 // NOLINT
{
	std::array<float, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	auto result = NAME_MANGLED(min, float, ARRAY(float))(unsized);
	EXPECT_FLOAT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_af64)	 // NOLINT
{
	std::array<double, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

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
	std::array<int, 3> x = { 0, 1, 2 };
	std::array<int, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(min, int, int, int)(x, y);
		ASSERT_EQ(result, std::min(x, y));
	}
}

TEST(Runtime, min_i64_i64)	 // NOLINT
{
	std::array<long, 3> x = { 0, 1, 2 };
	std::array<long, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(min, long, long, long)(x, y);
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

TEST(Runtime, max_ai1)	 // NOLINT
{
	std::array<bool, 4> data = { false, true, true, false };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	auto result = NAME_MANGLED(max, bool, ARRAY(bool))(unsized);
	EXPECT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_ai32)	 // NOLINT
{
	std::array<int, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<int, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	auto result = NAME_MANGLED(max, int, ARRAY(int))(unsized);
	EXPECT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_ai64)	 // NOLINT
{
	std::array<long, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<long, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	auto result = NAME_MANGLED(max, long, ARRAY(long))(unsized);
	EXPECT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_af32)	 // NOLINT
{
	std::array<float, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	auto result = NAME_MANGLED(max, float, ARRAY(float))(unsized);
	EXPECT_FLOAT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_af64)	 // NOLINT
{
	std::array<double, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

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
	std::array<int, 3> x = { 0, 1, 2 };
	std::array<int, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(max, int, int, int)(x, y);
		ASSERT_EQ(result, std::max(x, y));
	}
}

TEST(Runtime, max_i64_i64)	 // NOLINT
{
	std::array<long, 3> x = { 0, 1, 2 };
	std::array<long, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		auto result = NAME_MANGLED(max, long, long, long)(x, y);
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
	std::array<int, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int, 1> descriptor(data);
	UnsizedArrayDescriptor<int> unsized(descriptor);
	auto result = NAME_MANGLED(sum, int, ARRAY(int))(unsized);
	EXPECT_EQ(result, (int) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_ai64)	 // NOLINT
{
	std::array<long, 3> data = { 1, 2, 3 };
	ArrayDescriptor<long, 1> descriptor(data);
	UnsizedArrayDescriptor<long> unsized(descriptor);
	auto result = NAME_MANGLED(sum, long, ARRAY(long))(unsized);
	EXPECT_EQ(result, (long) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
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
	std::array<int, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int, 1> descriptor(data);
	UnsizedArrayDescriptor<int> unsized(descriptor);
	auto result = NAME_MANGLED(product, int, ARRAY(int))(unsized);
	EXPECT_EQ(result, (int) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_ai64)	 // NOLINT
{
	std::array<long, 3> data = { 1, 2, 3 };
	ArrayDescriptor<long, 1> descriptor(data);
	UnsizedArrayDescriptor<long> unsized(descriptor);
	auto result = NAME_MANGLED(product, long, ARRAY(long))(unsized);
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
	std::array<int, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(int))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai1_ai64)	 // NOLINT
{
	std::array<long, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(long))(unsizedDestination, unsizedSource);

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
	std::array<int, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai32_ai32)	 // NOLINT
{
	std::array<int, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int), ARRAY(int))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai32_ai64)	 // NOLINT
{
	std::array<long, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int), ARRAY(long))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai32_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai32_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai64_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<long, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(long), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (long) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai64_ai32)	 // NOLINT
{
	std::array<int, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<long, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(long), ARRAY(int))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (long) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai64_ai64)	 // NOLINT
{
	std::array<long, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<long, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(long), ARRAY(long))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (long) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai64_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<long, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(long), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (long) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai64_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<long, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(long), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (long) sourceDescriptor.get(i, j));
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
	std::array<int, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(int))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af32_ai64)	 // NOLINT
{
	std::array<long, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(long))(unsizedDestination, unsizedSource);

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
	std::array<int, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(int))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_af64_ai64)	 // NOLINT
{
	std::array<long, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(long))(unsizedDestination, unsizedSource);

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
	std::array<int, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(int))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (bool) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai1_ai64)	 // NOLINT
{
	std::array<long, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(long))(unsizedDestination, unsizedSource);

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
	std::array<int, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai32_ai32)	 // NOLINT
{
	std::array<int, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int), ARRAY(int))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai32_ai64)	 // NOLINT
{
	std::array<long, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int), ARRAY(long))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai32_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai32_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai64_ai1)	 // NOLINT
{
	std::array<bool, 9> source = { true, false, true, true, false, true, true, false, true };
	std::array<long, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(long), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (long) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (long) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai64_ai32)	 // NOLINT
{
	std::array<int, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<long, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(long), ARRAY(int))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (int) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (int) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai64_ai64)	 // NOLINT
{
	std::array<long, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<long, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(long), ARRAY(long))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (long) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (long) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai64_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<long, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(long), ARRAY(float))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (long) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (long) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_ai64_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<long, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(long), ARRAY(double))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (long) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (long) sourceDescriptor.get(i, j));
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
	std::array<int, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(int))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (float) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af32_ai64)	 // NOLINT
{
	std::array<long, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(long))(unsizedDestination, unsizedSource);

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
	std::array<int, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(int))(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
		}
}

TEST(Runtime, symmetric_af64_ai64)	 // NOLINT
{
	std::array<long, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(long))(unsizedDestination, unsizedSource);

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
