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
	_Mfill_ai1_i1(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_i32)	 // NOLINT
{
	std::array<int, 3> data = { 0, 0, 0 };
	ArrayDescriptor<int, 1> descriptor(data);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	int value = 1;
	_Mfill_ai32_i32(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_i64)	 // NOLINT
{
	std::array<long, 3> data = { 0, 0, 0 };
	ArrayDescriptor<long, 1> descriptor(data);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	long value = 1;
	_Mfill_ai64_i64(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_f32)	 // NOLINT
{
	std::array<float, 3> data = { 0, 0, 0 };
	ArrayDescriptor<float, 1> descriptor(data);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	float value = 1;
	_Mfill_af32_f32(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_f64)	 // NOLINT
{
	std::array<double, 3> data = { 0, 0, 0 };
	ArrayDescriptor<double, 1> descriptor(data);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	double value = 1;
	_Mfill_af64_f64(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, identitySquareMatrix_i1)	 // NOLINT
{
	std::array<bool, 9> data = { false, true, true, true, false, true, true, true, false };
	std::array<unsigned long, 2> sizes = { 3, 3 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	_Midentity_ai1(unsized);

	for (size_t i = 0; i < sizes[0]; ++i)
		for (size_t j = 0; j < sizes[1]; j++)
			EXPECT_EQ(data[3 * i + j], i == j ? true : false);
}

TEST(Runtime, identitySquareMatrix_i32)	 // NOLINT
{
	std::array<int, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	std::array<unsigned long, 2> sizes = { 3, 3 };

	ArrayDescriptor<int, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	_Midentity_ai32(unsized);

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

	_Midentity_ai64(unsized);

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

	_Midentity_af32(unsized);

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

	_Midentity_af64(unsized);

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

	_Mdiagonal_ai1_ai1(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai1_ai32(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai1_ai64(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai1_af32(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai1_af64(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai32_ai1(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai32_ai32(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai32_ai64(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai32_af32(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai32_af64(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai64_ai1(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai64_ai32(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai64_ai64(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai64_af32(unsizedDestination, unsizedValues);

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

	_Mdiagonal_ai64_af64(unsizedDestination, unsizedValues);

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

	_Mdiagonal_af32_ai1(unsizedDestination, unsizedValues);

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

	_Mdiagonal_af32_ai32(unsizedDestination, unsizedValues);

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

	_Mdiagonal_af32_ai64(unsizedDestination, unsizedValues);

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

	_Mdiagonal_af32_af32(unsizedDestination, unsizedValues);

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

	_Mdiagonal_af32_af64(unsizedDestination, unsizedValues);

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

	_Mdiagonal_af64_ai1(unsizedDestination, unsizedValues);

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

	_Mdiagonal_af64_ai32(unsizedDestination, unsizedValues);

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

	_Mdiagonal_af64_ai64(unsizedDestination, unsizedValues);

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

	_Mdiagonal_af64_af32(unsizedDestination, unsizedValues);

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

	_Mdiagonal_af64_af64(unsizedDestination, unsizedValues);

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

	_Mzeros_ai1(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, false);
}

TEST(Runtime, zeros_i32)	 // NOLINT
{
	std::array<int, 4> data = { 1, 1, 1, 1 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<int, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	_Mzeros_ai32(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}

TEST(Runtime, zeros_i64)	 // NOLINT
{
	std::array<long, 4> data = { 1, 1, 1, 1 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<long, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	_Mzeros_ai64(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}

TEST(Runtime, zeros_f32)	 // NOLINT
{
	std::array<float, 4> data = { 1, 1, 1, 1 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	_Mzeros_af32(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}

TEST(Runtime, zeros_f64)	 // NOLINT
{
	std::array<double, 4> data = { 1, 1, 1, 1 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	_Mzeros_af64(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 0);
}

TEST(Runtime, ones_i1)	 // NOLINT
{
	std::array<bool, 4> data = { false, false, false, false };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	_Mones_ai1(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, true);
}

TEST(Runtime, ones_i32)	 // NOLINT
{
	std::array<int, 4> data = { 0, 0, 0, 0 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<int, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	_Mones_ai32(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 1);
}

TEST(Runtime, ones_i64)	 // NOLINT
{
	std::array<long, 4> data = { 0, 0, 0, 0 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<long, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	_Mones_ai64(unsized);

	for (const auto& element : data)
		EXPECT_EQ(element, 1);
}

TEST(Runtime, ones_f32)	 // NOLINT
{
	std::array<float, 4> data = { 0, 0, 0, 0 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	_Mones_af32(unsized);

	for (const auto& element : data)
		EXPECT_FLOAT_EQ(element, 1);
}

TEST(Runtime, ones_f64)	 // NOLINT
{
	std::array<double, 4> data = { 0, 0, 0, 0 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	_Mones_af64(unsized);

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

	_Mlinspace_ai1_f64_f64(unsized, start, end);

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

	_Mlinspace_ai32_f64_f64(unsized, start, end);

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

	_Mlinspace_ai64_f64_f64(unsized, start, end);

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

	_Mlinspace_af32_f64_f64(unsized, start, end);

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

	_Mlinspace_af64_f64_f64(unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i)
		EXPECT_FLOAT_EQ(data[i], start +  i * (end - start) / (data.size() - 1));
}

TEST(Runtime, min_ai1)	 // NOLINT
{
	std::array<bool, 4> data = { false, true, true, false };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	EXPECT_EQ(_Mmin_ai1(unsized), *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_ai32)	 // NOLINT
{
	std::array<int, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<int, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	EXPECT_EQ(_Mmin_ai32(unsized), *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_ai64)	 // NOLINT
{
	std::array<long, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<long, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	EXPECT_EQ(_Mmin_ai64(unsized), *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_af32)	 // NOLINT
{
	std::array<float, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	EXPECT_FLOAT_EQ(_Mmin_af32(unsized), *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_af64)	 // NOLINT
{
	std::array<double, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	EXPECT_DOUBLE_EQ(_Mmin_af64(unsized), *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_i1_i1)	 // NOLINT
{
	std::array<bool, 4> x = { false, false, true, true };
	std::array<bool, 4> y = { false, true, false, true };

	for (const auto& [x, y] : llvm::zip(x, y))
		ASSERT_EQ(_Mmin_i1_i1(x, y), std::min(x, y));
}

TEST(Runtime, min_i32_i32)	 // NOLINT
{
	std::array<int, 3> x = { 0, 1, 2 };
	std::array<int, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
		ASSERT_EQ(_Mmin_i32_i32(x, y), std::min(x, y));
}

TEST(Runtime, min_i64_i64)	 // NOLINT
{
	std::array<long, 3> x = { 0, 1, 2 };
	std::array<long, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
		ASSERT_EQ(_Mmin_i64_i64(x, y), std::min(x, y));
}

TEST(Runtime, min_f32_f32)	 // NOLINT
{
	std::array<float, 3> x = { 0, 1, 2 };
	std::array<float, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
		ASSERT_FLOAT_EQ(_Mmin_f32_f32(x, y), std::min(x, y));
}

TEST(Runtime, min_f64_f64)	 // NOLINT
{
	std::array<double, 3> x = { 0, 1, 2 };
	std::array<double, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
		ASSERT_DOUBLE_EQ(_Mmin_f64_f64(x, y), std::min(x, y));
}

TEST(Runtime, max_ai1)	 // NOLINT
{
	std::array<bool, 4> data = { false, true, true, false };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	EXPECT_EQ(_Mmax_ai1(unsized), *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_ai32)	 // NOLINT
{
	std::array<int, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<int, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	EXPECT_EQ(_Mmax_ai32(unsized), *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_ai64)	 // NOLINT
{
	std::array<long, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<long, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	EXPECT_EQ(_Mmax_ai64(unsized), *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_af32)	 // NOLINT
{
	std::array<float, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	EXPECT_FLOAT_EQ(_Mmax_af32(unsized), *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_af64)	 // NOLINT
{
	std::array<double, 4> data = { 5, 0, -3, 2 };
	std::array<unsigned long, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	EXPECT_DOUBLE_EQ(_Mmax_af64(unsized), *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_i1_i1)	 // NOLINT
{
	std::array<bool, 4> x = { false, false, true, true };
	std::array<bool, 4> y = { false, true, false, true };

	for (const auto& [x, y] : llvm::zip(x, y))
		ASSERT_EQ(_Mmax_i1_i1(x, y), std::max(x, y));
}

TEST(Runtime, max_i32_i32)	 // NOLINT
{
	std::array<int, 3> x = { 0, 1, 2 };
	std::array<int, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
		ASSERT_EQ(_Mmax_i32_i32(x, y), std::max(x, y));
}

TEST(Runtime, max_i64_i64)	 // NOLINT
{
	std::array<long, 3> x = { 0, 1, 2 };
	std::array<long, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
		ASSERT_EQ(_Mmax_i64_i64(x, y), std::max(x, y));
}

TEST(Runtime, max_f32_f32)	 // NOLINT
{
	std::array<float, 3> x = { 0, 1, 2 };
	std::array<float, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
		ASSERT_FLOAT_EQ(_Mmax_f32_f32(x, y), std::max(x, y));
}

TEST(Runtime, max_f64_f64)	 // NOLINT
{
	std::array<double, 3> x = { 0, 1, 2 };
	std::array<double, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y))
		ASSERT_DOUBLE_EQ(_Mmax_f64_f64(x, y), std::max(x, y));
}

TEST(Runtime, sum_ai1)	 // NOLINT
{
	std::array<bool, 3> data = { false, true, true };
	ArrayDescriptor<bool, 1> descriptor(data);
	UnsizedArrayDescriptor<bool> unsized(descriptor);
	EXPECT_EQ(_Msum_ai1(unsized), (bool) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_ai32)	 // NOLINT
{
	std::array<int, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int, 1> descriptor(data);
	UnsizedArrayDescriptor<int> unsized(descriptor);
	EXPECT_EQ(_Msum_ai32(unsized), (int) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_ai64)	 // NOLINT
{
	std::array<long, 3> data = { 1, 2, 3 };
	ArrayDescriptor<long, 1> descriptor(data);
	UnsizedArrayDescriptor<long> unsized(descriptor);
	EXPECT_EQ(_Msum_ai64(unsized), (long) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_af32)	 // NOLINT
{
	std::array<float, 3> data = { 1, 2, 3 };
	ArrayDescriptor<float, 1> descriptor(data);
	UnsizedArrayDescriptor<float> unsized(descriptor);
	EXPECT_FLOAT_EQ(_Msum_af32(unsized), (float) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_af64)	 // NOLINT
{
	std::array<double, 3> data = { 1, 2, 3 };
	ArrayDescriptor<double, 1> descriptor(data);
	UnsizedArrayDescriptor<double> unsized(descriptor);
	EXPECT_DOUBLE_EQ(_Msum_af64(unsized), (double) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, product_ai1)	 // NOLINT
{
	std::array<bool, 3> data = { false, true, true };
	ArrayDescriptor<bool, 1> descriptor(data);
	UnsizedArrayDescriptor<bool> unsized(descriptor);
	EXPECT_EQ(_Mproduct_ai1(unsized), (bool) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_ai32)	 // NOLINT
{
	std::array<int, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int, 1> descriptor(data);
	UnsizedArrayDescriptor<int> unsized(descriptor);
	EXPECT_EQ(_Mproduct_ai32(unsized), (int) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_ai64)	 // NOLINT
{
	std::array<long, 3> data = { 1, 2, 3 };
	ArrayDescriptor<long, 1> descriptor(data);
	UnsizedArrayDescriptor<long> unsized(descriptor);
	EXPECT_EQ(_Mproduct_ai64(unsized), (long) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_af32)	 // NOLINT
{
	std::array<float, 3> data = { 1, 2, 3 };
	ArrayDescriptor<float, 1> descriptor(data);
	UnsizedArrayDescriptor<float> unsized(descriptor);
	EXPECT_FLOAT_EQ(_Mproduct_af32(unsized), (float) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_af64)	 // NOLINT
{
	std::array<double, 3> data = { 1, 2, 3 };
	ArrayDescriptor<double, 1> descriptor(data);
	UnsizedArrayDescriptor<double> unsized(descriptor);
	EXPECT_DOUBLE_EQ(_Mproduct_af64(unsized), (double) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, transpose_ai1_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	_Mtranspose_ai1_ai1(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai1_ai32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai1_ai64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai1_af32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai1_af64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai32_ai1(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai32_ai32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai32_ai64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai32_af32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai32_af64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai64_ai1(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai64_ai32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai64_ai64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai64_af32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_ai64_af64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_af32_ai1(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_af32_ai32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_af32_ai64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_af32_af32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_af32_af64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_af64_ai1(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_af64_ai32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_af64_ai64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_af64_af32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Mtranspose_af64_af64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai1_ai1(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai1_ai32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai1_ai64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai1_af32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai1_af64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai32_ai1(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai32_ai32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai32_ai64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai32_af32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai32_af64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai64_ai1(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai64_ai32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai64_ai64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai64_af32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_ai64_af64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_af32_ai1(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_af32_ai32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_af32_ai64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_af32_af32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_af32_af64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_af64_ai1(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_af64_ai32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_af64_ai64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_af64_af32(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
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

	_Msymmetric_af64_af64(unsizedDestination, unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimensionSize(0); ++i)
		for (size_t j = i; j < sourceDescriptor.getDimensionSize(1); ++j)
		{
			EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
			EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
		}
}
