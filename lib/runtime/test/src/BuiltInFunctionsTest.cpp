#include <gtest/gtest.h>
#include <modelica/runtime/Runtime.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>

TEST(Runtime, fill_i1)	 // NOLINT
{
	std::array<bool, 3> data = { false, false, false };
	std::array<unsigned long, 1> sizes = { 3 };

	ArrayDescriptor<bool, 1> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	bool value = true;
	_Mfill_ai1_i1(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_i32)	 // NOLINT
{
	std::array<int, 3> data = { 0, 0, 0 };
	std::array<unsigned long, 1> sizes = { 3 };

	ArrayDescriptor<int, 1> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	int value = 1;
	_Mfill_ai32_i32(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_i64)	 // NOLINT
{
	std::array<long, 3> data = { 0, 0, 0 };
	std::array<unsigned long, 1> sizes = { 3 };

	ArrayDescriptor<long, 1> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	long value = 1;
	_Mfill_ai64_i64(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_f32)	 // NOLINT
{
	std::array<float, 3> data = { 0, 0, 0 };
	std::array<unsigned long, 1> sizes = { 3 };

	ArrayDescriptor<float, 1> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	float value = 1;
	_Mfill_af32_f32(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, fill_f64)	 // NOLINT
{
	std::array<double, 3> data = { 0, 0, 0 };
	std::array<unsigned long, 1> sizes = { 3 };

	ArrayDescriptor<double, 1> descriptor(data.data(), sizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<bool, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<int, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<long, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<float, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<double, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<bool, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<int, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<long, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<float, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<double, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<bool, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<int, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<long, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<float, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<double, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<bool, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<int, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<long, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<float, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<double, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<bool, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<int, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<long, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<float, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> valuesSizes = { 3 };

	ArrayDescriptor<double, 1> valuesDescriptor(values.data(), valuesSizes);
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
	std::array<unsigned long, 1> sizes = { 4 };

	ArrayDescriptor<bool, 1> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	double start = 0;
	double end = 1;

	_Mlinspace_ai1_f64_f64(unsized, start, end);

	for (size_t i = 0; i < sizes[0]; ++i)
		EXPECT_EQ(data[i], (start + i * (end - start) / (sizes[0] - 1)) > 0);
}

TEST(Runtime, linspace_i32)	 // NOLINT
{
	std::array<int, 4> data = { -1, -1, -1, -1 };
	std::array<unsigned long, 1> sizes = { 4 };

	ArrayDescriptor<int, 1> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	double start = 0;
	double end = 2;

	_Mlinspace_ai32_f64_f64(unsized, start, end);

	for (size_t i = 0; i < sizes[0]; ++i)
		EXPECT_EQ(data[i], (int) (start + i * (end - start) / (sizes[0] - 1)));
}

TEST(Runtime, linspace_i64)	 // NOLINT
{
	std::array<long, 4> data = { -1, -1, -1, -1 };
	std::array<unsigned long, 1> sizes = { 4 };

	ArrayDescriptor<long, 1> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	double start = 0;
	double end = 2;

	_Mlinspace_ai64_f64_f64(unsized, start, end);

	for (size_t i = 0; i < sizes[0]; ++i)
		EXPECT_EQ(data[i], (long) (start + i * (end - start) / (sizes[0] - 1)));
}

TEST(Runtime, linspace_f32)	 // NOLINT
{
	std::array<float, 4> data = { -1, -1, -1, -1 };
	std::array<unsigned long, 1> sizes = { 4 };

	ArrayDescriptor<float, 1> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	double start = 0;
	double end = 2;

	_Mlinspace_af32_f64_f64(unsized, start, end);

	for (size_t i = 0; i < sizes[0]; ++i)
		EXPECT_FLOAT_EQ(data[i], start +  i * (end - start) / (sizes[0] - 1));
}

TEST(Runtime, linspace_f64)	 // NOLINT
{
	std::array<double, 4> data = { -1, -1, -1, -1 };
	std::array<unsigned long, 1> sizes = { 4 };

	ArrayDescriptor<double, 1> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	double start = 0;
	double end = 2;

	_Mlinspace_af64_f64_f64(unsized, start, end);

	for (size_t i = 0; i < sizes[0]; ++i)
		EXPECT_FLOAT_EQ(data[i], start +  i * (end - start) / (sizes[0] - 1));
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
