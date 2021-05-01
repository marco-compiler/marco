#include <gtest/gtest.h>
#include <modelica/runtime/Runtime.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>

TEST(Runtime, fill_i1)	 // NOLINT
{
	std::array<bool, 3> data = { false, false, false };
	std::array<long, 1> sizes = { 3 };

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
	std::array<long, 1> sizes = { 3 };

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
	std::array<long, 1> sizes = { 3 };

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
	std::array<long, 1> sizes = { 3 };

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
	std::array<long, 1> sizes = { 3 };

	ArrayDescriptor<double, 1> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	double value = 1;
	_Mfill_af64_f64(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}

TEST(Runtime, identitySquareMatrix_i1)	 // NOLINT
{
	std::array<bool, 9> data = { false, false, false, false, false, false, false, false, false };
	std::array<long, 2> sizes = { 3, 3 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	_Midentity_ai1(unsized);

	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 3; j++)
			EXPECT_EQ(data[3 * i + j], i == j ? true : false);
}

TEST(Runtime, identitySquareMatrix_i32)	 // NOLINT
{
	std::array<int, 9> data = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	std::array<long, 2> sizes = { 3, 3 };

	ArrayDescriptor<int, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int> unsized(descriptor);

	_Midentity_ai32(unsized);

	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 3; j++)
			EXPECT_EQ(data[3 * i + j], i == j ? 1 : 0);
}

TEST(Runtime, identitySquareMatrix_i64)	 // NOLINT
{
	std::array<long, 9> data = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	std::array<long, 2> sizes = { 3, 3 };

	ArrayDescriptor<long, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<long> unsized(descriptor);

	_Midentity_ai64(unsized);

	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 3; j++)
			EXPECT_EQ(data[3 * i + j], i == j ? 1 : 0);
}

TEST(Runtime, identitySquareMatrix_f32)	 // NOLINT
{
	std::array<float, 9> data = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	std::array<long, 2> sizes = { 3, 3 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	_Midentity_af32(unsized);

	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 3; j++)
			EXPECT_EQ(data[3 * i + j], i == j ? 1 : 0);
}

TEST(Runtime, identitySquareMatrix_f64)	 // NOLINT
{
	std::array<double, 9> data = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	std::array<long, 2> sizes = { 3, 3 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	_Midentity_af64(unsized);

	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 3; j++)
			EXPECT_EQ(data[3 * i + j], i == j ? 1 : 0);
}
