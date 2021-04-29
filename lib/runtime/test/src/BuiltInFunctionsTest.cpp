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
	_Mfill_i1(unsized, value);

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
	_Mfill_i32(unsized, value);

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
	_Mfill_i64(unsized, value);

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
	_Mfill_f32(unsized, value);

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
	_Mfill_f64(unsized, value);

	for (const auto& element : data)
		EXPECT_EQ(value, element);
}
