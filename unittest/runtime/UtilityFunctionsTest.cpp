#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"
#include "marco/runtime/UtilityFunctions.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

TEST(Runtime, clone_ai1_ai1)
{
	std::array<bool, 6> source = { true, true, true, true, true, true };
	std::array<bool, 6> destination = { false, false, false, false, false, false };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (bool) source);
}

TEST(Runtime, clone_ai1_ai32)
{
	std::array<int32_t, 6> source = { 1, 1, 1, 1, 1, 1 };
	std::array<bool, 6> destination = { false, false, false, false, false, false };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (bool) source);
}

TEST(Runtime, clone_ai1_ai64)
{
	std::array<int64_t, 6> source = { 1, 1, 1, 1, 1, 1 };
	std::array<bool, 6> destination = { false, false, false, false, false, false };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (bool) source);
}

TEST(Runtime, clone_ai1_af32)
{
	std::array<float, 6> source = { 1, 1, 1, 1, 1, 1 };
	std::array<bool, 6> destination = { false, false, false, false, false, false };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (bool) source);
}

TEST(Runtime, clone_ai1_af64)
{
	std::array<double, 6> source = { 1, 1, 1, 1, 1, 1 };
	std::array<bool, 6> destination = { false, false, false, false, false, false };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (bool) source);
}

TEST(Runtime, clone_ai32_ai1)
{
	std::array<bool, 6> source = { true, true, true, true, true, true };
	std::array<int32_t, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int32_t), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int32_t) source);
}

TEST(Runtime, clone_ai32_ai32)
{
	std::array<int32_t, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int32_t, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int32_t), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int32_t) source);
}

TEST(Runtime, clone_ai32_ai64)
{
	std::array<int64_t, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int32_t, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int32_t), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int32_t) source);
}

TEST(Runtime, clone_ai32_af32)
{
	std::array<float, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int32_t, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int32_t), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int32_t) source);
}

TEST(Runtime, clone_ai32_af64)
{
	std::array<double, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int32_t, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int32_t), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int32_t) source);
}

TEST(Runtime, clone_ai64_ai1)
{
	std::array<bool, 6> source = { true, true, true, true, true, true };
	std::array<int64_t, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int64_t), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int64_t) source);
}

TEST(Runtime, clone_ai64_ai32)
{
	std::array<int32_t, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int64_t, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int64_t), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int64_t) source);
}

TEST(Runtime, clone_ai64_ai64)
{
	std::array<int64_t, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int64_t, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int64_t), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int64_t) source);
}

TEST(Runtime, clone_ai64_af32)
{
	std::array<float, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int64_t, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int64_t), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int64_t) source);
}

TEST(Runtime, clone_ai64_af64)
{
	std::array<double, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int64_t, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int64_t), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int64_t) source);
}

TEST(Runtime, clone_af32_ai1)
{
	std::array<bool, 6> source = { true, true, true, true, true, true };
	std::array<float, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(float), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_FLOAT_EQ(destination, (float) source);
}

TEST(Runtime, clone_af32_ai32)
{
	std::array<int32_t, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<float, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(float), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_FLOAT_EQ(destination, (float) source);
}

TEST(Runtime, clone_af32_ai64)
{
	std::array<int64_t, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<float, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(float), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_FLOAT_EQ(destination, (float) source);
}

TEST(Runtime, clone_af32_af32)
{
	std::array<float, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<float, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(float), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_FLOAT_EQ(destination, (float) source);
}

TEST(Runtime, clone_af32_af64)
{
	std::array<double, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<float, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(float), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_FLOAT_EQ(destination, (float) source);
}

TEST(Runtime, clone_af64_ai1)
{
	std::array<bool, 6> source = { true, true, true, true, true, true };
	std::array<double, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(double), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_DOUBLE_EQ(destination, (double) source);
}

TEST(Runtime, clone_af64_ai32)
{
	std::array<int32_t, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<double, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(double), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_DOUBLE_EQ(destination, source);
}

TEST(Runtime, clone_af64_ai64)
{
	std::array<int64_t, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<double, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(double), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_DOUBLE_EQ(destination, source);
}

TEST(Runtime, clone_af64_af32)
{
	std::array<float, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<double, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(double), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_DOUBLE_EQ(destination, source);
}

TEST(Runtime, clone_af64_af64)
{
	std::array<double, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<double, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(double), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_DOUBLE_EQ(destination, source);
}

TEST(Runtime, print_bool)
{
	NAME_MANGLED(print, void, bool)(1.0);
}

TEST(Runtime, print_float)
{
	NAME_MANGLED(print, void, float)(1.0);
}

TEST(Runtime, print_double)
{
	NAME_MANGLED(print, void, double)(1.0);
}

TEST(Runtime, print_int32)
{
	NAME_MANGLED(print, void, int32_t)(1.0);
}

TEST(Runtime, print_int64)
{
	NAME_MANGLED(print, void, int64_t)(1.0);
}

TEST(Runtime, print_array_double)
{
	std::array<double, 6> array = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<double, 2> sourceDescriptor(array.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);
	NAME_MANGLED(print, void, ARRAY(double))(&unsizedSource);
}

