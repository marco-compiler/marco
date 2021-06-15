#include <gtest/gtest.h>
#include <modelica/runtime/Runtime.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>

TEST(Runtime, clone_ai1_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { true, true, true, true, true, true };
	std::array<bool, 6> destination = { false, false, false, false, false, false };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (bool) source);
}

TEST(Runtime, clone_ai1_ai32)	 // NOLINT
{
	std::array<int, 6> source = { 1, 1, 1, 1, 1, 1 };
	std::array<bool, 6> destination = { false, false, false, false, false, false };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(int))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (bool) source);
}

TEST(Runtime, clone_ai1_ai64)	 // NOLINT
{
	std::array<long, 6> source = { 1, 1, 1, 1, 1, 1 };
	std::array<bool, 6> destination = { false, false, false, false, false, false };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(long))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (bool) source);
}

TEST(Runtime, clone_ai1_af32)	 // NOLINT
{
	std::array<float, 6> source = { 1, 1, 1, 1, 1, 1 };
	std::array<bool, 6> destination = { false, false, false, false, false, false };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(float))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (bool) source);
}

TEST(Runtime, clone_ai1_af64)	 // NOLINT
{
	std::array<double, 6> source = { 1, 1, 1, 1, 1, 1 };
	std::array<bool, 6> destination = { false, false, false, false, false, false };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(double))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (bool) source);
}

TEST(Runtime, clone_ai32_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { true, true, true, true, true, true };
	std::array<int, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int) source);
}

TEST(Runtime, clone_ai32_ai32)	 // NOLINT
{
	std::array<int, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int), ARRAY(int))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int) source);
}

TEST(Runtime, clone_ai32_ai64)	 // NOLINT
{
	std::array<long, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int), ARRAY(long))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int) source);
}

TEST(Runtime, clone_ai32_af32)	 // NOLINT
{
	std::array<float, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int), ARRAY(float))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int) source);
}

TEST(Runtime, clone_ai32_af64)	 // NOLINT
{
	std::array<double, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<int, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(int), ARRAY(double))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (int) source);
}

TEST(Runtime, clone_ai64_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { true, true, true, true, true, true };
	std::array<long, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(long), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (long) source);
}

TEST(Runtime, clone_ai64_ai32)	 // NOLINT
{
	std::array<int, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<long, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(long), ARRAY(int))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (long) source);
}

TEST(Runtime, clone_ai64_ai64)	 // NOLINT
{
	std::array<long, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<long, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(long), ARRAY(long))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (long) source);
}

TEST(Runtime, clone_ai64_af32)	 // NOLINT
{
	std::array<float, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<long, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(long), ARRAY(float))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (long) source);
}

TEST(Runtime, clone_ai64_af64)	 // NOLINT
{
	std::array<double, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<long, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<long, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(long), ARRAY(double))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_EQ(destination, (long) source);
}

TEST(Runtime, clone_af32_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { true, true, true, true, true, true };
	std::array<float, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(float), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_FLOAT_EQ(destination, (float) source);
}

TEST(Runtime, clone_af32_ai32)	 // NOLINT
{
	std::array<int, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<float, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(float), ARRAY(int))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_FLOAT_EQ(destination, (float) source);
}

TEST(Runtime, clone_af32_ai64)	 // NOLINT
{
	std::array<long, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<float, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(float), ARRAY(long))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_FLOAT_EQ(destination, (float) source);
}

TEST(Runtime, clone_af32_af32)	 // NOLINT
{
	std::array<float, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<float, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(float), ARRAY(float))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_FLOAT_EQ(destination, (float) source);
}

TEST(Runtime, clone_af32_af64)	 // NOLINT
{
	std::array<double, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<float, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(float), ARRAY(double))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_FLOAT_EQ(destination, (float) source);
}

TEST(Runtime, clone_af64_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { true, true, true, true, true, true };
	std::array<double, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(double), ARRAY(bool))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_DOUBLE_EQ(destination, (double) source);
}

TEST(Runtime, clone_af64_ai32)	 // NOLINT
{
	std::array<int, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<double, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(double), ARRAY(int))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_DOUBLE_EQ(destination, source);
}

TEST(Runtime, clone_af64_ai64)	 // NOLINT
{
	std::array<long, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<double, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<long, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<long> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(double), ARRAY(long))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_DOUBLE_EQ(destination, source);
}

TEST(Runtime, clone_af64_af32)	 // NOLINT
{
	std::array<float, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<double, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(double), ARRAY(float))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_DOUBLE_EQ(destination, source);
}

TEST(Runtime, clone_af64_af64)	 // NOLINT
{
	std::array<double, 6> source = { 1, 2, 3, 4, 5, 6 };
	std::array<double, 6> destination = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(clone, void, ARRAY(double), ARRAY(double))(unsizedDestination, unsizedSource);

	for (const auto& [source, destination] : llvm::zip(sourceDescriptor, destinationDescriptor))
		EXPECT_DOUBLE_EQ(destination, source);
}
