#include <gtest/gtest.h>
#include <modelica/runtime/Runtime.h>

TEST(Runtime, arrayDescriptor1D)	 // NOLINT
{
	std::array<long, 3> data = { 0, 1, 2 };
	std::array<long, 1> sizes = { 3 };

	ArrayDescriptor<long, 1> descriptor(data.data(), sizes);
	EXPECT_EQ(descriptor.getRank(), 1);

	for (auto [ actual, expected ] : llvm::zip(descriptor, data))
		EXPECT_EQ(actual, expected);
}

TEST(Runtime, arrayDescriptor2D)	 // NOLINT
{
	std::array<long, 6> data = { 0, 1, 2, 3, 4, 5 };
	std::array<long, 2> sizes = { 3, 2 };

	ArrayDescriptor<long, 2> descriptor(data.data(), sizes);
	EXPECT_EQ(descriptor.getRank(), 2);

	for (auto [ actual, expected ] : llvm::zip(descriptor, data))
		EXPECT_EQ(actual, expected);
}
