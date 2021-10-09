#include <gtest/gtest.h>
#include <marco/runtime/Runtime.h>

template<typename T, unsigned int N> using ArraySizes =
		std::array<typename ArrayDescriptor<T, N>::dimension_t, N>;

TEST(Runtime, arrayDescriptor1D)
{
	std::array<int64_t, 3> data = { 0, 1, 2 };
	ArraySizes<int64_t, 1> sizes = { 3 };

	ArrayDescriptor<int64_t, 1> descriptor(data.data(), sizes);
	EXPECT_EQ(descriptor.getRank(), 1);

	for (auto [actual, expected] : llvm::zip(descriptor, data))
		EXPECT_EQ(actual, expected);
}

TEST(Runtime, arrayDescriptor2D)
{
	std::array<int64_t, 6> data = { 0, 1, 2, 3, 4, 5 };
	ArraySizes<int64_t, 2> sizes = { 3, 2 };

	ArrayDescriptor<int64_t, 2> descriptor(data.data(), sizes);
	EXPECT_EQ(descriptor.getRank(), 2);

	for (auto [actual, expected] : llvm::zip(descriptor, data))
		EXPECT_EQ(actual, expected);
}
