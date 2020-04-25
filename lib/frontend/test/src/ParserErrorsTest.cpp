#include "gtest/gtest.h"

#include "modelica/frontend/ParserErrors.hpp"

using namespace modelica;

TEST(ParserErrorsTest, errorCodeShouldReturnCorrectMessage)
{
	std::error_code err(2, ParserErrorCategory::category);
	EXPECT_EQ("Unexpected Token", err.message());
	std::error_code err2(1, ParserErrorCategory::category);
	EXPECT_EQ("Not Implemented", err2.message());
}

llvm::Expected<int> returnError()
{
	return llvm::make_error<NotImplemented>("test");
}

TEST(ParserErrorsTest, llvmExpectedShouldBeCompatible)
{
	if (auto f = returnError())
		FAIL();
	else
	{
		llvm::handleAllErrors(f.takeError(), [](const NotImplemented&) {});
	}
}
