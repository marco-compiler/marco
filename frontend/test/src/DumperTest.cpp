#include "gtest/gtest.h"

#include "modelica/Dumper/Dumper.hpp"
#include "modelica/Parser.hpp"

using namespace modelica;
TEST(dumperTest, emptyClass)
{
	auto parser = Parser("package test end test");

	auto eq = parser.classDefinition();
	if (!eq)
		FAIL();

	std::string out;
	llvm::raw_string_ostream stringStream(out);
	dump(move(*eq), stringStream);
	stringStream.str();

	EXPECT_EQ(
			out,
			"Long Class Declaration\nClass Decl test pure Subtype: Package\n "
			"Composition\n  Composition Section \n  Composition Section \n  "
			"Composition Section \n");
}
