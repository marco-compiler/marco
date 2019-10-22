#include "gtest/gtest.h"

#include "modelica/Parser.hpp"
#include "modelica/omcToModel/OmcToModelPass.hpp"

using namespace llvm;
using namespace std;
using namespace modelica;

TEST(OmcToModelTest, singleDeclaration)	 // NOLINT
{
	Parser parser("class C final parameter Integer N = 10; end C;");

	auto expectedAST = parser.classDefinition();
	if (!expectedAST)
		FAIL();

	UniqueDecl ast = move(*expectedAST);

	EntryModel model;
	OmcToModelPass pass(model);
	ast = topDownVisit(move(ast), pass);

	const auto& initialization = model.getVar("N").getInit();
	EXPECT_TRUE(initialization.isConstant());
	const auto& constant = initialization.getConstant<int>();
	EXPECT_EQ(constant.get(0), 10);
}

TEST(OmcToModelTest, uninitializedDeclaration)	// NOLINT
{
	Parser parser(
			"class C final parameter Real[10, 10] Qb(unit = \"W\"); end C;");

	auto expectedAST = parser.classDefinition();
	if (!expectedAST)
		FAIL();

	UniqueDecl ast = move(*expectedAST);

	EntryModel model;
	OmcToModelPass pass(model);
	ast = topDownVisit(move(ast), pass);

	const auto& var = model.getVar("Qb");
	const auto& initialization = var.getInit();
	EXPECT_EQ(
			initialization.getModType(), ModType(BultinModTypes::FLOAT, 10, 10));
	EXPECT_TRUE(initialization.isCall());
}

TEST(OmcToModelTest, startDeclaration)	// NOLINT
{
	Parser parser("class C Real[10, 10, 4] T(start = 313.15); end C;");

	auto expectedAST = parser.classDefinition();
	if (!expectedAST)
		FAIL();

	UniqueDecl ast = move(*expectedAST);

	EntryModel model;
	OmcToModelPass pass(model);
	ast = topDownVisit(move(ast), pass);

	const auto& var = model.getVar("T");
	const auto& initialization = var.getInit();
	EXPECT_EQ(
			initialization.getModType(), ModType(BultinModTypes::FLOAT, 10, 10, 4));
	EXPECT_TRUE(initialization.isCall());
}
