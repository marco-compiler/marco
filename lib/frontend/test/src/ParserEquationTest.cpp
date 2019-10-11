#include "gtest/gtest.h"

#include "modelica/Parser.hpp"

using namespace modelica;
TEST(ParserTest, simpleEquation)
{
	auto parser = Parser("2 = \"s\"");

	auto eq = parser.equation();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<SimpleEquation>(ptr));
	auto casted = llvm::cast<SimpleEquation>(ptr);
	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(casted->getLeftHand()));
	EXPECT_EQ(true, llvm::isa<StringLiteralExpr>(casted->getRightHand()));
}

TEST(ParserTest, IfEquation)
{
	auto parser = Parser("if true then 1 = 1; end if");

	auto eq = parser.equation();
	if (!eq)
	{
		llvm::outs() << eq.takeError();
		FAIL();
	}

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<IfEquation>(ptr));
	auto casted = llvm::cast<IfEquation>(ptr);
	EXPECT_EQ(false, casted->hasFinalElse());
	EXPECT_EQ(true, llvm::isa<BoolLiteralExpr>(casted->getCondition(0)));
	EXPECT_EQ(true, llvm::isa<CompositeEquation>(casted->getEquation(0)));
	EXPECT_EQ(1, casted->branchesSize());
}

TEST(ParserTest, IfEquationWithElse)
{
	auto parser = Parser("if true then 1 = 1; else 2 = 3; end if");

	auto eq = parser.equation();
	if (!eq)
	{
		llvm::outs() << eq.takeError();
		FAIL();
	}

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<IfEquation>(ptr));
	auto casted = llvm::cast<IfEquation>(ptr);
	EXPECT_EQ(true, casted->hasFinalElse());
	EXPECT_EQ(true, llvm::isa<BoolLiteralExpr>(casted->getCondition(0)));
	EXPECT_EQ(true, llvm::isa<CompositeEquation>(casted->getEquation(0)));
	EXPECT_EQ(true, llvm::isa<CompositeEquation>(casted->getEquation(1)));
	EXPECT_EQ(2, casted->branchesSize());
}

TEST(ParserTest, IfEquationWithElseIf)
{
	auto parser =
			Parser("if true then 1 = 1; elseif false then 3=4; else 2 = 3; end if");

	auto eq = parser.equation();
	if (!eq)
	{
		llvm::outs() << eq.takeError();
		FAIL();
	}

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<IfEquation>(ptr));
	auto casted = llvm::cast<IfEquation>(ptr);
	EXPECT_EQ(true, casted->hasFinalElse());
	EXPECT_EQ(true, llvm::isa<BoolLiteralExpr>(casted->getCondition(0)));
	EXPECT_EQ(true, llvm::isa<CompositeEquation>(casted->getEquation(0)));
	EXPECT_EQ(true, llvm::isa<CompositeEquation>(casted->getEquation(1)));
	EXPECT_EQ(3, casted->branchesSize());
}

TEST(ParserTest, forEquation)
{
	auto parser = Parser("for i in {1, 2,3} loop 1 = 1; end for");

	auto eq = parser.equation();
	if (!eq)
	{
		llvm::outs() << eq.takeError();
		FAIL();
	}

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<ForEquation>(ptr));
	auto casted = llvm::cast<ForEquation>(ptr);
	EXPECT_EQ(true, llvm::isa<SimpleEquation>(casted->getEquation(0)));
	EXPECT_EQ(
			true, llvm::isa<DirectArrayConstructorExpr>(casted->getForExpression(0)));
}

TEST(ParserTest, whenEquation)
{
	auto parser =
			Parser("when true then 1 = 1; elsewhen false then 2 = 2; end when");

	auto eq = parser.equation();
	if (!eq)
	{
		llvm::outs() << eq.takeError();
		FAIL();
	}

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<WhenEquation>(ptr));
	auto casted = llvm::cast<WhenEquation>(ptr);
	EXPECT_EQ(true, llvm::isa<CompositeEquation>(casted->getEquation(0)));
	EXPECT_EQ(true, llvm::isa<BoolLiteralExpr>(casted->getCondition(0)));
	EXPECT_EQ(2, casted->branchesSize());
}

TEST(ParserTest, connectClause)
{
	auto parser = Parser("connect(a, b)");

	auto eq = parser.equation();
	if (!eq)
	{
		llvm::outs() << eq.takeError();
		FAIL();
	}

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<ConnectClause>(ptr));
	auto casted = llvm::cast<ConnectClause>(ptr);
	EXPECT_EQ(true, llvm::isa<ComponentReferenceExpr>(casted->getFirstParam()));
	EXPECT_EQ(true, llvm::isa<ComponentReferenceExpr>(casted->getSecondParam()));
}

TEST(ParserTest, callEquation)
{
	auto parser = Parser("assert(a, 5)");

	auto eq = parser.equation();
	if (!eq)
	{
		llvm::outs() << eq.takeError();
		FAIL();
	}

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<CallEquation>(ptr));
	auto casted = llvm::cast<CallEquation>(ptr);
	EXPECT_EQ(true, llvm::isa<ComponentFunctionCallExpr>(casted->getCallExpr()));
}
