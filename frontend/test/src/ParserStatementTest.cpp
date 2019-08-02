#include "gtest/gtest.h"

#include "modelica/Parser.hpp"

using namespace modelica;
TEST(ParserTest, simpleStatement)
{
	auto parser = Parser("a := 4");

	auto eq = parser.statement();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<AssignStatement>(ptr));
	auto casted = llvm::cast<AssignStatement>(ptr);
	EXPECT_EQ(true, llvm::isa<ComponentReferenceExpr>(casted->getLeftHand()));
	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(casted->getRightHand()));
}

TEST(ParserTest, callStatement)
{
	auto parser = Parser("a(args, args2)");

	auto eq = parser.statement();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<CallStatement>(ptr));
	auto casted = llvm::cast<CallStatement>(ptr);
	EXPECT_EQ(true, llvm::isa<ComponentFunctionCallExpr>(casted->getCallExpr()));
}

TEST(ParserTest, multiReturnCallStatement)
{
	auto parser = Parser("(a, b, c) := a(args, args2)");

	auto eq = parser.statement();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<AssignStatement>(ptr));
	auto casted = llvm::cast<AssignStatement>(ptr);
	EXPECT_EQ(true, llvm::isa<ComponentFunctionCallExpr>(casted->getRightHand()));
}

TEST(ParserTest, ifStatement)
{
	auto parser = Parser("if true then a := 1; end if");

	auto eq = parser.statement();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<IfStatement>(ptr));
}

TEST(ParserTest, ifElseStatement)
{
	auto parser = Parser("if true then a := 1; else a :=2 ; end if");

	auto eq = parser.statement();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<IfStatement>(ptr));
}

TEST(ParserTest, ifElseIfStatement)
{
	auto parser = Parser("if true then a := 1; elseif false then a :=2 ; end if");

	auto eq = parser.statement();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<IfStatement>(ptr));
}

TEST(ParserTest, whileStatement)
{
	auto parser = Parser("while true loop a := 1; end while");

	auto eq = parser.statement();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<WhileStatement>(ptr));
}

TEST(ParserTest, whenStatement)
{
	auto parser =
			Parser("when true then a := 1; elsewhen false then b := 2; end when");

	auto eq = parser.statement();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<WhenStatement>(ptr));
}

TEST(ParserTest, forStatement)
{
	auto parser = Parser("for a in {1, 3, 4} loop a := 1; end for");

	auto eq = parser.statement();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();
	EXPECT_EQ(true, llvm::isa<ForStatement>(ptr));
}
