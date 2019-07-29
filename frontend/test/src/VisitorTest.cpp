#include "gtest/gtest.h"

#include "modelica/AST/Visitor.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

struct Visitor
{
	template<typename T>
	unique_ptr<T> visit(std::unique_ptr<T>)
	{
		failed = true;
		return nullptr;
	}
	unique_ptr<IntLiteralExpr> visit(std::unique_ptr<IntLiteralExpr> expr)
	{
		numOfCalls++;
		content += expr->getValue();
		return expr;
	}

	unique_ptr<ExprList> visit(std::unique_ptr<ExprList> expr)
	{
		listVisited = true;

		return expr;
	}

	unique_ptr<WhenEquation> visit(std::unique_ptr<WhenEquation> expr)
	{
		whenVisited = true;

		return expr;
	}
	unique_ptr<SimpleEquation> visit(std::unique_ptr<SimpleEquation> expr)
	{
		simpleVisited = true;

		return expr;
	}

	int numOfCalls{ 0 };
	int content{ 0 };
	bool failed{ false };
	bool listVisited{ false };
	bool whenVisited{ false };
	bool simpleVisited{ false };
};
TEST(VisitorTest, simpleVisitation)
{
	auto b = std::make_unique<IntLiteralExpr>(SourceRange(1, 1, 1, 1), 3);
	Visitor v;
	auto newPtr = topDownVisit(move(b), v);
	EXPECT_EQ(v.numOfCalls, 1);
	EXPECT_EQ(v.content, 3);
	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(newPtr));
	EXPECT_EQ(false, v.failed);
	EXPECT_NE(nullptr, newPtr);

	std::unique_ptr<Expr> ptr = move(newPtr);
	v = Visitor();
	ptr = topDownVisit(move(ptr), v);
	EXPECT_EQ(v.numOfCalls, 1);
	EXPECT_EQ(v.content, 3);
	EXPECT_EQ(false, v.failed);
}

TEST(VisitorTest, listExpr)
{
	SourceRange r(1, 1, 1, 1);
	auto exp1 = std::make_unique<IntLiteralExpr>(r, 3);
	auto exp2 = std::make_unique<IntLiteralExpr>(r, 3);
	vectorUnique<Expr> v;
	v.push_back(move(exp1));
	v.push_back(move(exp2));
	auto exp3 = std::make_unique<ExprList>(r, move(v));
	EXPECT_EQ(false, llvm::isa<BoolLiteralExpr>(exp3));
	Visitor vis;
	auto ptr = topDownVisit(move(exp3), vis);
	EXPECT_EQ(vis.numOfCalls, 2);
	EXPECT_EQ(vis.content, 6);
	EXPECT_EQ(vis.listVisited, true);
	EXPECT_EQ(false, vis.failed);
}

TEST(VisitorTest, equationVisitor)
{
	SourceRange r(1, 1, 1, 1);
	auto exp1 = std::make_unique<IntLiteralExpr>(r, 3);
	auto exp2 = std::make_unique<IntLiteralExpr>(r, 3);
	auto exp3 = std::make_unique<IntLiteralExpr>(r, 3);
	auto eq1 = std::make_unique<SimpleEquation>(r, move(exp2), move(exp3));
	vectorUnique<Equation> eq;
	vectorUnique<Expr> exp;
	eq.push_back(move(eq1));
	exp.push_back(move(exp1));
	auto eq2 = std::make_unique<WhenEquation>(r, move(exp), move(eq));
	Visitor vis;
	auto ptr = topDownVisit(move(eq2), vis);
	EXPECT_EQ(vis.numOfCalls, 3);
	EXPECT_EQ(vis.content, 9);
	EXPECT_EQ(vis.listVisited, false);
	EXPECT_EQ(false, vis.failed);
	EXPECT_EQ(true, vis.whenVisited);
	EXPECT_EQ(true, vis.simpleVisited);
}
