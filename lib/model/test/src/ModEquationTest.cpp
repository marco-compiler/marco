#include "gtest/gtest.h"

#include "llvm/Support/raw_ostream.h"
#include "marco/model/Assigment.hpp"
#include "marco/model/ModConst.hpp"
#include "marco/model/ModEquation.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModType.hpp"
#include "marco/model/ModVariable.hpp"
#include "marco/utils/IndexSet.hpp"
#include "marco/utils/Interval.hpp"

using namespace marco;
using namespace llvm;
using namespace std;

TEST(ModelTest, ModEquationToIndexSet)
{
	ModExp left(ModConst(0));
	ModExp right(ModConst(0));
	Interval v0(1, 3);
	Interval v1(7, 10);
	MultiDimInterval vars{ v0, v1 };
	ModEquation eq(left, right, "", vars);

	auto res = eq.getInductions();
	EXPECT_EQ(res, MultiDimInterval({ { 1, 3 }, { 7, 10 } }));
}

TEST(ModelTest, ModEquationWithNoInductions)
{
	ModExp left(ModConst(0));
	ModExp right(ModConst(0));
	ModEquation eq(left, right);

	auto res = eq.getInductions();
	EXPECT_EQ(res, MultiDimInterval({ { 0, 1 } }));
}

TEST(ModelTest, ModVariableToIndexSet)
{
	ModExp dim(ModConst(0, 1, 2, 3), ModType(BultinModTypes::INT, 2, 2));
	ModVariable variable("var", dim);
	auto res = variable.toIndexSet();
	EXPECT_EQ(res, IndexSet({ { 0, 2 }, { 0, 2 } }));
}

TEST(ModEquationTest, ModEquationConstantFolding)
{
	ModExp l(ModConst(2));
	ModExp r(ModConst(5));

	ModExp lRes(ModConst(4));
	ModExp rRes(ModConst(10));
	MultiDimInterval vars{ { 0, 1 } };

	ModEquation eq(ModExp::add(l, l), ModExp::add(r, r), "", vars);
	eq.foldConstants();

	EXPECT_EQ(eq.getLeft(), lRes);
	EXPECT_EQ(eq.getRight(), rRes);
}

TEST(ModEquationTest, ModEquationConstantFoldingWithSub)
{
	ModExp lConst(ModConst(5));
	ModExp lVar = ModExp("lVar", ModType(BultinModTypes::FLOAT, 1));
	ModExp rVar = ModExp("rVar", ModType(BultinModTypes::FLOAT, 1));
	MultiDimInterval vars{ { 0, 1 } };

	ModEquation eq(lVar, ModExp::subtract(rVar, lConst), "", vars);
	ModExp rRes(ModExp::add(rVar, ModExp::negate(lConst)));

	eq.foldConstants();

	EXPECT_EQ(eq.getLeft(), lVar);
	EXPECT_EQ(eq.getRight(), rRes);
}

TEST(ModEquationTest, negateShouldBeInvertible)
{
	ModEquation eq(ModExp::negate(ModExp(ModConst(5))), ModExp(ModConst(4)));
	EXPECT_FALSE(eq.explicitate(0, true));
	EXPECT_EQ(eq.getLeft(), ModExp(ModConst(5)));
	EXPECT_EQ(eq.getRight(), ModExp::negate(ModExp(ModConst(4))));
}

TEST(ModEquationTest, addShouldBeInvertible)
{
	ModEquation eq(
			ModExp(ModConst(5)) + ModExp(ModConst(4)), ModExp(ModConst(4)));
	EXPECT_FALSE(eq.explicitate(0, true));
	EXPECT_EQ(eq.getLeft(), ModExp(ModConst(5)));
	EXPECT_EQ(eq.getRight(), ModExp(ModConst(4)) - ModExp(ModConst(4)));
}

TEST(ModEquationTest, multShouldBeInvertible)
{
	ModEquation eq(
			ModExp(ModConst(5)) * ModExp(ModConst(4)), ModExp(ModConst(4)));
	EXPECT_FALSE(eq.explicitate(0, true));
	EXPECT_EQ(eq.getLeft(), ModExp(ModConst(5)));
	EXPECT_EQ(eq.getRight(), ModExp(ModConst(4)) / ModExp(ModConst(4)));
}

TEST(ModEquationTest, minusFirstArgShouldBeInvertible)
{
	ModEquation eq(
			ModExp(ModConst(5)) - ModExp(ModConst(4)), ModExp(ModConst(4)));
	EXPECT_FALSE(eq.explicitate(0, true));
	EXPECT_EQ(eq.getLeft(), ModExp(ModConst(5)));
	EXPECT_EQ(eq.getRight(), ModExp(ModConst(4)) + ModExp(ModConst(4)));
}

TEST(ModEquationTest, minusSecondArgShouldBeInvertible)
{
	ModEquation eq(
			ModExp(ModConst(5)) - ModExp(ModConst(6)), ModExp(ModConst(4)));
	EXPECT_FALSE(eq.explicitate(1, true));
	EXPECT_EQ(eq.getLeft(), ModExp(ModConst(6)));
	EXPECT_EQ(eq.getRight(), ModExp(ModConst(5)) - ModExp(ModConst(4)));
}

TEST(ModEquationTest, divideSecondArgShouldBeInvertible)
{
	ModEquation eq(
			ModExp(ModConst(5)) / ModExp(ModConst(6)), ModExp(ModConst(4)));
	EXPECT_FALSE(eq.explicitate(1, true));
	EXPECT_EQ(eq.getLeft(), ModExp(ModConst(6)));
	EXPECT_EQ(eq.getRight(), ModExp(ModConst(5)) / ModExp(ModConst(4)));
}

TEST(ModEquationTest, divideFirstArgShouldBeInvertible)
{
	ModEquation eq(
			ModExp(ModConst(5)) / ModExp(ModConst(4)), ModExp(ModConst(4)));
	EXPECT_FALSE(eq.explicitate(0, true));
	EXPECT_EQ(eq.getLeft(), ModExp(ModConst(5)));
	EXPECT_EQ(eq.getRight(), ModExp(ModConst(4)) * ModExp(ModConst(4)));
}

TEST(ModEquationTest, copiedEquationShouldHaveSameTemplate)
{
	ModEquation eq(
			ModExp(ModConst(5)) / ModExp(ModConst(4)), ModExp(ModConst(4)));
	auto copy = eq;
	EXPECT_EQ(eq.getTemplate(), copy.getTemplate());
}

TEST(ModEquationTest, clonedEquationShouldNotHaveSameTemplate)
{
	ModEquation eq(
			ModExp(ModConst(5)) / ModExp(ModConst(4)), ModExp(ModConst(4)));
	auto copy = eq.clone("newName");
	EXPECT_NE(eq.getTemplate(), copy.getTemplate());
}

TEST(ModEquationTest, groupLeftTest)
{
	ModEquation eq(
			ModExp("hey", ModType(BultinModTypes::FLOAT, 1)),
			ModExp(ModConst(4.0)) *
					(ModExp(ModConst(0.0)) +
					 ModExp("hey", ModType(BultinModTypes::FLOAT, 1))));

	auto e = eq.groupLeftHand();
	EXPECT_EQ(e.getLeft(), eq.getLeft());
}

TEST(ModEquationTest, implicitEquations)
{
	ModExp varExp = ModExp("var", BultinModTypes::FLOAT);
	ModExp vVarExp = ModExp("var3", ModType(BultinModTypes::FLOAT, 2));
	ModExp varAcc = ModExp::at(ModExp(vVarExp), ModExp::index(ModConst(1)));

	ModEquation eq1 = ModEquation(varExp, ModConst(2));
	ModEquation eq2 = ModEquation(varAcc, ModConst(3));
	ModEquation eq3 = ModEquation(varExp, ModExp::multiply(varExp, varExp));
	ModEquation eq4 = ModEquation(varAcc, ModExp::divide(ModConst(1.0), varAcc));

	EXPECT_FALSE(eq1.isImplicit());
	EXPECT_FALSE(eq2.isImplicit());
	EXPECT_TRUE(eq3.isImplicit());
	EXPECT_TRUE(eq4.isImplicit());
}
