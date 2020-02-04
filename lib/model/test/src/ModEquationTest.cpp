#include "gtest/gtest.h"

#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModType.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/utils/IndexSet.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

TEST(ModelTest, ModEquationToIndexSet)
{
	ModExp left(ModConst(0));
	ModExp right(ModConst(0));
	InductionVar v0(1, 3);
	InductionVar v1(7, 10);
	SmallVector<InductionVar, 3> vars{ v0, v1 };
	ModEquation eq(left, right, vars);

	auto res = eq.toIndexSet();
	EXPECT_EQ(res, IndexSet({ { 1, 3 }, { 7, 10 } }));
}

TEST(ModelTest, ModVariableToIndexSet)
{
	ModExp dim(ModConst(0, 1, 2, 3), ModType(BultinModTypes::INT, 2, 2));
	ModVariable variable("var", dim);
	auto res = variable.toIndexSet();
	EXPECT_EQ(res, IndexSet({ { 1, 2 }, { 1, 2 } }));
}

TEST(ModelTest, ModEquationConstantFolding)
{
	ModExp l(ModConst(2));
	ModExp r(ModConst(5));

	ModExp lRes(ModConst(4));
	ModExp rRes(ModConst(10));
	SmallVector<InductionVar, 3> vars{ { 0, 1 } };

	ModEquation eq(ModExp::add(l, l), ModExp::add(r, r), vars);
	eq.foldConstants();

	EXPECT_EQ(eq.getLeft(), lRes);
	EXPECT_EQ(eq.getRight(), rRes);
}

TEST(ModelTest, ModEquationConstantFoldingWithReferences)
{
	ModExp l(ModConst(2));
	ModExp r(ModConst(5));
	ModExp sum("Hey", ModType(BultinModTypes::INT, 1));

	ModExp lRes(ModConst(4));
	ModExp rRes(ModConst(10));
	SmallVector<InductionVar, 3> vars{ { 0, 1 } };

	ModExp inner = ModExp::add(l, sum);
	ModEquation eq(l + inner, ModExp::add(r, r), vars);
	eq.foldConstants();

	EXPECT_EQ(eq.getLeft(), ModExp::add(sum, lRes));
	EXPECT_EQ(eq.getRight(), rRes);
}
