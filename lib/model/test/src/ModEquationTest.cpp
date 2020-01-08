#include "gtest/gtest.h"

#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModType.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/utils/IndexSet.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

TEST(ModelTest, ModEquationToIndexSet)
{
	ModExp left(ModConst<int>(0));
	ModExp right(ModConst<int>(0));
	InductionVar v0(1, 3);
	InductionVar v1(7, 10);
	SmallVector<InductionVar, 3> vars{ v0, v1 };
	ModEquation eq(left, right, vars);

	auto res = eq.toIndexSet();
	EXPECT_EQ(res, IndexSet({ { 1, 3 }, { 7, 10 } }));
}

TEST(ModelTest, ModVariableToIndexSet)
{
	ModExp dim(ModConst<int>(0, 1, 2, 3), ModType(BultinModTypes::INT, 2, 2));
	ModVariable variable("var", dim);
	auto res = variable.toIndexSet();
	EXPECT_EQ(res, IndexSet({ { 1, 2 }, { 1, 2 } }));
}
