#include "gtest/gtest.h"

#include "marco/model/ModEquation.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/SymbolicDifferentiation.hpp"
#include "marco/model/VectorAccess.hpp"

using namespace marco;

const ModExp var = ModExp("var", BultinModTypes::FLOAT);
const ModExp varExp = ModExp(var);
const ModExp varExp2 = ModExp("var2", BultinModTypes::FLOAT);

const ModExp vectorVar = ModExp("var3", ModType(BultinModTypes::FLOAT, 2));
const ModExp vVarExp = ModExp(vectorVar);
const ModExp vAccess = ModExp::at(ModExp(vVarExp), ModExp::index(ModConst(0)));

TEST(SymbolicDifferentiationTest, DifferentiateScalarConstant)
{
	ModExp exp = ModConst(5.0);
	ModExp der = differentiate(exp, var);
	EXPECT_EQ(der, ModConst(0.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateScalarVariable)
{
	ModExp der1 = differentiate(varExp, var);
	ModExp der2 = differentiate(varExp2, var);
	ModExp der3 = differentiate(vVarExp, var);

	EXPECT_EQ(der1, ModConst(1.0));
	EXPECT_EQ(der2, ModConst(0.0));
	EXPECT_EQ(der3, ModConst(0.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateVectorConstant)
{
	ModExp dim = ModExp(ModConst(0), ModType(BultinModTypes::FLOAT, 1));
	ModVariable constant = ModVariable("constant", dim, false, true);
	ModExp exp = ModExp("constant", ModType(BultinModTypes::FLOAT, 2));

	ModExp der = differentiate(exp, var);
	EXPECT_EQ(der, ModConst(0.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateVectorVariable)
{
	ModExp vectorVarExp2 = ModExp("var4", ModType(BultinModTypes::FLOAT, 2));
	ModExp vectorAccess2 = ModExp::at(
			ModExp(vVarExp), ModExp::add(ModExp::index(ModConst(0)), ModConst(2)));

	ModExp exp1 = ModConst(5);
	ModExp exp2 = ModExp(varExp);
	ModExp exp3 = ModExp(vAccess);
	ModExp exp4 = ModExp(vectorAccess2);
	ModExp exp5 = ModExp::at(ModExp(vVarExp), ModExp::index(ModConst(2)));
	ModExp exp6 = ModExp(vectorAccess2);
	ModExp exp7 = ModExp::at(
			ModExp(vVarExp), ModExp::add(ModExp::index(ModConst(0)), ModConst(3)));

	ModExp der1 = differentiate(exp1, vAccess);
	ModExp der2 = differentiate(exp2, vAccess);
	ModExp der3 = differentiate(exp3, vAccess);
	ModExp der4 = differentiate(exp4, vAccess);
	ModExp der5 = differentiate(exp5, vAccess);
	ModExp der6 = differentiate(exp6, vectorAccess2);
	ModExp der7 = differentiate(exp7, vectorAccess2);

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(0.0));
	EXPECT_EQ(der3, ModConst(1.0));
	EXPECT_EQ(der4, ModConst(0.0));
	EXPECT_EQ(der5, ModConst(0.0));
	EXPECT_EQ(der6, ModConst(1.0));
	EXPECT_EQ(der7, ModConst(0.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateNegate)
{
	ModExp exp1 = ModExp::negate(ModConst(5.0));
	ModExp exp2 = ModExp::negate(varExp);
	ModExp exp3 = ModExp::negate(varExp2);
	ModExp exp4 = ModExp::negate(vAccess);

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, vAccess);

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(-1.0));
	EXPECT_EQ(der3, ModConst(0.0));
	EXPECT_EQ(der4, ModConst(-1.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateAddition)
{
	ModExp exp1 = ModExp::add(ModConst(3.0), ModConst(4.0));
	ModExp exp2 = ModExp::add(varExp, ModConst(5.0));
	ModExp exp3 = ModExp::add(ModConst(5.0), varExp);
	ModExp exp4 = ModExp::add(varExp, varExp);
	ModExp exp5 = ModExp::add(varExp, varExp2);
	ModExp exp6 = ModExp::add(varExp2, varExp);
	ModExp exp7 = ModExp::add(vAccess, ModConst(5.0));

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, var);
	ModExp der5 = differentiate(exp5, var);
	ModExp der6 = differentiate(exp6, var);
	ModExp der7 = differentiate(exp7, vAccess);

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(1.0));
	EXPECT_EQ(der3, ModConst(1.0));
	EXPECT_EQ(der4, ModConst(2.0));
	EXPECT_EQ(der5, ModConst(1.0));
	EXPECT_EQ(der6, ModConst(1.0));
	EXPECT_EQ(der7, ModConst(1.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateSubtraction)
{
	ModExp exp1 = ModExp::subtract(ModConst(3.0), ModConst(4.0));
	ModExp exp2 = ModExp::subtract(varExp, ModConst(5.0));
	ModExp exp3 = ModExp::subtract(ModConst(5.0), varExp);
	ModExp exp4 = ModExp::subtract(varExp, varExp);
	ModExp exp5 = ModExp::subtract(varExp, varExp2);
	ModExp exp6 = ModExp::subtract(varExp2, varExp);
	ModExp exp7 = ModExp::subtract(vAccess, ModConst(5.0));

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, var);
	ModExp der5 = differentiate(exp5, var);
	ModExp der6 = differentiate(exp6, var);
	ModExp der7 = differentiate(exp7, vAccess);

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(1.0));
	EXPECT_EQ(der3, ModConst(-1.0));
	EXPECT_EQ(der4, ModConst(0.0));
	EXPECT_EQ(der5, ModConst(1.0));
	EXPECT_EQ(der6, ModConst(-1.0));
	EXPECT_EQ(der7, ModConst(1.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateMultiplication)
{
	ModExp exp1 = ModExp::multiply(ModConst(3.0), ModConst(4.0));
	ModExp exp2 = ModExp::multiply(ModConst(5.0), varExp);
	ModExp exp3 = ModExp::multiply(varExp, ModConst(5.0));
	ModExp exp4 = ModExp::multiply(varExp, varExp);
	ModExp exp5 = ModExp::multiply(varExp, varExp2);
	ModExp exp6 = ModExp::multiply(varExp2, varExp);
	ModExp exp7 = ModExp::multiply(vAccess, ModConst(7.0));

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, var);
	ModExp der5 = differentiate(exp5, var);
	ModExp der6 = differentiate(exp6, var);
	ModExp der7 = differentiate(exp7, vAccess);

	ModExp res4 = ModExp::add(varExp, varExp);

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(5.0));
	EXPECT_EQ(der3, ModConst(5.0));
	EXPECT_EQ(der4, res4);
	EXPECT_EQ(der5, varExp2);
	EXPECT_EQ(der6, varExp2);
	EXPECT_EQ(der7, ModConst(7.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateDivision)
{
	ModExp exp1 = ModExp::divide(ModConst(3.0), ModConst(4.0));
	ModExp exp2 = ModExp::divide(varExp, ModConst(5.0));
	ModExp exp3 = ModExp::divide(ModConst(5.0), varExp);
	ModExp exp4 = ModExp::divide(varExp, varExp);
	ModExp exp5 = ModExp::divide(varExp, varExp2);
	ModExp exp6 = ModExp::divide(varExp2, varExp);
	ModExp exp7 = ModExp::divide(vAccess, ModConst(5.0));

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, var);
	ModExp der5 = differentiate(exp5, var);
	ModExp der6 = differentiate(exp6, var);
	ModExp der7 = differentiate(exp7, vAccess);

	ModExp res3 =
			ModExp::divide(ModConst(-5.0), ModExp::multiply(varExp, varExp));
	ModExp res4 = ModExp::divide(
			ModExp::subtract(varExp, varExp), ModExp::multiply(varExp, varExp));
	ModExp res5 = ModExp::divide(varExp2, ModExp::multiply(varExp2, varExp2));
	ModExp res6 =
			ModExp::divide(ModExp::negate(varExp2), ModExp::multiply(varExp, varExp));

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(0.2));
	EXPECT_EQ(der3, res3);
	EXPECT_EQ(der4, res4);
	EXPECT_EQ(der5, res5);
	EXPECT_EQ(der6, res6);
	EXPECT_EQ(der7, ModConst(0.2));
}

TEST(SymbolicDifferentiationTest, DifferentiateElevation)
{
	ModExp exp1 = ModExp::elevate(ModConst(3.0), ModConst(4));
	ModExp exp2 = ModExp::elevate(varExp, ModConst(1));
	ModExp exp3 = ModExp::elevate(varExp, ModConst(2));
	ModExp exp4 = ModExp::elevate(varExp, ModConst(3));
	ModExp exp5 = ModExp::elevate(varExp2, ModConst(5));
	ModExp exp6 = ModExp::elevate(vAccess, ModConst(2));

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, var);
	ModExp der5 = differentiate(exp5, var);
	ModExp der6 = differentiate(exp6, vAccess);

	ModExp res3 = ModExp::multiply(ModConst(2.0), varExp);
	ModExp res4 =
			ModExp::multiply(ModConst(3.0), ModExp::elevate(varExp, ModConst(2.0)));
	ModExp res6 = ModExp::multiply(ModConst(2.0), vAccess);

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(1.0));
	EXPECT_EQ(der3, res3);
	EXPECT_EQ(der4, res4);
	EXPECT_EQ(der5, ModConst(0.0));
	EXPECT_EQ(der6, res6);
}

TEST(SymbolicDifferentiationTest, DifferentiateInduction)
{
	ModExp exp1 = ModExp::induction(ModConst(3));
	ModExp exp2 = ModExp::multiply(varExp, ModExp::induction(ModConst(4)));
	ModExp exp3 = ModExp::add(varExp2, ModExp::induction(ModConst(5)));
	ModExp exp4 = ModExp::multiply(ModExp::induction(ModConst(6)), vAccess);

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, vAccess);

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModExp::induction(ModConst(4)));
	EXPECT_EQ(der3, ModConst(0.0));
	EXPECT_EQ(der4, ModExp::induction(ModConst(6)));
}

TEST(SymbolicDifferentiationTest, DifferentiateEquation)
{
	ModExp left = ModExp::add(
			ModExp::elevate(varExp, ModConst(3)), ModExp::divide(varExp, varExp2));
	ModExp right = ModExp::subtract(
			ModExp::multiply(varExp2, varExp), ModExp::add(vAccess, varExp));

	ModEquation eq = ModEquation(left, right);
	ModEquation der = differentiate(eq, var);

	EXPECT_EQ(der.getLeft(), differentiate(left, var));
	EXPECT_EQ(der.getRight(), differentiate(right, var));
}

TEST(SymbolicDifferentiationTest, DifferentiateMultiDimVariables)
{
	ModExp multiVarExp = ModExp("var4", ModType(BultinModTypes::FLOAT, 2, 2));
	ModExp multiDimAcc = ModExp::at(
			ModExp::at(ModExp(multiVarExp), ModExp::index(ModConst(1))),
			ModExp::index(ModConst(0)));

	ModExp der1 = differentiate(multiDimAcc, multiDimAcc);
	ModExp der2 = differentiate(multiDimAcc, vAccess);
	ModExp der3 = differentiate(vAccess, multiDimAcc);

	EXPECT_EQ(der1, ModConst(1.0));
	EXPECT_EQ(der2, ModConst(0.0));
	EXPECT_EQ(der3, ModConst(0.0));
}
