#include "gtest/gtest.h"

#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/SymbolicDifferentiation.hpp"

using namespace modelica;

const ModExp dim = ModExp(ModConst(0, 1, 2), ModType(BultinModTypes::FLOAT, 1));
const ModVariable var = ModVariable("var", dim);
const ModExp varExp = ModExp("var", BultinModTypes::FLOAT);
const ModExp varExp2 = ModExp("var2", BultinModTypes::FLOAT);

TEST(SymbolicDifferentiationTest, DifferentiateScalarConstant)
{
	ModExp exp = ModConst(5.0);
	ModExp derivate = differentiate(exp, var);
	EXPECT_EQ(derivate, ModConst(0.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateScalarVariable)
{
	ModExp der1 = differentiate(varExp, var);
	ModExp der2 = differentiate(varExp2, var);

	EXPECT_EQ(der1, ModConst(1.0));
	EXPECT_EQ(der2, ModConst(0.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateNegate)
{
	ModExp exp1 = ModExp::negate(ModConst(5.0));
	ModExp exp2 = ModExp::negate(varExp);
	ModExp exp3 = ModExp::negate(varExp2);

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(-1.0));
	EXPECT_EQ(der3, ModConst(0.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateAddition)
{
	ModExp exp1 = ModExp::add(ModConst(3.0), ModConst(4.0));
	ModExp exp2 = ModExp::add(varExp, ModConst(5.0));
	ModExp exp3 = ModExp::add(ModConst(5.0), varExp);
	ModExp exp4 = ModExp::add(varExp, varExp);
	ModExp exp5 = ModExp::add(varExp, varExp2);
	ModExp exp6 = ModExp::add(varExp2, varExp);

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, var);
	ModExp der5 = differentiate(exp5, var);
	ModExp der6 = differentiate(exp6, var);

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(1.0));
	EXPECT_EQ(der3, ModConst(1.0));
	EXPECT_EQ(der4, ModConst(2.0));
	EXPECT_EQ(der5, ModConst(1.0));
	EXPECT_EQ(der6, ModConst(1.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateSubtraction)
{
	ModExp exp1 = ModExp::subtract(ModConst(3.0), ModConst(4.0));
	ModExp exp2 = ModExp::subtract(varExp, ModConst(5.0));
	ModExp exp3 = ModExp::subtract(ModConst(5.0), varExp);
	ModExp exp4 = ModExp::subtract(varExp, varExp);
	ModExp exp5 = ModExp::subtract(varExp, varExp2);
	ModExp exp6 = ModExp::subtract(varExp2, varExp);

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, var);
	ModExp der5 = differentiate(exp5, var);
	ModExp der6 = differentiate(exp6, var);

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(1.0));
	EXPECT_EQ(der3, ModConst(-1.0));
	EXPECT_EQ(der4, ModConst(0.0));
	EXPECT_EQ(der5, ModConst(1.0));
	EXPECT_EQ(der6, ModConst(-1.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateMultiplication)
{
	ModExp exp1 = ModExp::multiply(ModConst(3.0), ModConst(4.0));
	ModExp exp2 = ModExp::multiply(ModConst(5.0), varExp);
	ModExp exp3 = ModExp::multiply(varExp, ModConst(5.0));
	ModExp exp4 = ModExp::multiply(varExp, varExp);
	ModExp exp5 = ModExp::multiply(varExp, varExp2);
	ModExp exp6 = ModExp::multiply(varExp2, varExp);

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, var);
	ModExp der5 = differentiate(exp5, var);
	ModExp der6 = differentiate(exp5, var);

	ModExp res4 = ModExp::add(varExp, varExp);

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(5.0));
	EXPECT_EQ(der3, ModConst(5.0));
	EXPECT_EQ(der4, res4);
	EXPECT_EQ(der5, varExp2);
	EXPECT_EQ(der6, varExp2);
}

TEST(SymbolicDifferentiationTest, DifferentiateDivision)
{
	ModExp exp1 = ModExp::divide(ModConst(3.0), ModConst(4.0));
	ModExp exp2 = ModExp::divide(varExp, ModConst(5.0));
	ModExp exp3 = ModExp::divide(ModConst(5.0), varExp);
	ModExp exp4 = ModExp::divide(varExp, varExp);
	ModExp exp5 = ModExp::divide(varExp, varExp2);
	ModExp exp6 = ModExp::divide(varExp2, varExp);

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, var);
	ModExp der5 = differentiate(exp5, var);
	ModExp der6 = differentiate(exp6, var);

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
}

TEST(SymbolicDifferentiationTest, DifferentiateElevation)
{
	ModExp exp1 = ModExp::elevate(ModConst(3.0), ModConst(4));
	ModExp exp2 = ModExp::elevate(varExp, ModConst(1));
	ModExp exp3 = ModExp::elevate(varExp, ModConst(2));
	ModExp exp4 = ModExp::elevate(varExp, ModConst(3));
	ModExp exp5 = ModExp::elevate(varExp2, ModConst(5));

	ModExp der1 = differentiate(exp1, var);
	ModExp der2 = differentiate(exp2, var);
	ModExp der3 = differentiate(exp3, var);
	ModExp der4 = differentiate(exp4, var);
	ModExp der5 = differentiate(exp5, var);

	ModExp res3 = ModExp::multiply(ModConst(2.0), varExp);
	ModExp res4 =
			ModExp::multiply(ModConst(3.0), ModExp::elevate(varExp, ModConst(2.0)));

	EXPECT_EQ(der1, ModConst(0.0));
	EXPECT_EQ(der2, ModConst(1.0));
	EXPECT_EQ(der3, res3);
	EXPECT_EQ(der4, res4);
	EXPECT_EQ(der5, ModConst(0.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateVectorConstant)
{
	ModVariable constant = ModVariable("constant", dim, false, true);
	ModExp exp = ModExp("constant", BultinModTypes::FLOAT);

	ModExp derivate = differentiate(exp, var);
	EXPECT_EQ(derivate, ModConst(0.0));
}

TEST(SymbolicDifferentiationTest, DifferentiateVectorVariable)
{
	EXPECT_EQ(true, true);	// TODO
}

TEST(SymbolicDifferentiationTest, DifferentiateEquation)
{
	EXPECT_EQ(true, true);	// TODO
}
