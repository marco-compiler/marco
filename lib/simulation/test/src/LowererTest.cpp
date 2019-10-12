#include "gtest/gtest.h"

#include "modelica/simulation/SimExp.hpp"

using namespace modelica;
using namespace std;

TEST(ConstantTest, construtorTest)	// NOLINT
{
	IntSimConst constant(1);
	FloatSimConst constant2(1.0F);
	BoolSimConst constant3(false);

	EXPECT_EQ(constant.get(0), 1);
	EXPECT_EQ(constant2.get(0), 1.0F);
	EXPECT_EQ(constant3.get(0), false);
}

TEST(ExpressionTest, constantExpression)	// NOLINT
{
	SimExp exp(SimConst(1));
	EXPECT_TRUE(exp.isConstant<int>());
	EXPECT_TRUE(exp.isConstant());
	EXPECT_EQ(exp.getConstant<int>().get(0), 1);
}

TEST(ExpressionTest, negateExpression)	// NOLINT
{
	SimExp exp(SimConst(1));
	auto exp4 = exp;
	EXPECT_TRUE(exp == exp4);
	auto exp2 = SimExp::negate(std::move(exp));

	EXPECT_TRUE(exp2.isOperation());
	EXPECT_EQ(exp2.getKind(), SimExpKind::negate);
	EXPECT_TRUE(exp2.getLeftHand().isConstant());

	auto exp3 = !exp2;
	EXPECT_TRUE(exp3.isOperation());
	EXPECT_EQ(exp3.getKind(), SimExpKind::negate);
	EXPECT_FALSE(exp3.getLeftHand().isConstant());
	EXPECT_TRUE(exp3.getLeftHand().getLeftHand().isConstant());
}

TEST(ConstantTest, dumpConstant)	// NOLINT
{
	IntSimConst constant(1);
	FloatSimConst constant2(1.0F);
	BoolSimConst constant3(false);

	std::string intString;
	llvm::raw_string_ostream intStream(intString);

	dumpConstant(constant, intStream);
	intStream.str();

	EXPECT_EQ(intString, "1");

	std::string floatString;
	llvm::raw_string_ostream floatStream(floatString);

	dumpConstant(constant2, floatStream);
	floatStream.str();

	float val = std::stof(floatString);
	EXPECT_NEAR(val, 1.0F, 0.1F);

	std::string boolString;
	llvm::raw_string_ostream boolStream(boolString);

	dumpConstant(constant3, boolStream);
	boolStream.str();

	EXPECT_EQ(boolString, "0");
}

TEST(ExpressionTest, operatorGreaterShouldReturnBool)	 // NOLINT
{
	auto exp = SimExp(SimConst<int>(3)) > SimExp(SimConst<int>(4));
	EXPECT_EQ(exp.getSimType(), SimType(BultinSimTypes::BOOL));
}

TEST(ExpressionTest, ternaryExp)	// NOLINT
{
	auto cond = SimExp::cond(
			SimExp("leftHand", BultinSimTypes::INT) >
					SimExp("rightHand", BultinSimTypes::INT),
			SimExp(SimConst<int>(1)),
			SimExp(SimConst<int>(9)));

	EXPECT_EQ(cond.isTernary(), true);
	EXPECT_EQ(cond.getCondition().getSimType(), SimType(BultinSimTypes::BOOL));
	EXPECT_EQ(cond.getCondition().isOperation(), true);
}
