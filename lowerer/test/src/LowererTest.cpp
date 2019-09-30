#include "gtest/gtest.h"

#include "modelica/lowerer/Expression.hpp"

using namespace modelica;
using namespace std;

TEST(ConstantTest, construtorTest)	// NOLINT
{
	IntConstant constant(1);
	FloatConstant constant2(1.0F);
	BoolConstant constant3(false);

	EXPECT_EQ(constant.get(0), 1);
	EXPECT_EQ(constant2.get(0), 1.0F);
	EXPECT_EQ(constant3.get(0), false);
}

TEST(ExpressionTest, constantExpression)	// NOLINT
{
	Expression exp(Constant(1));
	EXPECT_TRUE(exp.isConstant<int>());
	EXPECT_TRUE(exp.isConstant());
	EXPECT_EQ(exp.getConstant<int>().get(0), 1);
}

TEST(ExpressionTest, negateExpression)	// NOLINT
{
	Expression exp(Constant(1));
	auto exp4 = exp;
	EXPECT_TRUE(exp == exp4);
	auto exp2 = Expression::negate(std::move(exp));

	EXPECT_TRUE(exp2.isOperation());
	EXPECT_EQ(exp2.getKind(), ExpressionKind::negate);
	EXPECT_TRUE(exp2.getLeftHand().isConstant());

	auto exp3 = !exp2;
	EXPECT_TRUE(exp3.isOperation());
	EXPECT_EQ(exp3.getKind(), ExpressionKind::negate);
	EXPECT_FALSE(exp3.getLeftHand().isConstant());
	EXPECT_TRUE(exp3.getLeftHand().getLeftHand().isConstant());
}

TEST(ConstantTest, dumpConstant)	// NOLINT
{
	IntConstant constant(1);
	FloatConstant constant2(1.0F);
	BoolConstant constant3(false);

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
