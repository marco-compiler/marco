#include "gtest/gtest.h"

#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModCall.hpp"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModExp.hpp"

using namespace modelica;
using namespace std;

TEST(ConstantTest, construtorTest)	// NOLINT
{
	ModConst constant(1);
	ModConst constant2(1.0F);
	ModConst constant3(false);

	EXPECT_EQ(constant.get<int>(0), 1);
	EXPECT_EQ(constant2.get<float>(0), 1.0F);
	EXPECT_EQ(constant3.get<bool>(0), false);
}

TEST(ExpressionTest, constantExpression)	// NOLINT
{
	ModExp exp(ModConst(1));
	EXPECT_TRUE(exp.isConstant<int>());
	EXPECT_TRUE(exp.isConstant());
	EXPECT_EQ(exp.getConstant().get<int>(0), 1);
}

TEST(ExpressionTest, negateExpression)	// NOLINT
{
	ModExp exp(ModConst(1));
	auto exp4 = exp;
	EXPECT_TRUE(exp == exp4);
	auto exp2 = ModExp::negate(std::move(exp));

	EXPECT_TRUE(exp2.isOperation());
	EXPECT_EQ(exp2.getKind(), ModExpKind::negate);
	EXPECT_TRUE(exp2.getLeftHand().isConstant());

	auto exp3 = !exp2;
	EXPECT_TRUE(exp3.isOperation());
	EXPECT_EQ(exp3.getKind(), ModExpKind::negate);
	EXPECT_FALSE(exp3.getLeftHand().isConstant());
	EXPECT_TRUE(exp3.getLeftHand().getLeftHand().isConstant());
}

TEST(ConstantTest, dumpConstant)	// NOLINT
{
	ModConst constant(1);
	ModConst constant2(1.0F);
	ModConst constant3(false);

	std::string intString;
	llvm::raw_string_ostream intStream(intString);

	constant.dump(intStream);
	intStream.str();

	EXPECT_EQ(intString, "{1}");

	std::string boolString;
	llvm::raw_string_ostream boolStream(boolString);

	constant3.dump(boolStream);
	boolStream.str();

	EXPECT_EQ(boolString, "{0}");
}

TEST(ExpressionTest, operatorGreaterShouldReturnBool)	 // NOLINT
{
	auto exp = ModExp(ModConst(3)) > ModExp(ModConst(4));
	EXPECT_EQ(exp.getModType(), ModType(BultinModTypes::BOOL));
}

TEST(ExpressionTest, ternaryExp)	// NOLINT
{
	auto cond = ModExp::cond(
			ModExp("leftHand", BultinModTypes::INT) >
					ModExp("rightHand", BultinModTypes::INT),
			ModExp(ModConst(1)),
			ModExp(ModConst(9)));

	EXPECT_EQ(cond.isTernary(), true);
	EXPECT_EQ(cond.getCondition().getModType(), ModType(BultinModTypes::BOOL));
	EXPECT_EQ(cond.getCondition().isOperation(), true);
}

TEST(ModCallTest, testDeepEquality)	 // NOLINT
{
	ModExp ref("ref", BultinModTypes::BOOL);
	ModCall call("hey", { ref, ref }, ModType(BultinModTypes::BOOL));

	auto copy = call;

	EXPECT_EQ(call, copy);
}

TEST(ExpressionTest, callExpression)	// NOLINT
{
	ModExp ref("ref", BultinModTypes::BOOL);
	ModExp exp(ModCall("Hey", { ref, ref }, BultinModTypes::INT));

	EXPECT_EQ(exp.isCall(), true);
	EXPECT_EQ(exp.getCall().getName(), "Hey");
	EXPECT_EQ(exp.getModType(), ModType(BultinModTypes::INT));
	EXPECT_EQ(exp.getCall().argsSize(), 2);
	EXPECT_EQ(exp.getCall().at(0), ref);
}

TEST(ModExpTest, sumCanBeFolded)
{
	auto sum = ModExp(ModConst(1)) + ModExp(ModConst(3));
	EXPECT_TRUE(sum.tryFoldConstant());
	EXPECT_TRUE(sum.isConstant<int>());
	EXPECT_EQ(sum.getConstant().get<int>(0), 4);
}

TEST(ModExpTest, operatorGreaterCanBeFolded)
{
	auto sum = ModExp(ModConst(1)) > ModExp(ModConst(3));
	EXPECT_TRUE(sum.tryFoldConstant());
	EXPECT_TRUE(sum.isConstant<bool>());
	EXPECT_EQ(sum.getConstant().get<bool>(0), false);
}

TEST(ModExpTest, operatorLowerCanBeFolded)
{
	auto sum = ModExp(ModConst(1)) < ModExp(ModConst(3));
	EXPECT_TRUE(sum.tryFoldConstant());
	EXPECT_TRUE(sum.isConstant<bool>());
	EXPECT_EQ(sum.getConstant().get<bool>(0), true);
}

TEST(ModExpTest, conditionalCanBeFolded)
{
	auto sum = ModExp::cond(
			ModExp(ModConst(false)), ModExp(ModConst(1)), ModExp(ModConst(3)));
	EXPECT_TRUE(sum.tryFoldConstant());
	sum.dump();
	EXPECT_TRUE(sum.isConstant<int>());
	EXPECT_EQ(sum.getConstant().get<int>(0), 3);
}

TEST(ModelTest, entryModelIsIteratable)
{
	EntryModel model;

	for (auto& e : model)
		FAIL();

	model.emplaceEquation(
			ModExp("Hey", BultinModTypes::INT), ModExp("Huy", BultinModTypes::INT));

	for (auto& e : model)
		EXPECT_TRUE(e.getLeft().isReference());
}
