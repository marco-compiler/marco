#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "marco/model/ModEqTemplate.hpp"
#include "marco/model/ModParser.hpp"

using namespace marco;
using namespace std;

TEST(ModParserTest, contIntVectorShouldParse)
{
	auto parser = ModParser("{1, 2, 3}");

	auto vec = parser.intVector();
	if (!vec)
		FAIL();

	auto constVector = *vec;

	EXPECT_EQ(constVector.size(), 3);
	EXPECT_EQ(constVector.get<long>(0), 1);
	EXPECT_EQ(constVector.get<long>(1), 2);
	EXPECT_EQ(constVector.get<long>(2), 3);
}

TEST(ModParserTest, contFloatVectorShouldParse)
{
	auto parser = ModParser("{1.4, 2.1, 3.9}");

	auto vec = parser.floatVector();
	if (!vec)
		FAIL();

	auto constVector = *vec;

	EXPECT_EQ(constVector.size(), 3);
	EXPECT_NEAR(constVector.get<double>(0), 1.4f, 0.1f);
	EXPECT_NEAR(constVector.get<double>(1), 2.1f, 0.1f);
	EXPECT_NEAR(constVector.get<double>(2), 3.9f, 0.1f);
}

TEST(ModParserTest, contBoolVectorShouldParse)
{
	auto parser = ModParser("{1, 2, 0}");

	auto vec = parser.boolVector();
	if (!vec)
		FAIL();

	auto constVector = *vec;

	EXPECT_EQ(constVector.size(), 3);
	EXPECT_EQ(constVector.get<bool>(0), true);
	EXPECT_EQ(constVector.get<bool>(1), true);
	EXPECT_EQ(constVector.get<bool>(2), false);
}

TEST(ModParserTest, constExp)
{
	auto parser = ModParser("INT[1]{4, 1, 9}");

	auto vec = parser.expression();
	if (!vec)
		FAIL();

	auto exp = *vec;
	EXPECT_TRUE(exp.isConstant<int>());

	auto& constant = exp.getConstant();

	EXPECT_EQ(constant.size(), 3);
	EXPECT_EQ(constant.get<long>(0), 4);
	EXPECT_EQ(constant.get<long>(1), 1);
	EXPECT_EQ(constant.get<long>(2), 9);
}

TEST(ModParserTest, simCall)
{
	auto parser = ModParser("call fun INT[1](INT[1]{1}, INT[1]{2}, INT[1]{3})");

	auto vec = parser.call();
	if (!vec)
		FAIL();

	auto call = *vec;
	EXPECT_EQ("fun", call.getName());
	EXPECT_EQ(call.getType(), ModType(BultinModTypes::INT));

	EXPECT_EQ(call.argsSize(), 3);
	EXPECT_TRUE(call.at(0).isConstant<int>());
	EXPECT_TRUE(call.at(1).isConstant<int>());
	EXPECT_TRUE(call.at(2).isConstant<int>());
}

TEST(ModParserTest, simCallExp)
{
	auto parser =
			ModParser("FLOAT[1] call fun INT[1](INT[1]{1}, INT[1]{2}, INT[1]{3})");

	auto vec = parser.expression();
	if (!vec)
		FAIL();

	auto exp = *vec;
	EXPECT_TRUE(exp.isCall());
	auto& call = exp.getCall();
	EXPECT_EQ("fun", call.getName());
	EXPECT_EQ(call.getType(), ModType(BultinModTypes::INT));

	EXPECT_EQ(call.argsSize(), 3);
	EXPECT_TRUE(call.at(0).isConstant<int>());
	EXPECT_TRUE(call.at(1).isConstant<int>());
	EXPECT_TRUE(call.at(2).isConstant<int>());
}

TEST(ModParserTest, simRefExp)
{
	auto parser = ModParser("FLOAT[1] ref");

	auto vec = parser.expression();
	if (!vec)
		FAIL();

	auto exp = *vec;
	EXPECT_TRUE(exp.isReference());
	EXPECT_EQ("ref", exp.getReference());
}

TEST(ModParserTest, simOperation)
{
	auto parser = ModParser("FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.expression();
	if (!vec)
		FAIL();

	auto exp = *vec;
	EXPECT_TRUE(exp.isOperation());
	EXPECT_EQ(ModExpKind::add, exp.getKind());
}

TEST(ModParserTest, statement)
{
	auto parser = ModParser("id = FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.statement();
	if (!vec)
		FAIL();

	auto [name, exp] = *vec;
	EXPECT_EQ("id", name);
	EXPECT_TRUE(exp.isOperation());
	EXPECT_EQ(ModExpKind::add, exp.getKind());
}

TEST(ModParserTest, forUpdateStatement)
{
	auto parser =
			ModParser("for [1,3][1,4]id = FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.updateStatement({});
	if (!vec)
		FAIL();

	EXPECT_EQ("id", vec->getLeft().getReference());
	EXPECT_TRUE(vec->getRight().isOperation());
	EXPECT_EQ(ModExpKind::add, vec->getRight().getKind());
	EXPECT_EQ(vec->getInductions().at(0).min(), 1);
	EXPECT_EQ(vec->getInductions().at(0).max(), 3);
	EXPECT_EQ(vec->getInductions().at(1).min(), 1);
	EXPECT_EQ(vec->getInductions().at(1).max(), 4);
}

TEST(ModParserTest, sectionStatement)
{
	auto parser = ModParser("init id = FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.initSection();
	if (!vec)
		FAIL();

	EXPECT_TRUE(vec->find("id") != vec->end());
}

TEST(ModParserTest, updateSection)
{
	auto parser = ModParser("update id = FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.updateSection();
	if (!vec)
		FAIL();

	EXPECT_TRUE(vec.get()[0].getLeft().getReference() == "id");
}

TEST(ModParserTest, backwardUpdate)
{
	auto parser = ModParser("backward id = FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.updateStatement({});
	if (!vec)
		FAIL();

	EXPECT_TRUE(vec.get().getLeft().getReference() == "id");
	EXPECT_FALSE(vec.get().isForward());
}

TEST(ModParserTest, templateSection)
{
	auto parser =
			ModParser("template t1 INT[1] id = FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.templates();
	if (!vec)
		FAIL();

	EXPECT_EQ((*vec.get().find("t1")).second->getLeft().getReference(), "id");
}

TEST(ModParserTest, templatedEquation)
{
	auto parser = ModParser("template t1 INT[1] id = FLOAT[1] (+ INT[1]{1}, "
													"INT[1]{2}) update template t1");

	auto vec = parser.templates();
	if (!vec)
		FAIL();

	auto updates = parser.updateSection(*vec);
	if (!updates)
		FAIL();

	EXPECT_EQ(updates.get()[0].getLeft().getReference(), "id");
}

TEST(ModParserTest, simulation)
{
	auto parser = ModParser("init id = FLOAT[1] (+ INT[1]{1}, INT[1]{2}) update "
													"id = FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.simulation();
	if (!vec)
		FAIL();

	auto model = move(*vec);

	EXPECT_TRUE(model.getVars().find("id") != model.getVars().end());
	EXPECT_TRUE(model.getEquation(0).getLeft().getReference() == "id");
	EXPECT_TRUE(model.getVar("id").getInit() == model.getEquation(0).getRight());
}
