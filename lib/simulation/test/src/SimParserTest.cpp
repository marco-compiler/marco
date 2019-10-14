#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "modelica/simulation/SimParser.hpp"

using namespace modelica;
using namespace std;

TEST(SimParserTest, contIntVectorShouldParse)
{
	auto parser = SimParser("{1, 2, 3}");

	auto vec = parser.intVector();
	if (!vec)
		FAIL();

	auto constVector = *vec;

	EXPECT_EQ(constVector.size(), 3);
	EXPECT_EQ(constVector.get(0), 1);
	EXPECT_EQ(constVector.get(1), 2);
	EXPECT_EQ(constVector.get(2), 3);
}

TEST(SimParserTest, contFloatVectorShouldParse)
{
	auto parser = SimParser("{1.4, 2.1, 3.9}");

	auto vec = parser.floatVector();
	if (!vec)
		FAIL();

	auto constVector = *vec;

	EXPECT_EQ(constVector.size(), 3);
	EXPECT_NEAR(constVector.get(0), 1.4f, 0.1f);
	EXPECT_NEAR(constVector.get(1), 2.1f, 0.1f);
	EXPECT_NEAR(constVector.get(2), 3.9f, 0.1f);
}

TEST(SimParserTest, contBoolVectorShouldParse)
{
	auto parser = SimParser("{1, 2, 0}");

	auto vec = parser.boolVector();
	if (!vec)
		FAIL();

	auto constVector = *vec;

	EXPECT_EQ(constVector.size(), 3);
	EXPECT_EQ(constVector.get(0), true);
	EXPECT_EQ(constVector.get(1), true);
	EXPECT_EQ(constVector.get(2), false);
}

TEST(SimParserTest, constExp)
{
	auto parser = SimParser("INT[1]{4, 1, 9}");

	auto vec = parser.expression();
	if (!vec)
		FAIL();

	auto exp = *vec;
	EXPECT_TRUE(exp.isConstant<int>());

	auto& constant = exp.getConstant<int>();

	EXPECT_EQ(constant.size(), 3);
	EXPECT_EQ(constant.get(0), 4);
	EXPECT_EQ(constant.get(1), 1);
	EXPECT_EQ(constant.get(2), 9);
}

TEST(SimParserTest, simCall)
{
	auto parser = SimParser("call fun INT[1](INT[1]{1}, INT[1]{2}, INT[1]{3})");

	auto vec = parser.call();
	if (!vec)
		FAIL();

	auto call = *vec;
	EXPECT_EQ("fun", call.getName());
	EXPECT_EQ(call.getType(), SimType(BultinSimTypes::INT));

	EXPECT_EQ(call.argsSize(), 3);
	EXPECT_TRUE(call.at(0).isConstant<int>());
	EXPECT_TRUE(call.at(1).isConstant<int>());
	EXPECT_TRUE(call.at(2).isConstant<int>());
}

TEST(SimParserTest, simCallExp)
{
	auto parser =
			SimParser("FLOAT[1] call fun INT[1](INT[1]{1}, INT[1]{2}, INT[1]{3})");

	auto vec = parser.expression();
	if (!vec)
		FAIL();

	auto exp = *vec;
	EXPECT_TRUE(exp.isCall());
	auto& call = exp.getCall();
	EXPECT_EQ("fun", call.getName());
	EXPECT_EQ(call.getType(), SimType(BultinSimTypes::INT));

	EXPECT_EQ(call.argsSize(), 3);
	EXPECT_TRUE(call.at(0).isConstant<int>());
	EXPECT_TRUE(call.at(1).isConstant<int>());
	EXPECT_TRUE(call.at(2).isConstant<int>());
}

TEST(SimParserTest, simRefExp)
{
	auto parser = SimParser("FLOAT[1] ref");

	auto vec = parser.expression();
	if (!vec)
		FAIL();

	auto exp = *vec;
	EXPECT_TRUE(exp.isReference());
	EXPECT_EQ("ref", exp.getReference());
}

TEST(SimParserTest, simOperation)
{
	auto parser = SimParser("FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.expression();
	if (!vec)
		FAIL();

	auto exp = *vec;
	EXPECT_TRUE(exp.isOperation());
	EXPECT_EQ(SimExpKind::add, exp.getKind());
}

TEST(SimParserTest, statement)
{
	auto parser = SimParser("id = FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.statement();
	if (!vec)
		FAIL();

	auto [name, exp] = *vec;
	EXPECT_EQ("id", name);
	EXPECT_TRUE(exp.isOperation());
	EXPECT_EQ(SimExpKind::add, exp.getKind());
}

TEST(SimParserTest, sectionStatement)
{
	auto parser = SimParser("init id = FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.initSection();
	if (!vec)
		FAIL();

	EXPECT_TRUE(vec->find("id") != vec->end());
}

TEST(SimParserTest, updateSection)
{
	auto parser = SimParser("update id = FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.updateSection();
	if (!vec)
		FAIL();

	EXPECT_TRUE(vec->find("id") != vec->end());
}

TEST(SimParserTest, simulation)
{
	auto parser = SimParser("init id = FLOAT[1] (+ INT[1]{1}, INT[1]{2}) update "
													"id = FLOAT[1] (+ INT[1]{1}, INT[1]{2})");

	auto vec = parser.simulation();
	if (!vec)
		FAIL();

	auto [init, update] = move(*vec);

	EXPECT_TRUE(init.find("id") != init.end());
	EXPECT_TRUE(update.find("id") != update.end());
	EXPECT_TRUE(init.find("id")->second == update.find("id")->second);
}
