#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModParser.hpp"
#include "marco/model/ModType.hpp"
#include "marco/model/VectorAccess.hpp"

using namespace marco;
using namespace std;

TEST(VectorAccessTest, directAccess)
{
	ModParser parser("INT[1] (at INT[3] intVector, INT[1] (ind INT[1]{4})))");
	auto exp = parser.expression();

	if (!exp)
		FAIL();

	if (!VectorAccess::isCanonical(*exp))
		FAIL();

	auto access = AccessToVar::fromExp(*exp);
	EXPECT_EQ(access.getVarName(), "intVector");
	EXPECT_EQ(access.getAccess().getMappingOffset().size(), 1);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getInductionVar(), 4);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getOffset(), 0);
}

TEST(VectorAccessTest, multiDirectAccess)
{
	ModParser parser("INT[1] (at INT[3] (at INT[3, 2] intVector, INT[1] (ind "
									 "INT[1]{4})), INT[1] (ind INT[1]{2}))");
	auto exp = parser.expression();

	if (!exp)
		FAIL();

	if (!VectorAccess::isCanonical(*exp))
		FAIL();
	auto access = AccessToVar::fromExp(*exp);
	EXPECT_EQ(access.getVarName(), "intVector");
	EXPECT_EQ(access.getAccess().getMappingOffset().size(), 2);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getInductionVar(), 4);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getOffset(), 0);
	EXPECT_EQ(access.getAccess().getMappingOffset()[1].getInductionVar(), 2);
	EXPECT_EQ(access.getAccess().getMappingOffset()[1].getOffset(), 0);
}

TEST(VectorAccessTest, singleOffset)
{
	ModExp exp = ModExp::at(
			ModExp("referene", ModType(typeToBuiltin<int>(), { 2 })),
			ModExp::add(ModExp::induction(ModConst(2)), ModExp(ModConst(4))));

	if (!VectorAccess::isCanonical(exp))
		FAIL();
	auto access = AccessToVar::fromExp(exp);
	EXPECT_EQ(access.getVarName(), "referene");
	EXPECT_EQ(access.getAccess().getMappingOffset().size(), 1);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getInductionVar(), 2);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getOffset(), 4);
}

TEST(VectorAccessTest, singleDimMap)
{
	SingleDimensionAccess disp = SingleDimensionAccess::relative(3, 0);
	Interval source(0, 10);

	auto dest = disp.map(source);
	EXPECT_EQ(dest.min(), 3);
	EXPECT_EQ(dest.max(), 13);
}

TEST(VectorAccessTest, multiDimMap)
{
	SingleDimensionAccess disp = SingleDimensionAccess::relative(3, 1);

	MultiDimInterval intervall({ { 0, 10 }, { 4, 8 } });

	auto dest = disp.map(intervall);

	EXPECT_EQ(dest.min(), 7);
	EXPECT_EQ(dest.max(), 11);
}

TEST(VectorAccessTest, mapFromExp)
{
	// equivalent to reference[b + 4][a + 10][20];
	ModExp exp = ModExp::at(
			ModExp("referene", ModType(typeToBuiltin<int>(), 2, 3, 3)),
			ModExp::add(ModExp::induction(ModConst(1)), ModExp(ModConst(4))));

	exp = ModExp::at(
			move(exp),
			ModExp::add(ModExp::induction(ModConst(0)), ModExp(ModConst(10))));

	exp = ModExp::at(move(exp), ModExp(ModConst(20)));

	if (!VectorAccess::isCanonical(exp))
		FAIL();
	auto access = AccessToVar::fromExp(exp);
	EXPECT_EQ(access.getVarName(), "referene");
	EXPECT_EQ(access.getAccess().getMappingOffset().size(), 3);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getInductionVar(), 1);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getOffset(), 4);
	EXPECT_EQ(access.getAccess().getMappingOffset()[1].getInductionVar(), 0);
	EXPECT_EQ(access.getAccess().getMappingOffset()[1].getOffset(), 10);
	EXPECT_EQ(access.getAccess().getMappingOffset()[2].getOffset(), 20);
	EXPECT_TRUE(access.getAccess().getMappingOffset()[2].isDirecAccess());

	MultiDimInterval intervall({ { 0, 10 }, { 4, 8 } });
	auto out = access.getAccess().map(intervall);

	EXPECT_EQ(out.at(0).min(), 8);
	EXPECT_EQ(out.at(0).max(), 12);
	EXPECT_EQ(out.at(1).min(), 10);
	EXPECT_EQ(out.at(1).max(), 20);
	EXPECT_EQ(out.at(2).min(), 20);
	EXPECT_EQ(out.at(2).max(), 21);
}

TEST(VectorAccessTest, inverseMapFromExp)
{
	// equivalent to reference[b + 4][a + 10][20];
	ModExp exp = ModExp::at(
			ModExp("referene", ModType(typeToBuiltin<int>(), 2, 3, 3)),
			ModExp::add(ModExp::induction(ModConst(1)), ModExp(ModConst(4))));

	exp = ModExp::at(
			move(exp),
			ModExp::add(ModExp::induction(ModConst(0)), ModExp(ModConst(10))));

	exp = ModExp::at(move(exp), ModExp(ModConst(20)));

	if (!VectorAccess::isCanonical(exp))
		FAIL();
	auto access = AccessToVar::fromExp(exp);

	EXPECT_EQ(2, access.getAccess().mappableDimensions());

	MultiDimInterval intervall({ { 8, 12 }, { 10, 20 }, { 20, 21 } });
	auto out = access.getAccess().invert().map(intervall);

	EXPECT_EQ(out.at(0).min(), 0);
	EXPECT_EQ(out.at(0).max(), 10);
	EXPECT_EQ(out.at(1).min(), 4);
	EXPECT_EQ(out.at(1).max(), 8);
}

TEST(VectorAccessTest, inverteTest)
{
	// equivalent to reference[b + 4][a + 10][20];
	ModExp exp = ModExp::at(
			ModExp("referene", ModType(typeToBuiltin<int>(), 2, 3, 3)),
			ModExp::add(ModExp::induction(ModConst(1)), ModExp(ModConst(4))));

	exp = ModExp::at(
			move(exp),
			ModExp::add(ModExp::induction(ModConst(0)), ModExp(ModConst(10))));

	exp = ModExp::at(move(exp), ModExp(ModConst(20)));

	if (!VectorAccess::isCanonical(exp))
		FAIL();
	auto access = AccessToVar::fromExp(exp);

	EXPECT_EQ(2, access.getAccess().mappableDimensions());

	auto inverted = access.getAccess().invert();

	EXPECT_EQ(2, inverted.mappableDimensions());
	EXPECT_EQ(access.getVarName(), "referene");
	EXPECT_EQ(inverted.getMappingOffset().size(), 2);
	EXPECT_EQ(inverted.getMappingOffset()[0].getInductionVar(), 1);
	EXPECT_EQ(inverted.getMappingOffset()[0].getOffset(), -10);
	EXPECT_EQ(inverted.getMappingOffset()[1].getInductionVar(), 0);
	EXPECT_EQ(inverted.getMappingOffset()[1].getOffset(), -4);
}

TEST(VectorAccessTest, toStringTest)
{
	ModExp exp = ModExp::at(
			ModExp("referene", ModType(typeToBuiltin<int>(), 2, 3, 3)),
			ModExp::add(ModExp::induction(ModConst(1)), ModExp(ModConst(4))));
	if (!VectorAccess::isCanonical(exp))
		FAIL();
	auto access = AccessToVar::fromExp(exp);

	EXPECT_EQ(access.getAccess().toString(), "[I1 + 4]");
}

TEST(VectorAccessTest, testCombineAbsoluteVectorAccess)
{
	VectorAccess v1({ SingleDimensionAccess::absolute(5) });
	VectorAccess v2({ SingleDimensionAccess::absolute(10) });
	auto result = v1.combine(v2);
	EXPECT_TRUE(result.getMappingOffset()[0].isDirecAccess());
	EXPECT_EQ(result.getMappingOffset()[0].getOffset(), 10);
}

TEST(VectorAccessTest, testMapMultiDimAbsolute)
{
	auto singleAcces = SingleDimensionAccess::absolute(5);
	auto result = singleAcces.map({ { 0, 1 }, { 3, 6 } });
	EXPECT_EQ(result.min(), 5);
	EXPECT_EQ(result.max(), 6);
}

TEST(VectorAccessTest, testCombineRelativeVectorAccess)
{
	VectorAccess v1({ SingleDimensionAccess::relative(10, 0),
										SingleDimensionAccess::relative(20, 1) });
	VectorAccess v2({ SingleDimensionAccess::relative(10, 1),
										SingleDimensionAccess::relative(10, 0) });
	auto result = v1.combine(v2);
	EXPECT_TRUE(result.getMappingOffset()[0].isOffset());
	EXPECT_EQ(result.getMappingOffset()[0].getOffset(), 30);
	EXPECT_EQ(result.getMappingOffset()[0].getInductionVar(), 1);
	EXPECT_EQ(result.getMappingOffset()[1].getOffset(), 20);
	EXPECT_EQ(result.getMappingOffset()[1].getInductionVar(), 0);
}
