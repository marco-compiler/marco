#include "gtest/gtest.h"
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>

#include "../TestingUtils.h"

using namespace modelica::codegen::model;

TEST(VectorAccessTest, DirectAccess)
{
	std::string stringModel = "model VectorAccess1 "
														"Real[4] x; "
														"equation "
														"x[4] = 0; "
														"end VectorAccess1; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	EXPECT_TRUE(VectorAccess::isCanonical(model.getEquations()[0].lhs()));

	AccessToVar access = AccessToVar::fromExp(model.getEquations()[0].lhs());

	EXPECT_EQ(access.getVar(), model.getVariables()[0]->getReference());
	EXPECT_EQ(access.getAccess().getMappingOffset().size(), 1);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getInductionVar(), 0);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getOffset(), 2);
}

TEST(VectorAccessTest, SingleOffset)
{
	std::string stringModel = "model VectorAccess3 "
														"Real[6] x; "
														"equation "
														"x[2 + 4] = 0; "
														"end VectorAccess3; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	EXPECT_TRUE(VectorAccess::isCanonical(model.getEquations()[0].lhs()));

	AccessToVar access = AccessToVar::fromExp(model.getEquations()[0].lhs());

	EXPECT_EQ(access.getVar(), model.getVariables()[0]->getReference());
	EXPECT_EQ(access.getAccess().getMappingOffset().size(), 1);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getInductionVar(), 0);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getOffset(), 4);
}

TEST(VectorAccessTest, MultiDirectAccess)
{
	std::string stringModel = "model VectorAccess2 "
														"Real[4, 2] x; "
														"equation "
														"x[4, 2] = 0; "
														"end VectorAccess2; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	EXPECT_TRUE(VectorAccess::isCanonical(model.getEquations()[0].lhs()));

	AccessToVar access = AccessToVar::fromExp(model.getEquations()[0].lhs());

	EXPECT_EQ(access.getVar(), model.getVariables()[0]->getReference());
	EXPECT_EQ(access.getAccess().getMappingOffset().size(), 2);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getInductionVar(), 0);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getOffset(), 3);
	EXPECT_EQ(access.getAccess().getMappingOffset()[1].getInductionVar(), 0);
	EXPECT_EQ(access.getAccess().getMappingOffset()[1].getOffset(), 1);
}

TEST(VectorAccessTest, SingleDimMap)
{
	SingleDimensionAccess disp = SingleDimensionAccess::relative(3, 0);
	modelica::Interval source(0, 10);
	modelica::Interval dest = disp.map(source);

	EXPECT_EQ(dest.min(), 3);
	EXPECT_EQ(dest.max(), 13);
}

TEST(VectorAccessTest, MultiDimMap)
{
	SingleDimensionAccess disp = SingleDimensionAccess::relative(3, 1);
	modelica::MultiDimInterval interval({ { 0, 10 }, { 4, 8 } });
	modelica::Interval dest = disp.map(interval);

	EXPECT_EQ(dest.min(), 7);
	EXPECT_EQ(dest.max(), 11);
}

TEST(VectorAccessTest, MapFromExpression)
{
	std::string stringModel = "model VectorAccess4 "
														"Real[10, 20, 30] x; "
														"equation "
														"for i in 1:3 loop "
														"for j in 1:7 loop "
														"x[i + 4, j + 10, 20] = 0; "
														"end for; "
														"end for; "
														"end VectorAccess4; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	EXPECT_TRUE(VectorAccess::isCanonical(model.getEquations()[0].lhs()));

	AccessToVar access = AccessToVar::fromExp(model.getEquations()[0].lhs());

	EXPECT_EQ(access.getAccess().mappableDimensions(), 2);
	EXPECT_EQ(access.getVar(), model.getVariables()[0]->getReference());
	EXPECT_EQ(access.getAccess().getMappingOffset().size(), 3);

	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getInductionVar(), 0);
	EXPECT_EQ(access.getAccess().getMappingOffset()[0].getOffset(), 3);
	EXPECT_EQ(access.getAccess().getMappingOffset()[1].getInductionVar(), 1);
	EXPECT_EQ(access.getAccess().getMappingOffset()[1].getOffset(), 9);
	EXPECT_EQ(access.getAccess().getMappingOffset()[2].getInductionVar(), 0);
	EXPECT_EQ(access.getAccess().getMappingOffset()[2].getOffset(), 19);

	EXPECT_TRUE(access.getAccess().getMappingOffset()[2].isDirecAccess());

	modelica::MultiDimInterval interval({ { 0, 10 }, { 4, 8 } });
	modelica::MultiDimInterval out = access.getAccess().map(interval);

	EXPECT_EQ(out.at(0).min(), 0 + 3);
	EXPECT_EQ(out.at(0).max(), 10 + 3);
	EXPECT_EQ(out.at(1).min(), 4 + 9);
	EXPECT_EQ(out.at(1).max(), 8 + 9);
	EXPECT_EQ(out.at(2).min(), 19 + 0);
	EXPECT_EQ(out.at(2).max(), 20 + 0);
}

TEST(VectorAccessTest, InvertedTest)
{
	std::string stringModel = "model VectorAccess4 "
														"Real[10, 20, 30] x; "
														"equation "
														"for i in 1:3 loop "
														"for j in 1:7 loop "
														"x[i + 4, j + 10, 20] = 0; "
														"end for; "
														"end for; "
														"end VectorAccess4; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	EXPECT_TRUE(VectorAccess::isCanonical(model.getEquations()[0].lhs()));

	AccessToVar access = AccessToVar::fromExp(model.getEquations()[0].lhs());

	EXPECT_EQ(access.getAccess().mappableDimensions(), 2);
	EXPECT_EQ(access.getVar(), model.getVariables()[0]->getReference());

	VectorAccess inverted = access.getAccess().invert();

	EXPECT_EQ(inverted.mappableDimensions(), 2);
	EXPECT_EQ(inverted.getMappingOffset().size(), 2);

	EXPECT_EQ(inverted.getMappingOffset()[0].getInductionVar(), 0);
	EXPECT_EQ(inverted.getMappingOffset()[0].getOffset(), -3);
	EXPECT_EQ(inverted.getMappingOffset()[1].getInductionVar(), 1);
	EXPECT_EQ(inverted.getMappingOffset()[1].getOffset(), -9);
}

TEST(VectorAccessTest, InverseMapFromExpression)
{
	std::string stringModel = "model VectorAccess4 "
														"Real[10, 20, 30] x; "
														"equation "
														"for i in 1:3 loop "
														"for j in 1:7 loop "
														"x[i + 4, j + 10, 20] = 0; "
														"end for; "
														"end for; "
														"end VectorAccess4; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	EXPECT_TRUE(VectorAccess::isCanonical(model.getEquations()[0].lhs()));

	AccessToVar access = AccessToVar::fromExp(model.getEquations()[0].lhs());

	EXPECT_EQ(access.getAccess().mappableDimensions(), 2);
	EXPECT_EQ(access.getAccess().invert().mappableDimensions(), 2);
	EXPECT_EQ(access.getVar(), model.getVariables()[0]->getReference());

	modelica::MultiDimInterval interval({ { 8, 12 }, { 10, 20 } });
	modelica::MultiDimInterval out = access.getAccess().invert().map(interval);

	EXPECT_EQ(out.at(0).min(), 8 - 3);
	EXPECT_EQ(out.at(0).max(), 12 - 3);
	EXPECT_EQ(out.at(1).min(), 10 - 9);
	EXPECT_EQ(out.at(1).max(), 20 - 9);
}

TEST(VectorAccessTest, TestCombineAbsoluteVectorAccess)
{
	VectorAccess v1({ SingleDimensionAccess::absolute(5) });
	VectorAccess v2({ SingleDimensionAccess::absolute(10) });
	VectorAccess result = v1.combine(v2);

	EXPECT_TRUE(result.getMappingOffset()[0].isDirecAccess());
	EXPECT_EQ(result.getMappingOffset()[0].getOffset(), 10);
}

TEST(VectorAccessTest, TestMapMultiDimAbsolute)
{
	SingleDimensionAccess singleAccess = SingleDimensionAccess::absolute(5);
	modelica::Interval result = singleAccess.map({ { 0, 1 }, { 3, 6 } });

	EXPECT_EQ(result.min(), 5);
	EXPECT_EQ(result.max(), 6);
}

TEST(VectorAccessTest, TestCombineRelativeVectorAccess)
{
	VectorAccess v1({ SingleDimensionAccess::relative(10, 0),
										SingleDimensionAccess::relative(20, 1) });
	VectorAccess v2({ SingleDimensionAccess::relative(10, 1),
										SingleDimensionAccess::relative(10, 0) });
	VectorAccess result = v1.combine(v2);

	EXPECT_TRUE(result.getMappingOffset()[0].isOffset());
	EXPECT_EQ(result.getMappingOffset()[0].getOffset(), 30);
	EXPECT_EQ(result.getMappingOffset()[0].getInductionVar(), 1);
	EXPECT_EQ(result.getMappingOffset()[1].getOffset(), 20);
	EXPECT_EQ(result.getMappingOffset()[1].getInductionVar(), 0);
}
