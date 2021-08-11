#include "gtest/gtest.h"
#include <modelica/mlirlowerer/passes/ida/IdaSolver.h>

#include "../TestingUtils.h"

using namespace modelica::codegen::model;

TEST(IdaSolverTest, SimpleDerivative)
{
	std::string stringModel = "model SimpleDer "
														"Real x; "
														"equation "
														"der(x) = 2.0; "
														"end SimpleDer; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	modelica::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getData()->rowLength.size(), 1);
	EXPECT_EQ(idaSolver.getData()->dimensions.size(), 1);
	EXPECT_EQ(idaSolver.getData()->residuals.size(), 1);
	EXPECT_EQ(idaSolver.getData()->jacobianMatrix.size(), 1);

	EXPECT_EQ(idaSolver.getData()->rowLength[0], 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[0].size(), 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[0][0].first, 0);
	EXPECT_EQ(idaSolver.getData()->dimensions[0][0].second, 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariables()[0], 2.0 * idaSolver.getTime(), 1e-4);
	EXPECT_EQ(idaSolver.getDerivatives()[0], 2.0);

	idaSolver.free();
}

TEST(IdaSolverTest, DoubleDerivative)
{
	std::string stringModel = "model DoubleDer "
														"Real x; "
														"Real y; "
														"equation "
														"der(x) = 2.0; "
														"der(y) = 1.5; "
														"end DoubleDer; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	modelica::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getData()->rowLength.size(), 2);
	EXPECT_EQ(idaSolver.getData()->dimensions.size(), 2);
	EXPECT_EQ(idaSolver.getData()->residuals.size(), 2);
	EXPECT_EQ(idaSolver.getData()->jacobianMatrix.size(), 2);

	EXPECT_EQ(idaSolver.getData()->rowLength[0], 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[0].size(), 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[0][0].first, 0);
	EXPECT_EQ(idaSolver.getData()->dimensions[0][0].second, 1);

	EXPECT_EQ(idaSolver.getData()->rowLength[1], 2);
	EXPECT_EQ(idaSolver.getData()->dimensions[1].size(), 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[1][0].first, 0);
	EXPECT_EQ(idaSolver.getData()->dimensions[1][0].second, 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariables()[0], 2.0 * idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(idaSolver.getVariables()[1], 1.5 * idaSolver.getTime(), 1e-4);
	EXPECT_EQ(idaSolver.getDerivatives()[0], 2.0);
	EXPECT_EQ(idaSolver.getDerivatives()[1], 1.5);

	idaSolver.free();
}

TEST(IdaSolverTest, SimpleDerWithSubstitution)
{
	std::string stringModel = "model SimpleDerSub "
														"Real y; "
														"Real x; "
														"equation "
														"der(x) = y; "
														"y = 2.0; "
														"end SimpleDerSub; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	modelica::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getData()->rowLength.size(), 1);
	EXPECT_EQ(idaSolver.getData()->dimensions.size(), 1);
	EXPECT_EQ(idaSolver.getData()->residuals.size(), 1);
	EXPECT_EQ(idaSolver.getData()->jacobianMatrix.size(), 1);

	EXPECT_EQ(idaSolver.getData()->rowLength[0], 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[0].size(), 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[0][0].first, 0);
	EXPECT_EQ(idaSolver.getData()->dimensions[0][0].second, 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariables()[0], 2.0 * idaSolver.getTime(), 1e-4);
	EXPECT_EQ(idaSolver.getDerivatives()[0], 2.0);

	idaSolver.free();
}

TEST(IdaSolverTest, DerivativeArray)
{
	std::string stringModel = "model DerArray "
														"final parameter Real tau = 5.0; "
														"Real[10] x(start = 0.0); "
														"equation "
														"tau*der(x[1]) = 1.0; "
														"for i in 2:10 loop "
														"tau*der(x[i]) = 2.0; "
														"end for; "
														"end DerArray; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	modelica::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getData()->rowLength.size(), 2);
	EXPECT_EQ(idaSolver.getData()->dimensions.size(), 2);
	EXPECT_EQ(idaSolver.getData()->residuals.size(), 2);
	EXPECT_EQ(idaSolver.getData()->jacobianMatrix.size(), 2);

	EXPECT_EQ(idaSolver.getData()->rowLength[0], 10);
	EXPECT_EQ(idaSolver.getData()->rowLength[1], 10);
	EXPECT_EQ(idaSolver.getData()->dimensions[0].size(), 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[1].size(), 1);

	EXPECT_EQ(idaSolver.getData()->dimensions[0][0].first, 0);
	EXPECT_EQ(idaSolver.getData()->dimensions[0][0].second, 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[1][0].first, 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[1][0].second, 10);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariables()[0], 0.2 * idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(idaSolver.getVariables()[1], 0.4 * idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(idaSolver.getDerivatives()[0], 0.2, 1e-4);
	EXPECT_NEAR(idaSolver.getDerivatives()[1], 0.4, 1e-4);

	idaSolver.free();
}

TEST(IdaSolverTest, MultidimensionalDerivative)
{
	std::string stringModel = "model MultidimDer "
														"Real[3, 4] x(start = 0.0); "
														"equation "
														"for i in 1:3 loop "
														"der(x[i, 1]) = 1.0; "
														"end for; "
														"for i in 1:3 loop "
														"for j in 2:4 loop "
														"der(x[i, j]) = 2.0; "
														"end for; "
														"end for; "
														"end MultidimDer; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	modelica::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getData()->rowLength.size(), 2);
	EXPECT_EQ(idaSolver.getData()->dimensions.size(), 2);
	EXPECT_EQ(idaSolver.getData()->residuals.size(), 2);
	EXPECT_EQ(idaSolver.getData()->jacobianMatrix.size(), 2);

	EXPECT_EQ(idaSolver.getData()->rowLength[0], 12);
	EXPECT_EQ(idaSolver.getData()->rowLength[1], 12);
	EXPECT_EQ(idaSolver.getData()->dimensions[0].size(), 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[1].size(), 2);

	EXPECT_EQ(idaSolver.getData()->dimensions[0][0].first, 0);
	EXPECT_EQ(idaSolver.getData()->dimensions[0][0].second, 3);
	EXPECT_EQ(idaSolver.getData()->dimensions[1][0].first, 0);
	EXPECT_EQ(idaSolver.getData()->dimensions[1][0].second, 3);
	EXPECT_EQ(idaSolver.getData()->dimensions[1][1].first, 1);
	EXPECT_EQ(idaSolver.getData()->dimensions[1][1].second, 4);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariables()[0], 1.0 * idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(idaSolver.getVariables()[1], 2.0 * idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(idaSolver.getVariables()[4], 1.0 * idaSolver.getTime(), 1e-4);
	EXPECT_EQ(idaSolver.getDerivatives()[0], 1.0);
	EXPECT_EQ(idaSolver.getDerivatives()[1], 2.0);
	EXPECT_EQ(idaSolver.getDerivatives()[4], 1.0);

	idaSolver.free();
}

TEST(IdaSolverTest, AlgebraicLoop)
{
	std::string stringModel = "model AlgebraicLoop "
														"int[5] x; "
														"equation "
														"x[1] + x[2] = 2; "
														"x[1] - x[2] = 4; "
														"x[3] + x[4] = 1; "
														"x[3] - x[4] = -1; "
														"x[5] = x[4] + x[1]; "
														"end AlgebraicLoop; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	modelica::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getData()->rowLength.size(), 5);
	EXPECT_EQ(idaSolver.getData()->dimensions.size(), 5);
	EXPECT_EQ(idaSolver.getData()->residuals.size(), 5);
	EXPECT_EQ(idaSolver.getData()->jacobianMatrix.size(), 5);

	for (size_t i : modelica::irange(5))
	{
		EXPECT_EQ(idaSolver.getData()->rowLength[i], 5);
		EXPECT_EQ(idaSolver.getData()->dimensions[i].size(), 1);
		EXPECT_EQ(idaSolver.getData()->dimensions[i][0].first, 0);
		EXPECT_EQ(idaSolver.getData()->dimensions[i][0].second, 1);
	}

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_EQ(idaSolver.getVariables()[0], 3.0);
	EXPECT_EQ(idaSolver.getVariables()[1], -1.0);
	EXPECT_EQ(idaSolver.getVariables()[2], 0.0);
	EXPECT_EQ(idaSolver.getVariables()[3], 1.0);
	EXPECT_EQ(idaSolver.getVariables()[4], 4.0);

	for (size_t i : modelica::irange(5))
	{
		EXPECT_EQ(idaSolver.getDerivatives()[i], 0.0);
	}

	idaSolver.free();
}
