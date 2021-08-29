#include "gtest/gtest.h"
#include <marco/mlirlowerer/dialects/ida/IdaSolver.h>

#include "../TestingUtils.h"

using namespace marco::codegen::model;

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

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getProblemSize(), 1);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 1);

	EXPECT_EQ(idaSolver.getRowLength(0), 1);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariable(0), 2.0 * idaSolver.getTime(), 1e-4);
	EXPECT_EQ(idaSolver.getDerivative(0), 2.0);

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, DoubleDerivative)
{
	std::string stringModel = "model DoubleDer "
														"Real x; "
														"Real y; "
														"equation "
														"der(x) = 2.0; "
														"der(y) = time; "
														"end DoubleDer; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getProblemSize(), 2);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 2);

	EXPECT_EQ(idaSolver.getRowLength(0), 1);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 1);

	EXPECT_EQ(idaSolver.getRowLength(1), 2);
	EXPECT_EQ(idaSolver.getDimension(1).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(1)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(1)[0].second, 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariable(0), 2.0 * idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(
			idaSolver.getVariable(1),
			idaSolver.getTime() * idaSolver.getTime() / 2.0,
			1e-4);
	EXPECT_NEAR(idaSolver.getDerivative(0), 2.0, 1e-4);
	EXPECT_NEAR(idaSolver.getDerivative(1), idaSolver.getTime(), 1e-4);

	if (failed(idaSolver.free()))
		FAIL();
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

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getProblemSize(), 1);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 1);

	EXPECT_EQ(idaSolver.getRowLength(0), 1);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariable(0), 2.0 * idaSolver.getTime(), 1e-4);
	EXPECT_EQ(idaSolver.getDerivative(0), 2.0);

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, DerivativeArray)
{
	std::string stringModel = "model DerArray "
														"final parameter Real tau = 5.0; "
														"Real[10] x(start = 1.0); "
														"equation "
														"tau*der(x[1]) = 1.0; "
														"for i in 2:10 loop "
														"tau*der(x[i]) = 2.0; "
														"end for; "
														"end DerArray; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getProblemSize(), 2);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 10);

	EXPECT_EQ(idaSolver.getRowLength(0), 10);
	EXPECT_EQ(idaSolver.getRowLength(1), 10);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(1).size(), 1);

	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 1);
	EXPECT_EQ(idaSolver.getDimension(1)[0].first, 1);
	EXPECT_EQ(idaSolver.getDimension(1)[0].second, 10);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariable(0), 1 + 0.2 * idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(idaSolver.getVariable(1), 1 + 0.4 * idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(idaSolver.getDerivative(0), 0.2, 1e-4);
	EXPECT_NEAR(idaSolver.getDerivative(1), 0.4, 1e-4);

	if (failed(idaSolver.free()))
		FAIL();
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

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getProblemSize(), 2);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 12);

	EXPECT_EQ(idaSolver.getRowLength(0), 12);
	EXPECT_EQ(idaSolver.getRowLength(1), 12);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(1).size(), 2);

	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 3);
	EXPECT_EQ(idaSolver.getDimension(1)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(1)[0].second, 3);
	EXPECT_EQ(idaSolver.getDimension(1)[1].first, 1);
	EXPECT_EQ(idaSolver.getDimension(1)[1].second, 4);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariable(0), 1.0 * idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(idaSolver.getVariable(1), 2.0 * idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(idaSolver.getVariable(4), 1.0 * idaSolver.getTime(), 1e-4);
	EXPECT_EQ(idaSolver.getDerivative(0), 1.0);
	EXPECT_EQ(idaSolver.getDerivative(1), 2.0);
	EXPECT_EQ(idaSolver.getDerivative(4), 1.0);

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, MultipleArraysWithState)
{
	std::string stringModel = "model ArraysWithState "
														"Real[5] x; "
														"Real[5] y; "
														"equation "
														"for i in 1:5 loop "
														"der(x[i]) = 1.0; "
														"end for; "
														"for i in 1:5 loop "
														"der(y[i]) = 3 + x[i]; "
														"end for; "
														"end ArraysWithState; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getProblemSize(), 2);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 10);

	EXPECT_EQ(idaSolver.getRowLength(0), 5);
	EXPECT_EQ(idaSolver.getRowLength(1), 10);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(1).size(), 1);

	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 5);
	EXPECT_EQ(idaSolver.getDimension(1)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(1)[0].second, 5);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariable(0), idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(
			idaSolver.getVariable(5),
			(3.0 + idaSolver.getTime() / 2) * idaSolver.getTime(),
			1e-4);
	EXPECT_NEAR(idaSolver.getDerivative(0), 1.0, 1e-4);
	EXPECT_NEAR(idaSolver.getDerivative(5), 3.0 + idaSolver.getTime(), 1e-4);

	if (failed(idaSolver.free()))
		FAIL();
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

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getProblemSize(), 5);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 5);

	for (int64_t i : marco::irange(5))
	{
		EXPECT_EQ(idaSolver.getRowLength(i), 5);
		EXPECT_EQ(idaSolver.getDimension(i).size(), 1);
		EXPECT_EQ(idaSolver.getDimension(i)[0].first, 0);
		EXPECT_EQ(idaSolver.getDimension(i)[0].second, 1);
	}

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_EQ(idaSolver.getVariable(0), 3.0);
	EXPECT_EQ(idaSolver.getVariable(1), -1.0);
	EXPECT_EQ(idaSolver.getVariable(2), 0.0);
	EXPECT_EQ(idaSolver.getVariable(3), 1.0);
	EXPECT_EQ(idaSolver.getVariable(4), 4.0);

	for (int64_t i : marco::irange(5))
		EXPECT_EQ(idaSolver.getDerivative(i), 0.0);

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, ImplicitEquation)
{
	std::string stringModel = "model ImplicitEq "
														"Real x(start = 1.5); "
														"equation "
														"x + x * x * x = 7.0; "
														"end ImplicitEq; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getProblemSize(), 1);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 1);

	EXPECT_EQ(idaSolver.getRowLength(0), 1);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariable(0), 1.7392, 1e-4);

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, ImplicitEqKepler)
{
	std::string stringModel = "model ImplicitKepler "
														"Real[2] x(start = 3.7); "
														"equation "
														"for i in 1:2 loop "
														"5.0 = x[i] - 2.72 * sin(x[i]); "
														"end for; "
														"end ImplicitKepler; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getProblemSize(), 1);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 2);

	EXPECT_EQ(idaSolver.getRowLength(0), 2);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 2);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariable(0), 3.6577, 1e-4);
	EXPECT_NEAR(idaSolver.getVariable(1), 3.6577, 1e-4);

	if (failed(idaSolver.free()))
		FAIL();
}
