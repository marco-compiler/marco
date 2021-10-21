#include "gtest/gtest.h"
#include <marco/ida/IdaSolver.h>

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

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 1);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 1);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 1);

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

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 2);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 2);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 2);

	EXPECT_EQ(idaSolver.getRowLength(0), 1);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 1);

	EXPECT_EQ(idaSolver.getRowLength(1), 1);
	EXPECT_EQ(idaSolver.getDimension(1).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(1)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(1)[0].second, 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(
			idaSolver.getVariable(0) + idaSolver.getVariable(1),
			2.0 * idaSolver.getTime() +
					idaSolver.getTime() * idaSolver.getTime() / 2.0,
			1e-4);

	EXPECT_NEAR(
			idaSolver.getDerivative(0) + idaSolver.getDerivative(1),
			2.0 + idaSolver.getTime(),
			1e-4);

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

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 1);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 1);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 1);

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
														"tau*der(x[i]) = 2.0 * i; "
														"end for; "
														"end DerArray; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 2);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 10);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 10);

	EXPECT_EQ(idaSolver.getRowLength(0), 1);
	EXPECT_EQ(idaSolver.getRowLength(1), 1);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(1).size(), 1);

	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 1);
	EXPECT_EQ(idaSolver.getDimension(1)[0].first, 1);
	EXPECT_EQ(idaSolver.getDimension(1)[0].second, 10);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariable(0), 1 + 0.2 * idaSolver.getTime(), 1e-4);
	EXPECT_NEAR(idaSolver.getDerivative(0), 0.2, 1e-4);

	for (size_t i : marco::irange(1, 10))
	{
		EXPECT_NEAR(
				idaSolver.getVariable(i),
				1 + 0.4 * (i + 1) * idaSolver.getTime(),
				1e-4);
		EXPECT_NEAR(idaSolver.getDerivative(i), 0.4 * (i + 1), 1e-4);
	}

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

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 2);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 12);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 12);

	EXPECT_EQ(idaSolver.getRowLength(0), 1);
	EXPECT_EQ(idaSolver.getRowLength(1), 1);
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

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 2);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 10);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 15);

	EXPECT_EQ(idaSolver.getRowLength(0), 1);
	EXPECT_EQ(idaSolver.getRowLength(1), 2);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(1).size(), 1);

	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 5);
	EXPECT_EQ(idaSolver.getDimension(1)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(1)[0].second, 5);

	if (failed(idaSolver.run()))
		FAIL();

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

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 5);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 5);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 11);

	for (size_t i : marco::irange(4))
	{
		EXPECT_EQ(idaSolver.getRowLength(i), 2);
		EXPECT_EQ(idaSolver.getDimension(i).size(), 1);
		EXPECT_EQ(idaSolver.getDimension(i)[0].first, 0);
		EXPECT_EQ(idaSolver.getDimension(i)[0].second, 1);
	}

	EXPECT_EQ(idaSolver.getRowLength(4), 3);
	EXPECT_EQ(idaSolver.getDimension(4).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(4)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(4)[0].second, 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_EQ(idaSolver.getVariable(0), 3.0);
	EXPECT_EQ(idaSolver.getVariable(1), -1.0);
	EXPECT_EQ(idaSolver.getVariable(2), 0.0);
	EXPECT_EQ(idaSolver.getVariable(3), 1.0);
	EXPECT_EQ(idaSolver.getVariable(4), 4.0);

	for (size_t i : marco::irange(5))
		EXPECT_EQ(idaSolver.getDerivative(i), 0.0);

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, AlgebraicLoopSparseEquations)
{
	std::string stringModel = "model SparseLoop "
														"Real[3] x; "
														"equation "
														"x[1] + x[2] = 3; "
														"x[1] + x[3] = 2; "
														"x[2] + x[3] = 1; "
														"end SparseLoop; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 3);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 3);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 6);

	for (size_t i : marco::irange(3))
	{
		EXPECT_EQ(idaSolver.getRowLength(i), 2);
		EXPECT_EQ(idaSolver.getDimension(i).size(), 1);
		EXPECT_EQ(idaSolver.getDimension(i)[0].first, 0);
		EXPECT_EQ(idaSolver.getDimension(i)[0].second, 1);
	}

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_EQ(idaSolver.getVariable(0), 2.0);
	EXPECT_EQ(idaSolver.getVariable(1), 1.0);
	EXPECT_EQ(idaSolver.getVariable(2), 0.0);

	for (size_t i : marco::irange(3))
		EXPECT_EQ(idaSolver.getDerivative(i), 0.0);

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, AlgebraicLoopDenseEquations)
{
	std::string stringModel = "model DenseLoop "
														"Real[2] x; "
														"Real[2] y; "
														"Real[2] z; "
														"equation "
														"for i in 1:2 loop "
														"x[i] + y[i] - z[i] = 1.0; "
														"x[i] - y[i] + z[i] = 2.0; "
														"-x[i] + y[i] + z[i] = 3.0; "
														"end for; "
														"end DenseLoop; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 3);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 6);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 18);

	for (size_t i : marco::irange(3))
	{
		EXPECT_EQ(idaSolver.getRowLength(i), 3);
		EXPECT_EQ(idaSolver.getDimension(i).size(), 1);
	}

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_EQ(
			idaSolver.getVariable(0) + idaSolver.getVariable(2) +
					idaSolver.getVariable(4),
			6.0);

	for (size_t i : marco::irange(6))
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

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 1);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 1);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 1);

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

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 1);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 2);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 2);

	EXPECT_EQ(idaSolver.getRowLength(0), 1);
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

TEST(IdaSolverTest, Robertson)
{
	std::string stringModel =
			"model Robertson "
			"Real y1(start = 1.0); "
			"Real y2(start = 0.0); "
			"Real y3(start = 0.0); "
			"equation "
			"der(y1) = -0.04 * y1 + 1e4 * y2 * y3; "
			"der(y2) = +0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 * y2; "
			"0 = y1 + y2 + y3 - 1; "
			"end Robertson; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model, 0, 4000);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 2);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 2);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 4);

	EXPECT_EQ(idaSolver.getRowLength(0), 2);
	EXPECT_EQ(idaSolver.getRowLength(1), 2);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(1).size(), 1);

	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 1);
	EXPECT_EQ(idaSolver.getDimension(1)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(1)[0].second, 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(
			idaSolver.getVariable(0) + idaSolver.getVariable(1), 0.1832, 1e-2);

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, LoopWithDerivative)
{
	std::string stringModel = "model Loop17 "
														"Real x; "
														"Real y; "
														"equation "
														"der(x) + y = 4; "
														"der(x) - y = 2; "
														"end Loop17; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 1);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 1);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 1);

	EXPECT_EQ(idaSolver.getRowLength(0), 1);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariable(0), 3.0 * idaSolver.getTime(), 1e-4);

	EXPECT_NEAR(idaSolver.getDerivative(0), 3.0, 1e-4);

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, LoopWithImplicitEquation)
{
	std::string stringModel = "model Loop18 "
														"Real x; "
														"Real y; "
														"equation "
														"x + y = 1; "
														"x*x*x + x = y; "
														"end Loop18; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 2);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 2);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 4);

	EXPECT_EQ(idaSolver.getRowLength(0), 2);
	EXPECT_EQ(idaSolver.getRowLength(1), 2);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);
	EXPECT_EQ(idaSolver.getDimension(1).size(), 1);

	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 1);
	EXPECT_EQ(idaSolver.getDimension(1)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(1)[0].second, 1);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_NEAR(idaSolver.getVariable(0) + idaSolver.getVariable(1), 1.0, 1e-4);

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, MultipleBltBlocksAndEquations)
{
	std::string stringModel = "model Sched4 "
														"Real[3] x; "
														"Real[2] y; "
														"Real[4] w; "
														"Real z; "
														"equation "
														"for i in 3:3 loop "
														"w[i] - w[i+1] = y[2] - w[i-2]; "
														"end for; "
														"for j in 1:2 loop "
														"y[j] = w[j] + x[j]; "
														"end for; "
														"for j in 1:1 loop "
														"w[j] + w[j+1] = x[3]; "
														"end for; "
														"w[4] = 7 - w[3]; "
														"w[3] = z - y[1]; "
														"for i in 1:3 loop "
														"x[i] = 5; "
														"end for; "
														"w[1] - w[2] = 3; "
														"end Sched4; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 4);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 4);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 10);

	for (size_t i : marco::irange(3))
	{
		EXPECT_EQ(idaSolver.getRowLength(i), 2);
		EXPECT_EQ(idaSolver.getDimension(i).size(), 1);
		EXPECT_EQ(idaSolver.getDimension(i)[0].first, 0);
		EXPECT_EQ(idaSolver.getDimension(i)[0].second, 1);
	}

	EXPECT_EQ(idaSolver.getRowLength(3), 4);
	EXPECT_EQ(idaSolver.getDimension(3).size(), 1);

	EXPECT_EQ(idaSolver.getDimension(3)[0].first, 2);
	EXPECT_EQ(idaSolver.getDimension(3)[0].second, 3);

	if (failed(idaSolver.run()))
		FAIL();

	EXPECT_EQ(idaSolver.getVariable(0), 4.0);
	EXPECT_EQ(idaSolver.getVariable(1), 1.0);
	EXPECT_EQ(idaSolver.getVariable(2), 4.5);
	EXPECT_EQ(idaSolver.getVariable(3), 2.5);

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, EquationWithInduction)
{
	std::string stringModel = "model IdaInduction "
														"Real[5] x; "
														"Real[10] y; "
														"equation "
														"for i in 1:10 loop "
														"y[i] = i + sin(time * 100); "
														"end for; "
														"for i in 1:5 loop "
														"der(x[i]) = y[i+5]; "
														"end for; "
														"end IdaInduction; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model, 0, 1, 0.1, 1e-10, 1e-10);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 1);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 5);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 5);

	EXPECT_EQ(idaSolver.getRowLength(0), 1);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 1);

	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 5);

	if (failed(idaSolver.run()))
		FAIL();

	for (size_t i : marco::irange(5))
	{
		EXPECT_NEAR(
				idaSolver.getDerivative(i),
				i + 6 + sin(idaSolver.getTime() * 100),
				1e-4);
	}

	if (failed(idaSolver.free()))
		FAIL();
}

TEST(IdaSolverTest, UnevenJacobianRowLength)
{
	std::string stringModel = "model UnevenRow "
														"Real[3, 4] x; "
														"equation "
														"for i in 1:3 loop "
														"for j in 1:4 loop "
														"der(x[i, j]) = 2 * der(x[2, 2]) - 4; "
														"end for; "
														"end for; "
														"end UnevenRow; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	marco::codegen::ida::IdaSolver idaSolver(model);
	if (failed(idaSolver.init()))
		FAIL();

	EXPECT_EQ(idaSolver.getForEquationsNumber(), 1);
	EXPECT_EQ(idaSolver.getEquationsNumber(), 12);
	EXPECT_EQ(idaSolver.getNonZeroValuesNumber(), 23);

	EXPECT_EQ(idaSolver.getRowLength(0), 2);
	EXPECT_EQ(idaSolver.getDimension(0).size(), 2);

	EXPECT_EQ(idaSolver.getDimension(0)[0].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[0].second, 3);
	EXPECT_EQ(idaSolver.getDimension(0)[1].first, 0);
	EXPECT_EQ(idaSolver.getDimension(0)[1].second, 4);

	if (failed(idaSolver.run()))
		FAIL();

	for (size_t i : marco::irange(12))
	{
		EXPECT_EQ(idaSolver.getVariable(i), idaSolver.getTime() * 4);
		EXPECT_EQ(idaSolver.getDerivative(i), 4);
	}

	if (failed(idaSolver.free()))
		FAIL();
}
