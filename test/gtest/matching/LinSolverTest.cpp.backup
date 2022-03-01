#include "gtest/gtest.h"
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>
#include <marco/mlirlowerer/passes/matching/LinSolver.h>
#include <marco/mlirlowerer/passes/matching/Matching.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Model.h>

#include "../TestingUtils.h"

using namespace marco::codegen::model;
using namespace marco::codegen::modelica;

TEST(LinSolverTest, LinearySolveTest)
{
	std::string stringModel = "model LinSolver "
														"Real x; "
														"Real y; "
														"equation "
														"x + y = 4; "
														"x - y = 2; "
														"end LinSolver; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getEquations().size(), 2);

	EXPECT_TRUE(model.getEquations()[0].lhs().isOperation());
	EXPECT_TRUE(model.getEquations()[1].lhs().isOperation());
	EXPECT_TRUE(model.getEquations()[0].rhs().isConstant());
	EXPECT_TRUE(model.getEquations()[1].rhs().isConstant());

	llvm::SmallVector<Equation, 3> equations;
	for (Equation& eq : model.getEquations())
	{
		if (failed(eq.explicitate()))
			FAIL();
		eq.normalize();
		equations.push_back(eq);
	}

	EXPECT_TRUE(canSolveSystem(model.getEquations()));

	EXPECT_TRUE(model.getEquations()[0].lhs().isReferenceAccess());
	EXPECT_TRUE(model.getEquations()[1].lhs().isReferenceAccess());
	EXPECT_TRUE(model.getEquations()[0].rhs().isOperation());
	EXPECT_TRUE(model.getEquations()[1].rhs().isOperation());

	mlir::OpBuilder builder(model.getOp());
	if (failed(linearySolve(builder, equations)))
		FAIL();

	for (Equation& eq : equations)
		if (failed(eq.explicitate()))
			FAIL();

	EXPECT_TRUE(equations[0].lhs().isReferenceAccess());
	EXPECT_TRUE(equations[1].lhs().isReferenceAccess());
}

TEST(LinSolverTest, DifferentReplaceUses)
{
	std::string stringModel = "model SubstituteTwice "
														"Real[2] x; "
														"Real y; "
														"equation "
														"x[1] = 2.0; "
														"x[2] = 1.0; "
														"der(y) = x[1] + x[2]; "
														"end SubstituteTwice; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);
	mlir::OpBuilder builder(model.getOp());

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getEquations().size(), 3);

	Equation destination = model.getEquations()[2];
	EXPECT_TRUE(destination.lhs().isReferenceAccess());
	EXPECT_TRUE(destination.rhs().isOperation());

	replaceUses(builder, model.getEquations()[0], destination);
	EXPECT_TRUE(destination.rhs().isOperation());

	replaceUses(builder, model.getEquations()[1], destination);
	EXPECT_TRUE(destination.rhs().isConstant());
	EXPECT_TRUE(mlir::isa<ConstantOp>(destination.rhs().getOp()));

	auto attribute = mlir::cast<ConstantOp>(destination.rhs().getOp()).value();
	EXPECT_TRUE(attribute.isa<RealAttribute>());
	EXPECT_EQ(attribute.cast<RealAttribute>().getValue(), 3.0);
}
