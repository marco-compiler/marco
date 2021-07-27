#include "gtest/gtest.h"
#include <modelica/mlirlowerer/passes/matching/LinSolver.h>
#include <modelica/mlirlowerer/passes/matching/Matching.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Model.h>

#include "../TestingUtils.h"

using namespace modelica::codegen::model;

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

	llvm::SmallVector<Equation, 3> equations;
	for (Equation& eq : model.getEquations())
	{
		if (failed(eq.explicitate()))
			FAIL();
		if (failed(eq.normalize()))
			FAIL();
		equations.push_back(eq);
	}

	EXPECT_TRUE(canSolveSystem(model.getEquations(), model));

	mlir::OpBuilder builder(model.getOp());
	if (failed(linearySolve(builder, equations)))
		FAIL();

	EXPECT_TRUE(model.getEquations()[0].lhs().isReferenceAccess());
	EXPECT_TRUE(model.getEquations()[1].lhs().isReferenceAccess());
	EXPECT_TRUE(model.getEquations()[0].rhs().isOperation());
	EXPECT_TRUE(model.getEquations()[1].rhs().isOperation());
}
