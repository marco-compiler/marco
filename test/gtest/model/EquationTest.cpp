#include "gtest/gtest.h"
#include <modelica/mlirlowerer/passes/matching/Matching.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>

#include "../TestingUtils.h"

using namespace modelica::codegen::model;

TEST(EquationTest, EquationCopyAndClone)
{
	std::string stringModel = "model Eq1 "
														"Real x; "
														"equation "
														"x = 5; "
														"end Eq1; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	Equation eq1 = Equation(model.getEquations()[0]);
	Equation eq2 = model.getEquations()[0].clone();

	EXPECT_EQ(eq1, model.getEquations()[0]);
	EXPECT_EQ(eq1.getOp(), model.getEquations()[0].getOp());
	EXPECT_NE(eq2, model.getEquations()[0]);
	EXPECT_NE(eq2.getOp(), model.getEquations()[0].getOp());
}

TEST(EquationTest, EquationToIndexSet)
{
	std::string stringModel = "model Eq1 "
														"Real[3, 7] x; "
														"equation "
														"for i in 1:3 loop "
														"for j in 1:7 loop "
														"x[i, j] = 5; "
														"end for; "
														"end for; "
														"end Eq1; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	EXPECT_EQ(
			model.getEquations()[0].getInductions(),
			modelica::MultiDimInterval({ { 1, 4 }, { 1, 8 } }));
}

TEST(EquationTest, ExplicitateEquations)
{
	std::string stringModel = "model Eq2 "
														"Real x; "
														"Real y; "
														"Real z; "
														"equation "
														"5 = x; "
														"7 - y = x; "
														"x - z = y * 2; "
														"end Eq2; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	EXPECT_TRUE(model.getEquations()[0].lhs().isConstant());
	EXPECT_TRUE(model.getEquations()[0].rhs().isReferenceAccess());

	EXPECT_TRUE(model.getEquations()[1].lhs().isOperation());
	EXPECT_TRUE(model.getEquations()[1].rhs().isReferenceAccess());

	EXPECT_TRUE(model.getEquations()[2].lhs().isOperation());
	EXPECT_TRUE(model.getEquations()[2].rhs().isOperation());

	if (failed(match(model, 1000)))
		FAIL();

	for (Equation& eq : model.getEquations())
	{
		if (failed(eq.explicitate()))
			FAIL();
		EXPECT_TRUE(eq.lhs().isReferenceAccess());
		EXPECT_FALSE(eq.rhs().isReferenceAccess());
	}
}
