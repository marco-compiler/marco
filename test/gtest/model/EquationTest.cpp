#include "gtest/gtest.h"
#include <marco/mlirlowerer/passes/matching/Matching.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Model.h>

#include "../TestingUtils.h"

using namespace marco::codegen::model;

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
			marco::MultiDimInterval({ { 1, 4 }, { 1, 8 } }));
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

TEST(EquationTest, ImplicitEquations)
{
	std::string stringModel = "model ImplicitEq "
														"Real x; "
														"Real y; "
														"Real z; "
														"Real u; "
														"Real v; "
														"equation "
														"3 = x; "
														"2 * y + 2 = x + 3; "
														"z*z*z - z + 2 = 3; "
														"u - 1/u + 2 = 3; "
														"5 = v - 2 * sin(v); "
														"end ImplicitEq; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_TRUE(model.getEquations()[0].isImplicit());
	EXPECT_TRUE(model.getEquations()[1].isImplicit());
	EXPECT_TRUE(model.getEquations()[2].isImplicit());
	EXPECT_TRUE(model.getEquations()[3].isImplicit());
	EXPECT_TRUE(model.getEquations()[4].isImplicit());

	EXPECT_FALSE(failed(model.getEquations()[0].explicitate()));
	EXPECT_FALSE(failed(model.getEquations()[1].explicitate()));
	EXPECT_TRUE(failed(model.getEquations()[2].explicitate()));
	EXPECT_TRUE(failed(model.getEquations()[3].explicitate()));
	EXPECT_TRUE(failed(model.getEquations()[4].explicitate()));

	EXPECT_FALSE(model.getEquations()[0].isImplicit());
	EXPECT_FALSE(model.getEquations()[1].isImplicit());
	EXPECT_TRUE(model.getEquations()[2].isImplicit());
	EXPECT_TRUE(model.getEquations()[3].isImplicit());
	EXPECT_TRUE(model.getEquations()[4].isImplicit());

	EXPECT_TRUE(model.getEquations()[0].lhs().isReferenceAccess());
	EXPECT_TRUE(model.getEquations()[1].lhs().isReferenceAccess());
	EXPECT_TRUE(model.getEquations()[2].rhs().isConstant());
	EXPECT_TRUE(model.getEquations()[3].rhs().isConstant());
	EXPECT_TRUE(model.getEquations()[4].rhs().isOperation());
}
