#include "gtest/gtest.h"
#include <marco/mlirlowerer/passes/SolveModel.h>
#include <marco/mlirlowerer/passes/model/BltBlock.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>

#include "../TestingUtils.h"

using namespace marco::codegen::model;

TEST(SolveModelTest, SubstituteTrivialVariablesTest)
{
	std::string stringModel = "model SubstituteTrivial "
														"Real a; "
														"Real[2] b; "
														"Real c; "
														"Real[4] x; "
														"Real[6] y; "
														"equation "
														"a = 3; "
														"for i in 1:2 loop "
														"a = b[i] * 2; "
														"end for; "
														"c + a = b[1] + 2; "
														"for i in 1:4 loop "
														"der(x[i]) = c + b[2] + i; "
														"end for; "
														"for i in 1:6 loop "
														"y[i] + y[i]*y[i] - i = a * b[1]; "
														"end for; "
														"end SubstituteTrivial; ";

	mlir::MLIRContext context;
	Model model;
	makeSolvedModel(context, stringModel, model);

	EXPECT_EQ(model.getVariables().size(), 6);
	EXPECT_EQ(model.getEquations().size(), 3);
	EXPECT_EQ(model.getBltBlocks().size(), 2);
	EXPECT_EQ(model.getBltBlocks()[0].getEquations().size(), 1);
	EXPECT_EQ(model.getBltBlocks()[1].getEquations().size(), 1);

	std::map<Variable, Equation*> trivialVariablesMap;
	for (Equation& eq : model.getEquations())
	{
		Variable var = model.getVariable(eq.getDeterminedVariable().getVar());
		trivialVariablesMap[var] = &eq;
		EXPECT_TRUE(var.isTrivial());
	}

	for (BltBlock& bltBlock : model.getBltBlocks())
	{
		for (Equation& eq : bltBlock.getEquations())
		{
			Variable var = model.getVariable(eq.getDeterminedVariable().getVar());
			EXPECT_FALSE(var.isTrivial());

			ReferenceMatcher matcher(eq);
			for (ExpressionPath expPath : matcher)
			{
				EXPECT_TRUE(expPath.getExpression().isReferenceAccess());
				var = model.getVariable(
						expPath.getExpression().getReferredVectorAccess());
				EXPECT_TRUE(
						!var.isTrivial() ||
						trivialVariablesMap.find(var) == trivialVariablesMap.end());
			}
		}
	}
}
