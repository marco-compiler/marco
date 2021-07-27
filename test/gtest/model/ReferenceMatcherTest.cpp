#include "gtest/gtest.h"
#include <modelica/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Model.h>

#include "../TestingUtils.h"

using namespace modelica::codegen::model;

TEST(ReferenceMatcherTest, ScalarRefVariableCollectorTest)
{
	std::string stringModel = "model Ref1 "
														"Real x; "
														"equation "
														"x = x + 3; "
														"end Ref1; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	ReferenceMatcher visitor(model.getEquations()[0]);

	EXPECT_EQ(visitor.size(), 2);
	EXPECT_TRUE(visitor.getExp(0).isReferenceAccess());
	EXPECT_TRUE(visitor.getExp(1).getChild(0).isReferenceAccess());

	EXPECT_EQ(visitor.getExp(0), model.getEquations()[0].lhs());
	EXPECT_EQ(visitor.getExp(1), model.getEquations()[0].rhs().getChild(0));

	EXPECT_EQ(
			visitor.getExp(0).getReferredVectorAccess(),
			model.getVariables()[0]->getReference());
	EXPECT_EQ(
			visitor.getExp(1).getReferredVectorAccess(),
			model.getVariables()[0]->getReference());
}

TEST(ReferenceMatcherTest, ArrayRefVariableCollectorTest)
{
	std::string stringModel = "model Ref2 "
														"Real x; "
														"Real[2, 2] y; "
														"equation "
														"x = -y[1, 1]; "
														"for i in 1:2 loop "
														"for j in 1:2 loop "
														"y[i, j] = 2; "
														"end for; "
														"end for; "
														"end Ref2; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	ReferenceMatcher visitor(model.getEquations()[0]);

	EXPECT_EQ(visitor.size(), 2);
	EXPECT_TRUE(visitor.getExp(0).isReferenceAccess());
	EXPECT_TRUE(visitor.getExp(1).getChild(0).isReferenceAccess());

	EXPECT_EQ(visitor.getExp(0), model.getEquations()[0].lhs());
	EXPECT_EQ(visitor.getExp(1), model.getEquations()[0].rhs().getChild(0));

	EXPECT_EQ(
			visitor.getExp(0).getReferredVectorAccess(),
			model.getVariables()[0]->getReference());
	EXPECT_EQ(
			visitor.getExp(1).getChild(0).getReferredVectorAccess(),
			model.getVariables()[1]->getReference());
}
