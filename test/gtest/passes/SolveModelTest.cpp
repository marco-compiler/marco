#include "gtest/gtest.h"
#include <mlir/Pass/PassManager.h>
#include <modelica/frontend/Parser.h>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/passes/SolveModel.h>
#include <modelica/mlirlowerer/passes/model/BltBlock.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <queue>

using namespace modelica::codegen::model;

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

	modelica::frontend::Parser parser(stringModel);
	auto ast = parser.classDefinition();
	if (!ast)
		FAIL();

	llvm::SmallVector<std::unique_ptr<modelica::frontend::Class>, 3> classes;
	classes.push_back(std::move(*ast));

	modelica::frontend::PassManager frontendPassManager;
	frontendPassManager.addPass(modelica::frontend::createTypeCheckingPass());
	frontendPassManager.addPass(modelica::frontend::createConstantFolderPass());

	if (frontendPassManager.run(classes))
		FAIL();

	modelica::codegen::ModelicaBuilder builder(&context);

	modelica::codegen::MLIRLowerer lowerer(context);
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	modelica::codegen::SolveModelOptions solveModelOptions;
	solveModelOptions.solver = modelica::codegen::CleverDAE;

	auto model = modelica::codegen::getSolvedModel(*moduleOp, solveModelOptions);
	if (!model)
		FAIL();

	EXPECT_EQ(model->getVariables().size(), 5 + 1 + 1);
	EXPECT_EQ(model->getEquations().size(), 3);
	EXPECT_EQ(model->getBltBlocks().size(), 2);
	EXPECT_EQ(model->getBltBlocks()[0].getEquations().size(), 1);
	EXPECT_EQ(model->getBltBlocks()[1].getEquations().size(), 1);

	std::map<Variable, Equation*> trivialVariablesMap;
	for (Equation& eq : model->getEquations())
	{
		Variable var = model->getVariable(eq.getDeterminedVariable().getVar());
		trivialVariablesMap[var] = &eq;
		EXPECT_TRUE(var.isTrivial());
	}

	for (BltBlock& bltBlock : model->getBltBlocks())
	{
		for (Equation& eq : bltBlock.getEquations())
		{
			Variable var = model->getVariable(eq.getDeterminedVariable().getVar());
			EXPECT_FALSE(var.isTrivial());

			std::queue<Expression> expQueue({ eq.lhs(), eq.rhs() });
			while (!expQueue.empty())
			{
				if (expQueue.front().isReferenceAccess())
				{
					var = model->getVariable(expQueue.front().getReferredVectorAccess());
					EXPECT_TRUE(
							!var.isTrivial() ||
							trivialVariablesMap.find(var) == trivialVariablesMap.end());
				}
				else if (expQueue.front().isOperation())
				{
					for (size_t i : modelica::irange(expQueue.front().childrenCount()))
						expQueue.push(expQueue.front().getChild(i));
				}

				expQueue.pop();
			}
		}
	}
}
