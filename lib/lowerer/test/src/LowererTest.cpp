#include "gtest/gtest.h"
#include <system_error>

#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"
#include "modelica/lowerer/Lowerer.hpp"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModType.hpp"
#include "modelica/model/ModVariable.hpp"

using namespace modelica;
using namespace std;

TEST(lowererTest, backwardLoopTest)
{
	AssignModel model;

	if (!model.addVar(ModVariable(
					"leftVar",
					ModExp(ModConst(0, 1, 2, 3), ModType(BultinModTypes::INT, 4)))))

		FAIL();
	model.addUpdate(Assigment(
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 4)),
					ModExp::induction(ModConst(0))),
			ModConst(3),
			"",
			{ { 0, 3 } },
			false));

	llvm::LLVMContext context;
	Lowerer lowerer(context, move(model.getVars()), move(model.getUpdates()));
	EXPECT_TRUE(!lowerer.lower());
}
