#include <gtest/gtest.h>
#include <mlir/IR/Dialect.h>
#include <modelica/frontend/ConstantFolder.hpp>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/TypeChecker.hpp>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/Runner.hpp>
#include <modelica/utils/SourceRange.hpp>

using namespace mlir;
using namespace modelica;
using namespace std;

TEST(MlirLowererTest, constantAssignment)	 // NOLINT
{
	string source = "function main"
									"  output Integer x;"
									"  algorithm"
									"    x := 57;"
									"end main";

	Parser parser(source);
	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto& cls = *expectedAst;

	modelica::TypeChecker typeChecker;

	if (typeChecker.checkType(cls, modelica::SymbolTable()))
		FAIL();

	modelica::ConstantFolder folder;

	if (folder.fold(cls, modelica::SymbolTable()))
		FAIL();

	MLIRContext context;
	MlirLowerer lowerer(context);
	ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);

	int x = 0;
	auto execution = runner.run("main", x);

	if (failed(execution))
		FAIL();

	EXPECT_EQ(x, 57);
}

TEST(MlirLowererTest, variableAssignment)	 // NOLINT
{
	string source = "function main"
									"  input Integer x;"
									"  output Integer y;"
									"algorithm"
									"  y := x;"
									"end main";

	Parser parser(source);
	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto& cls = *expectedAst;

	modelica::TypeChecker typeChecker;

	if (typeChecker.checkType(cls, modelica::SymbolTable()))
		FAIL();

	modelica::ConstantFolder folder;

	if (folder.fold(cls, modelica::SymbolTable()))
		FAIL();

	MLIRContext context;
	MlirLowerer lowerer(context);
	ModuleOp module = lowerer.lower({ cls });

	Runner runner(&context, module);

	int x = 57;
	int y = 0;
	auto execution = runner.run("main", x, y);

	if (failed(execution))
		FAIL();

	EXPECT_EQ(x, y);
}

TEST(MlirLowererTest, arrayElementAssignment)	 // NOLINT
{
	//llvm::DebugFlag = true;

	string source = "function main"
									"  output Integer y;"
									"  protected"
									"    Integer[3] z;"
									"  algorithm"
									"    z[1] := 57;"
									"    y := z[1];"
									"  end main";

	Parser parser(source);
	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto& cls = *expectedAst;

	modelica::TypeChecker typeChecker;

	if (typeChecker.checkType(cls, modelica::SymbolTable()))
		FAIL();

	modelica::ConstantFolder folder;

	if (folder.fold(cls, modelica::SymbolTable()))
		FAIL();

	MLIRContext context;
	MlirLowerer lowerer(context);
	ModuleOp module = lowerer.lower({ cls });
	module.dump();

	Runner runner(&context, module);

	int x[4] =  { 0, 0, 0, 0 };
	int y = 0;
	auto execution = runner.run("main", y);

	if (failed(execution))
		FAIL();

	EXPECT_EQ(y, 57);
}
