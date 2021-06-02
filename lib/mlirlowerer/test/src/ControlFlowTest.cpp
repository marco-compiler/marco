#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/runtime/ArrayDescriptor.h>

using namespace modelica;
using namespace frontend;
using namespace codegen;
using namespace std;

TEST(ControlFlow, breakAsLastOpInWhile)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := 0;
	 *     while true loop
	 *       y := 1;
	 *       break;
	 *     end while;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto condition = Expression::constant(location, makeType<bool>(), true);
	auto yRef = Expression::reference(location, makeType<int>(), "y");

	auto whileStatement = Statement::whileStatement(
			location, std::move(condition),
			llvm::ArrayRef({
					Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 1)),
					Statement::breakStatement(location)
			}));

	auto algorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 0)),
										std::move(whileStatement)
								}));

	auto cls = Class::standardFunction(
			location, true, "main",
			std::move(yMember),
			std::move(algorithm));

	PassManager passManager;
	passManager.addPass(createBreakRemovingPass());
	EXPECT_TRUE(!passManager.run(cls));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(y))));
	EXPECT_EQ(y, 1);
}

TEST(ControlFlow, breakAsLastOpInFor)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := 0;
	 *     for i in 1:10 loop
	 *       y := 1;
	 *       break;
	 *     end for;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto forAssignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 1)
	);

	auto forStatement = Statement::forStatement(
			location,
			Induction::build(
					location, "i",
					Expression::constant(location, makeType<int>(), 1),
					Expression::constant(location, makeType<int>(), 10)),
			std::move(forAssignment));

	auto algorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, Expression::reference(location, makeType<int>(), "y"), Expression::constant(location, makeType<int>(), 0)),
										std::move(forStatement)
								}));

	auto cls = Class::standardFunction(
			location, true, "main",
			std::move(yMember),
			std::move(algorithm));

	PassManager passManager;
	passManager.addPass(createBreakRemovingPass());
	EXPECT_TRUE(!passManager.run(cls));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(y))));
	EXPECT_EQ(y, 1);
}

TEST(ControlFlow, earlyReturn)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := 1;
   *     if true then
   *       return;
   *     end if;
	 *     y := 0;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto yRef = Expression::reference(location, makeType<int>(), "y");

	auto ifStatement = Statement::ifStatement(
			location, IfStatement::Block(Expression::constant(location, makeType<bool>(), true), Statement::returnStatement(location)));

	auto algorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 1)),
										std::move(ifStatement),
										Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 0)),
								}));

	auto cls = Class::standardFunction(
			location, true, "main",
			std::move(yMember),
			std::move(algorithm));

	PassManager passManager;
	passManager.addPass(createReturnRemovingPass());
	EXPECT_TRUE(!passManager.run(cls));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(y))));
	EXPECT_EQ(y, 1);
}

TEST(ControlFlow, allocationsInsideLoop)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[2] x;
	 *   output Integer y;
	 *
	 *   protected
	 *     Integer[:] z;
	 *
	 *   algorithm
	 *     z := x;
	 *     for i in 1:10 loop
	 *       z := z * 2;
	 *     end for;
	 *     y := sum(z);
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto zMember = Member::build(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::none));

	auto xRef = Expression::reference(location, makeType<int>(2), "x");
	auto yRef = Expression::reference(location, makeType<int>(), "y");
	auto zRef = Expression::reference(location, makeType<int>(2), "z");

	auto condition = Expression::operation(
			location, makeType<bool>(), OperationKind::equal,
			llvm::ArrayRef({
					yRef->clone(),
					Expression::constant(location, makeType<int>(), 0)
			}));

	auto ifStatement = Statement::ifStatement(
			location, IfStatement::Block(
										std::move(condition),
										llvm::ArrayRef({
												Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 1)),
												Statement::breakStatement(location)
										})));

	auto forStatement = Statement::forStatement(
			location,
			Induction::build(
					location, "i",
					Expression::constant(location, makeType<int>(), 1),
					Expression::constant(location, makeType<int>(), 10)),
			llvm::ArrayRef({
					Statement::assignmentStatement(location, zRef->clone(),
																				 Expression::operation(location, makeType<int>(-1), OperationKind::multiply,
																															 llvm::ArrayRef({
																																	 zRef->clone(),
																																	 Expression::constant(location, makeType<int>(), 2)
																															 })))
			}));

	auto algorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, zRef->clone(), xRef->clone()),
										std::move(forStatement),
										Statement::assignmentStatement(
												location, yRef->clone(),
												Expression::call(location, makeType<int>(), Expression::reference(location, makeType<int>(), "sum"), zRef->clone()))
								}));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			std::move(algorithm));

	PassManager passManager;
	passManager.addPass(createBreakRemovingPass());
	EXPECT_TRUE(!passManager.run(cls));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	std::array<int, 2> x = { 1, 2 };
	ArrayDescriptor<int, 1> xDesc(x);

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, jit::Runner::result(y))));
	EXPECT_EQ(y, 3072);
}