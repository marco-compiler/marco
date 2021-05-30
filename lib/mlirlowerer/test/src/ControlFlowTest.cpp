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

TEST(ControlFlow, thenBranchTaken)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     if x > 0 then
	 *       y := 1;
	 *     else
	 *       y := 2;
	 *     end if;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto condition = Expression::operation(
			location, makeType<bool>(), OperationKind::greater,
			llvm::ArrayRef({
					Expression::reference(location, makeType<int>(), "x"),
					Expression::constant(location, makeType<int>(), 0)
			}));

	auto thenStatement = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 1));

	auto elseStatement = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 2));

	auto ifStatement = Statement::ifStatement(
			location, llvm::ArrayRef({
										IfStatement::Block(std::move(condition), std::move(thenStatement)),
										IfStatement::Block(Expression::constant(location, makeType<bool>(), true), std::move(elseStatement))
								}));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, std::move(ifStatement)));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	int x = 1;
	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
	EXPECT_EQ(y, 1);
}

TEST(ControlFlow, elseBranchTaken)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     if x > 0 then
	 *       y := 1;
	 *     else
	 *       y := 2;
	 *     end if;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto condition = Expression::operation(
			location,
			makeType<bool>(),
			OperationKind::greater,
			llvm::ArrayRef({
					Expression::reference(location, makeType<int>(), "x"),
					Expression::constant(location, makeType<int>(), 0)
			}));

	auto thenStatement = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 1));

	auto elseStatement = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 2));

	auto ifStatement = Statement::ifStatement(
			location, llvm::ArrayRef({
										IfStatement::Block(std::move(condition), std::move(thenStatement)),
										IfStatement::Block(Expression::constant(location, makeType<bool>(), true), std::move(elseStatement))
								}));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, std::move(ifStatement)));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	int x = -1;
	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
	EXPECT_EQ(y, 2);
}

TEST(ControlFlow, elseIfBranchTaken)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     if x == 0 then
	 *       y := 1;
	 *     elseif x > 0 then
	 *       y := 2;
	 *     else
	 *       y := 3;
	 *     end if;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto ifCondition = Expression::operation(
			location,
			makeType<bool>(),
			OperationKind::equal,
			llvm::ArrayRef({
					Expression::reference(location, makeType<int>(), "x"),
					Expression::constant(location, makeType<int>(), 0)
			}));

	auto thenStatement = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 1));

	auto elseIfCondition = Expression::operation(
			location,
			makeType<bool>(),
			OperationKind::greater,
			llvm::ArrayRef({
					Expression::reference(location, makeType<int>(), "x"),
					Expression::constant(location, makeType<int>(), 0)
			}));

	auto elseIfStatement = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 2));

	auto elseStatement = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 3));

	auto ifStatement = Statement::ifStatement(
			location, llvm::ArrayRef({
			IfStatement::Block(std::move(ifCondition), std::move(thenStatement)),
			IfStatement::Block(std::move(elseIfCondition), std::move(elseIfStatement)),
			IfStatement::Block(Expression::constant(location, makeType<bool>(), true), std::move(elseStatement))
	}));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, std::move(ifStatement)));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	int x = 1;
	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
	EXPECT_EQ(y, 2);
}

TEST(ControlFlow, forLoop)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := 0;
	 *     for i in 1:x loop
	 *       y := y + i;
	 *     end for;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto forAssignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "y"),
																Expression::reference(location, makeType<int>(), "i")
														})));

	auto forStatement = Statement::forStatement(
			location,
			Induction::build(
					location, "i",
					Expression::constant(location, makeType<int>(), 1),
					Expression::reference(location, makeType<int>(), "x")),
			std::move(forAssignment));

	auto algorithm = Algorithm::build(
			location, llvm::ArrayRef({
			Statement::assignmentStatement(location, Expression::reference(location, makeType<int>(), "y"), Expression::constant(location, makeType<int>(), 0)),
			std::move(forStatement)
	}));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			std::move(algorithm));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	int x = 10;
	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
	EXPECT_EQ(y, 55);
}

TEST(ControlFlow, forNotExecuted)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := 1;
	 *     for i in 2:x loop
	 *       y := y + i;
	 *     end for;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto forAssignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "y"),
																Expression::reference(location, makeType<int>(), "i")
														})));

	auto forStatement = Statement::forStatement(
			location,
			Induction::build(
					location, "i",
					Expression::constant(location, makeType<int>(), 2),
					Expression::reference(location, makeType<int>(), "x")),
			std::move(forAssignment));

	auto algorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, Expression::reference(location, makeType<int>(), "y"), Expression::constant(location, makeType<int>(), 1)),
										std::move(forStatement)
								}));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			std::move(algorithm));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	int x = 1;
	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
	EXPECT_EQ(y, 1);
}

TEST(ControlFlow, whileLoop)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   protected
	 *     Integer i;
	 *
	 *   algorithm
	 *     y := 0;
	 *     i := 0;
	 *     while i < x loop
	 *       y := y + x;
	 *       i := i + 1;
	 *     end while;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto iMember = Member::build(location, "i", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::none));

	auto xRef = Expression::reference(location, makeType<int>(), "x");
	auto yRef = Expression::reference(location, makeType<int>(), "y");
	auto iRef = Expression::reference(location, makeType<int>(), "i");

	auto condition = Expression::operation(
			location, makeType<bool>(), OperationKind::less,
			llvm::ArrayRef({ iRef->clone(), xRef->clone() }));

	auto whileStatement = Statement::whileStatement(
			location, std::move(condition),
			llvm::ArrayRef({
					Statement::assignmentStatement(location, yRef->clone(), Expression::operation(location, makeType<int>(), OperationKind::add, llvm::ArrayRef({ yRef->clone(), xRef->clone() }))),
					Statement::assignmentStatement(location, iRef->clone(), Expression::operation(location, makeType<int>(), OperationKind::add, llvm::ArrayRef({ iRef->clone(), Expression::constant(location, makeType<int>(), 1) })))
			}));

	auto algorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, iRef->clone(), Expression::constant(location, makeType<int>(), 0)),
										Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 0)),
										std::move(whileStatement)
								}));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(iMember) }),
			std::move(algorithm));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	int x = 10;
	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
	EXPECT_EQ(y, 100);
}

TEST(ControlFlow, whileNotExecuted)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := 1;
	 *     while false loop
	 *       y := 0;
	 *       break;
	 *     end while;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto condition = Expression::constant(location, makeType<bool>(), false);
	auto yRef = Expression::reference(location, makeType<int>(), "y");

	auto whileStatement = Statement::whileStatement(
			location, std::move(condition),
			llvm::ArrayRef({
					Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 0)),
					Statement::breakStatement(location)
			}));

	auto algorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 1)),
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

TEST(ControlFlow, breakInInnermostWhile)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := 0;
	 *     while true loop
	 *       while true loop
	 *         y := 1;
	 *         break;
	 *       end while;
	 *       break;
	 *     end while;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto condition = Expression::constant(location, makeType<bool>(), true);
	auto yRef = Expression::reference(location, makeType<int>(), "y");

	auto innerWhile = Statement::whileStatement(
			location, condition->clone(),
			llvm::ArrayRef({
					Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 1)),
					Statement::breakStatement(location)
			}));

	auto outerWhile = Statement::whileStatement(
			location, condition->clone(),
			llvm::ArrayRef({
					std::move(innerWhile),
					Statement::breakStatement(location)
			}));

	auto algorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 0)),
										std::move(outerWhile)
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

TEST(ControlFlow, breakNestedInWhile)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := 0;
	 *     while true loop
	 *       if y == 0 then
	 *         y := 1;
	 *         break;
	 *       end if;
	 *       y := 0;
	 *     end while;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto yRef = Expression::reference(location, makeType<int>(), "y");

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

	auto whileStatement = Statement::whileStatement(
			location,
			Expression::constant(location, makeType<bool>(), true),
			llvm::ArrayRef({
					std::move(ifStatement),
					Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 0))
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

TEST(ControlFlow, breakNestedInFor)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := 0;
	 *     for i in 1:10 loop
	 *       if y == 0 then
	 *         y := 1;
	 *         break;
	 *       end if;
	 *       y := 0;
	 *     end for;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto yRef = Expression::reference(location, makeType<int>(), "y");

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
					std::move(ifStatement),
					Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 0))
			}));

	auto algorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 0)),
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