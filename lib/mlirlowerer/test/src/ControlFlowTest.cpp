#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/SourcePosition.h>

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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression condition = Expression::operation(
			location,
			makeType<bool>(),
			OperationKind::greater,
			Expression::reference(location, makeType<int>(), "x"),
			Expression::constant(location, makeType<int>(), 0));

	Statement thenStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 1));

	Statement elseStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 2));

	Statement ifStatement = IfStatement(location, {
			IfStatement::Block(condition, { thenStatement }),
			IfStatement::Block(Expression::constant(location, makeType<bool>(), true), { elseStatement })
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(location, ifStatement)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression condition = Expression::operation(
			location,
			makeType<bool>(),
			OperationKind::greater,
			Expression::reference(location, makeType<int>(), "x"),
			Expression::constant(location, makeType<int>(), 0));

	Statement thenStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 1));

	Statement elseStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 2));

	Statement ifStatement = IfStatement(location, {
			IfStatement::Block(condition, { thenStatement }),
			IfStatement::Block(Expression::constant(location, makeType<bool>(), true), { elseStatement })
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(location, ifStatement)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression ifCondition = Expression::operation(
			location,
			makeType<bool>(),
			OperationKind::equal,
			Expression::reference(location, makeType<int>(), "x"),
			Expression::constant(location, makeType<int>(), 0));

	Statement thenStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 1));

	Expression elseIfCondition = Expression::operation(
			location,
			makeType<bool>(),
			OperationKind::greater,
			Expression::reference(location, makeType<int>(), "x"),
			Expression::constant(location, makeType<int>(), 0));

	Statement elseIfStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 2));

	Statement elseStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 3));

	Statement ifStatement = IfStatement(location, {
			IfStatement::Block(ifCondition, { thenStatement }),
			IfStatement::Block(elseIfCondition, { elseIfStatement }),
			IfStatement::Block(Expression::constant(location, makeType<bool>(), true), { elseStatement })
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(location, ifStatement)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement forAssignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														Expression::reference(location, makeType<int>(), "y"),
														Expression::reference(location, makeType<int>(), "i"))
			);

	Statement forStatement = ForStatement(
			location,
			Induction(
					"i",
					Expression::constant(location, makeType<int>(), 1),
					Expression::reference(location, makeType<int>(), "x")),
			forAssignment);

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, Expression::reference(location, makeType<int>(), "y"), Expression::constant(location, makeType<int>(), 0)),
			forStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			algorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement forAssignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														Expression::reference(location, makeType<int>(), "y"),
														Expression::reference(location, makeType<int>(), "i"))
	);

	Statement forStatement = ForStatement(
			location,
			Induction(
					"i",
					Expression::constant(location, makeType<int>(), 2),
					Expression::reference(location, makeType<int>(), "x")),
			forAssignment);

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, Expression::reference(location, makeType<int>(), "y"), Expression::constant(location, makeType<int>(), 1)),
			forStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			algorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Member iMember(location, "i", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::none));

	Expression xRef = Expression::reference(location, makeType<int>(), "x");
	Expression yRef = Expression::reference(location, makeType<int>(), "y");
	Expression iRef = Expression::reference(location, makeType<int>(), "i");

	Expression condition = Expression::operation(location, makeType<bool>(), OperationKind::less, iRef, xRef);

	Statement whileStatement = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::operation(location, makeType<int>(), OperationKind::add, yRef, xRef)),
			AssignmentStatement(location, iRef, Expression::operation(location, makeType<int>(), OperationKind::add, iRef, Expression::constant(location, makeType<int>(), 1)))
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, iRef, Expression::constant(location, makeType<int>(), 0)),
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 0)),
			whileStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember, iMember },
			algorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression condition = Expression::constant(location, makeType<bool>(), false);
	Expression yRef = Expression::reference(location, makeType<int>(), "y");

	Statement whileStatement = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 0)),
			BreakStatement(location)
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 1)),
			whileStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	PassManager passManager;
	passManager.addPass(createBreakRemovingPass());
	EXPECT_TRUE(!passManager.run(cls));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression condition = Expression::constant(location, makeType<bool>(), true);
	Expression yRef = Expression::reference(location, makeType<int>(), "y");

	Statement innerWhile = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 1)),
			BreakStatement(location)
	});

	Statement outerWhile = WhileStatement(location, condition, {
			innerWhile,
			BreakStatement(location)
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 0)),
			outerWhile
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	PassManager passManager;
	passManager.addPass(createBreakRemovingPass());
	EXPECT_TRUE(!passManager.run(cls));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression condition = Expression::constant(location, makeType<bool>(), true);
	Expression yRef = Expression::reference(location, makeType<int>(), "y");

	Statement whileStatement = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 1)),
			BreakStatement(location)
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 0)),
			whileStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	PassManager passManager;
	passManager.addPass(createBreakRemovingPass());
	EXPECT_TRUE(!passManager.run(cls));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression yRef = Expression::reference(location, makeType<int>(), "y");

	Expression condition = Expression::operation(
			location,
			makeType<bool>(),
			OperationKind::equal,
			yRef,
			Expression::constant(location, makeType<int>(), 0));

	Statement ifStatement = IfStatement(location, IfStatement::Block(condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 1)),
			BreakStatement(location)
	}));

	Statement whileStatement = WhileStatement(
			location,
			Expression::constant(location, makeType<bool>(), true),
			{
					ifStatement,
					AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 0))
			});

	Algorithm algorithm = Algorithm(
			location,
			{
					AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 0)),
					whileStatement
			});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	PassManager passManager;
	passManager.addPass(createBreakRemovingPass());
	EXPECT_TRUE(!passManager.run(cls));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement forAssignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::constant(location, makeType<int>(), 1)
	);

	Statement forStatement = ForStatement(
			location,
			Induction(
					"i",
					Expression::constant(location, makeType<int>(), 1),
					Expression::constant(location, makeType<int>(), 10)),
			forAssignment);

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, Expression::reference(location, makeType<int>(), "y"), Expression::constant(location, makeType<int>(), 0)),
			forStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	PassManager passManager;
	passManager.addPass(createBreakRemovingPass());
	EXPECT_TRUE(!passManager.run(cls));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression yRef = Expression::reference(location, makeType<int>(), "y");

	Expression condition = Expression::operation(
			location,
			makeType<bool>(),
			OperationKind::equal,
			yRef,
			Expression::constant(location, makeType<int>(), 0));

	Statement ifStatement = IfStatement(location, IfStatement::Block(condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 1)),
			BreakStatement(location)
	}));

	Statement forStatement = ForStatement(
			location,
			Induction(
					"i",
					Expression::constant(location, makeType<int>(), 1),
					Expression::constant(location, makeType<int>(), 10)),
			{
					ifStatement,
					AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 0))
			});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 0)),
			forStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	PassManager passManager;
	passManager.addPass(createBreakRemovingPass());
	EXPECT_TRUE(!passManager.run(cls));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

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

	SourcePosition location = SourcePosition::unknown();

	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression yRef = Expression::reference(location, makeType<int>(), "y");
	Statement ifStatement = IfStatement(location, IfStatement::Block(Expression::constant(location, makeType<bool>(), true), { ReturnStatement(location) }));

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 1)),
			ifStatement,
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 0)),
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	PassManager passManager;
	passManager.addPass(createReturnRemovingPass());
	EXPECT_TRUE(!passManager.run(cls));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(y))));
	EXPECT_EQ(y, 1);
}
