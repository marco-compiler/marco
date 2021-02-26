#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Member.hpp>
#include <modelica/mlirlowerer/passes/BreakRemover.h>
#include <modelica/mlirlowerer/passes/ReturnRemover.h>
#include <modelica/mlirlowerer/MlirLowerer.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(IfOp, thenBranchTaken)	 // NOLINT
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

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression condition = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greater,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 0));

	Statement thenStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 1));

	Statement elseStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2));

	Statement ifStatement = IfStatement(location, {
			IfStatement::Block(condition, { thenStatement }),
			IfStatement::Block(Expression::constant(location, makeType<BuiltInType::Boolean>(), true), { elseStatement })
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(location, ifStatement)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int x = 1;
	int y = 0;

	Runner runner(module);
	runner.run("main", x, y);

	EXPECT_EQ(y, 1);
}

TEST(IfOp, elseBranchTaken)	 // NOLINT
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

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression condition = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greater,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 0));

	Statement thenStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 1));

	Statement elseStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2));

	Statement ifStatement = IfStatement(location, {
			IfStatement::Block(condition, { thenStatement }),
			IfStatement::Block(Expression::constant(location, makeType<BuiltInType::Boolean>(), true), { elseStatement })
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(location, ifStatement)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int x = -1;
	int y = 0;

	Runner runner(module);
	runner.run("main", x, y);

	EXPECT_EQ(y, 2);
}

TEST(IfOp, elseIfBranchTaken)	 // NOLINT
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

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression ifCondition = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::equal,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 0));

	Statement thenStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 1));

	Expression elseIfCondition = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greater,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 0));

	Statement elseIfStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2));

	Statement elseStatement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 3));

	Statement ifStatement = IfStatement(location, {
			IfStatement::Block(ifCondition, { thenStatement }),
			IfStatement::Block(elseIfCondition, { elseIfStatement }),
			IfStatement::Block(Expression::constant(location, makeType<BuiltInType::Boolean>(), true), { elseStatement })
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(location, ifStatement)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int x = 1;
	int y = 0;

	Runner runner(module);
	runner.run("main", x, y);
	EXPECT_EQ(y, 2);
}

TEST(ForOp, validLoop)	 // NOLINT
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

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement forAssignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "i"))
			);

	Statement forStatement = ForStatement(
			location,
			Induction(
					"i",
					Expression::constant(location, makeType<BuiltInType::Integer>(), 1),
					Expression::reference(location, makeType<BuiltInType::Integer>(), "x")),
			forAssignment);

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, Expression::reference(location, makeType<BuiltInType::Integer>(), "y"), Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			forStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			algorithm));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int x = 10;
	int y = 0;

	Runner runner(module);
	runner.run("main", x, y);

	EXPECT_EQ(y, 45);
}

TEST(ForOp, notExecutedLoop)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := 1;
	 *     for i in 1:x loop
	 *       y := y + i;
	 *     end for;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement forAssignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "i"))
	);

	Statement forStatement = ForStatement(
			location,
			Induction(
					"i",
					Expression::constant(location, makeType<BuiltInType::Integer>(), 1),
					Expression::reference(location, makeType<BuiltInType::Integer>(), "x")),
			forAssignment);

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, Expression::reference(location, makeType<BuiltInType::Integer>(), "y"), Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
			forStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			algorithm));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int x = 1;
	int y = 0;

	Runner runner(module);
	runner.run("main", x, y);

	EXPECT_EQ(y, 1);
}

TEST(WhileOp, validLoop)	 // NOLINT
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

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Member iMember(location, "i", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::none));

	Expression xRef = Expression::reference(location, makeType<BuiltInType::Integer>(), "x");
	Expression yRef = Expression::reference(location, makeType<BuiltInType::Integer>(), "y");
	Expression iRef = Expression::reference(location, makeType<BuiltInType::Integer>(), "i");

	Expression condition = Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::less, iRef, xRef);

	Statement whileStatement = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add, yRef, xRef)),
			AssignmentStatement(location, iRef, Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add, iRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 1)))
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, iRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			whileStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember, iMember },
			algorithm));

	BreakRemover breakRemover;
	breakRemover.fix(cls);

	ReturnRemover returnRemover;
	returnRemover.fix(cls);

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int x = 10;
	int y = 0;

	Runner runner(module);
	runner.run("main", x, y);

	EXPECT_EQ(y, 100);
}

TEST(WhileOp, notExecutedLoop)	 // NOLINT
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

	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression condition = Expression::constant(location, makeType<BuiltInType::Boolean>(), false);
	Expression yRef = Expression::reference(location, makeType<BuiltInType::Integer>(), "y");

	Statement whileStatement = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			BreakStatement(location)
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
			whileStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	BreakRemover breakRemover;
	breakRemover.fix(cls);

	ReturnRemover returnRemover;
	returnRemover.fix(cls);

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int y = 0;

	Runner runner(module);
	runner.run("main", y);

	EXPECT_EQ(y, 1);
}

TEST(BreakOp, nestedWhile)	 // NOLINT
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

	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression condition = Expression::constant(location, makeType<BuiltInType::Boolean>(), true);
	Expression yRef = Expression::reference(location, makeType<BuiltInType::Integer>(), "y");

	Statement innerWhile = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
			BreakStatement(location)
	});

	Statement outerWhile = WhileStatement(location, condition, {
			innerWhile,
			BreakStatement(location)
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			outerWhile
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	BreakRemover breakRemover;
	breakRemover.fix(cls);

	ReturnRemover returnRemover;
	returnRemover.fix(cls);

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int y = 0;

	Runner runner(module);
	runner.run("main", y);

	EXPECT_EQ(y, 1);
}

TEST(BreakOp, breakAsLastOpInWhile)	 // NOLINT
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

	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression condition = Expression::constant(location, makeType<BuiltInType::Boolean>(), true);
	Expression yRef = Expression::reference(location, makeType<BuiltInType::Integer>(), "y");

	Statement whileStatement = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
			BreakStatement(location)
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			whileStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	BreakRemover breakRemover;
	breakRemover.fix(cls);

	ReturnRemover returnRemover;
	returnRemover.fix(cls);

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int y = 0;

	Runner runner(module);
	runner.run("main", y);

	EXPECT_EQ(y, 1);
}

TEST(BreakOp, breakNestedInWhile)	 // NOLINT
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

	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression yRef = Expression::reference(location, makeType<BuiltInType::Integer>(), "y");

	Expression condition = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::equal,
			yRef,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 0));

	Statement ifStatement = IfStatement(location, IfStatement::Block(condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
			BreakStatement(location)
	}));

	Statement whileStatement = WhileStatement(
			location,
			Expression::constant(location, makeType<BuiltInType::Boolean>(), true),
			{
					ifStatement,
					AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 0))
			});

	Algorithm algorithm = Algorithm(
			location,
			{
					AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
					whileStatement
			});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	BreakRemover breakRemover;
	breakRemover.fix(cls);

	ReturnRemover returnRemover;
	returnRemover.fix(cls);

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int y = 0;

	Runner runner(module);
	runner.run("main", y);

	EXPECT_EQ(y, 1);
}

TEST(BreakOp, breakAsLastOpInFor)	 // NOLINT
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

	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement forAssignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 1)
	);

	Statement forStatement = ForStatement(
			location,
			Induction(
					"i",
					Expression::constant(location, makeType<BuiltInType::Integer>(), 1),
					Expression::constant(location, makeType<BuiltInType::Integer>(), 10)),
			forAssignment);

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, Expression::reference(location, makeType<BuiltInType::Integer>(), "y"), Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			forStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	BreakRemover breakRemover;
	breakRemover.fix(cls);

	ReturnRemover returnRemover;
	returnRemover.fix(cls);

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int y = 0;

	Runner runner(module);
	runner.run("main", y);

	EXPECT_EQ(y, 1);
}

TEST(BreakOp, breakNestedInFor)	 // NOLINT
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

	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression yRef = Expression::reference(location, makeType<BuiltInType::Integer>(), "y");

	Expression condition = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::equal,
			yRef,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 0));

	Statement ifStatement = IfStatement(location, IfStatement::Block(condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
			BreakStatement(location)
	}));

	Statement forStatement = ForStatement(
			location,
			Induction(
					"i",
					Expression::constant(location, makeType<BuiltInType::Integer>(), 1),
					Expression::constant(location, makeType<BuiltInType::Integer>(), 10)),
			{
					ifStatement,
					AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 0))
			});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			forStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	BreakRemover breakRemover;
	breakRemover.fix(cls);

	ReturnRemover returnRemover;
	returnRemover.fix(cls);

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int y = 0;

	Runner runner(module);
	runner.run("main", y);

	EXPECT_EQ(y, 1);
}

TEST(ReturnOp, earlyReturn)	 // NOLINT
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

	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression yRef = Expression::reference(location, makeType<BuiltInType::Integer>(), "y");
	Statement ifStatement = IfStatement(location, IfStatement::Block(Expression::constant(location, makeType<BuiltInType::Boolean>(), true), { ReturnStatement(location) }));

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
			ifStatement,
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	BreakRemover breakRemover;
	breakRemover.fix(cls);

	ReturnRemover returnRemover;
	returnRemover.fix(cls);

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int y = 0;

	Runner runner(module);
	runner.run("main", y);

	EXPECT_EQ(y, 1);
}
