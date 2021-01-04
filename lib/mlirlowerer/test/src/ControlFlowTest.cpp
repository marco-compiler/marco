#include <gtest/gtest.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Member.hpp>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/Runner.hpp>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(IfOp, thenBranchTaken)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *   algorithm
	 *     if x > 0 then
	 *       y := 1;
	 *     else
	 *       y := 2;
	 *     end if;
	 * end main
	 */

	SourcePosition location("-", 0, 0);

	Member xMember(location, "x", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression condition = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::greater,
			Expression::reference(location, Type::Int(), "x"),
			Expression::constant(location, Type::Int(), 0));

	Statement thenStatement = AssignmentStatement(
			location,
			Expression::reference(location, Type::Int(), "y"),
			Expression::constant(location, Type::Int(), 1));

	Statement elseStatement = AssignmentStatement(
			location,
			Expression::reference(location, Type::Int(), "y"),
			Expression::constant(location, Type::Int(), 2));

	Statement ifStatement = IfStatement(location, {
			IfStatement::Block(condition, { thenStatement }),
			IfStatement::Block(Expression::constant(location, Type::Bool(), true), { elseStatement })
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(location, ifStatement)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int x = 57;
	int y = 0;
	runner.run("main", x, y);
	EXPECT_EQ(y, 1);
}

TEST(IfOp, elseBranchTaken)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *   algorithm
	 *     if x > 0 then
	 *       y := 1;
	 *     else
	 *       y := 2;
	 *     end if;
	 * end main
	 */

	SourcePosition location("-", 0, 0);

	Member xMember(location, "x", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression condition = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::greater,
			Expression::reference(location, Type::Int(), "x"),
			Expression::constant(location, Type::Int(), 0));

	Statement thenStatement = AssignmentStatement(
			location,
			Expression::reference(location, Type::Int(), "y"),
			Expression::constant(location, Type::Int(), 1));

	Statement elseStatement = AssignmentStatement(
			location,
			Expression::reference(location, Type::Int(), "y"),
			Expression::constant(location, Type::Int(), 2));

	Statement ifStatement = IfStatement(location, {
			IfStatement::Block(condition, { thenStatement }),
			IfStatement::Block(Expression::constant(location, Type::Bool(), true), { elseStatement })
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(location, ifStatement)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	Runner runner(&context, module);
	int x = -57;
	int y = 0;
	runner.run("main", x, y);
	EXPECT_EQ(y, 2);
}

TEST(IfOp, elseIfBranchTaken)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
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

	SourcePosition location("-", 0, 0);

	Member xMember(location, "x", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression ifCondition = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::equal,
			Expression::reference(location, Type::Int(), "x"),
			Expression::constant(location, Type::Int(), 0));

	Statement thenStatement = AssignmentStatement(
			location,
			Expression::reference(location, Type::Int(), "y"),
			Expression::constant(location, Type::Int(), 1));

	Expression elseIfCondition = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::greater,
			Expression::reference(location, Type::Int(), "x"),
			Expression::constant(location, Type::Int(), 0));

	Statement elseIfStatement = AssignmentStatement(
			location,
			Expression::reference(location, Type::Int(), "y"),
			Expression::constant(location, Type::Int(), 2));

	Statement elseStatement = AssignmentStatement(
			location,
			Expression::reference(location, Type::Int(), "y"),
			Expression::constant(location, Type::Int(), 3));

	Statement ifStatement = IfStatement(location, {
			IfStatement::Block(ifCondition, { thenStatement }),
			IfStatement::Block(elseIfCondition, { elseIfStatement }),
			IfStatement::Block(Expression::constant(location, Type::Bool(), true), { elseStatement })
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(location, ifStatement)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int x = 57;
	int y = 0;
	runner.run("main", x, y);
	EXPECT_EQ(y, 2);
}

TEST(WhileOp, validLoop)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *   algorithm
	 *     y := 0;
	 *     while x > 0 loop
	 *       y := y + x;
	 *       x := x - 1;
	 *     end while;
	 * end main
	 */

	SourcePosition location("-", 0, 0);

	Member xMember(location, "x", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression condition = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::greater,
			Expression::reference(location, Type::Int(), "x"),
			Expression::constant(location, Type::Int(), 0));

	Expression xRef = Expression::reference(location, Type::Int(), "x");
	Expression yRef = Expression::reference(location, Type::Int(), "y");

	Statement whileStatement = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::operation(location, Type::Int(), OperationKind::add, yRef, xRef)),
			AssignmentStatement(location, xRef, Expression::operation(location, Type::Int(), OperationKind::subtract, xRef, Expression::constant(location, Type::Int(), 1)))
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, Type::Int(), 0)),
			whileStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			algorithm));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int x = 10;
	int y = 0;
	runner.run("main", x, y);
	EXPECT_EQ(y, 55);
}

TEST(WhileOp, notExecutedLoop)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
	 *   algorithm
	 *     y := 1;
	 *     while false loop
	 *       y := 0;
	 *       break;
	 *     end while;
	 * end main
	 */

	SourcePosition location("-", 0, 0);

	Member yMember(location, "y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression condition = Expression::constant(location, Type::Bool(), false);
	Expression yRef = Expression::reference(location, Type::Int(), "y");

	Statement whileStatement = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, Type::Int(), 0)),
			BreakStatement(location)
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, Type::Int(), 1)),
			whileStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int y = 0;
	runner.run("main", y);
	EXPECT_EQ(y, 1);
}

TEST(BreakOp, nestedWhile)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
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

	SourcePosition location("-", 0, 0);

	Member yMember(location, "y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression condition = Expression::constant(location, Type::Bool(), true);
	Expression yRef = Expression::reference(location, Type::Int(), "y");

	Statement innerWhile = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, Type::Int(), 1)),
			BreakStatement(location)
	});

	Statement outerWhile = WhileStatement(location, condition, {
			innerWhile,
			BreakStatement(location)
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, Type::Int(), 0)),
			outerWhile
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int y = 0;
	runner.run("main", y);
	EXPECT_EQ(y, 1);
}

TEST(BreakOp, breakAsLastOpInWhile)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
	 *   algorithm
	 *     y := 0;
	 *     while true loop
	 *       y := 1;
	 *       break;
	 *     end while;
	 * end main
	 */

	SourcePosition location("-", 0, 0);

	Member yMember(location, "y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression condition = Expression::constant(location, Type::Bool(), true);
	Expression yRef = Expression::reference(location, Type::Int(), "y");

	Statement whileStatement = WhileStatement(location, condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, Type::Int(), 1)),
			BreakStatement(location)
	});

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, Type::Int(), 0)),
			whileStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int y = 0;
	runner.run("main", y);
	EXPECT_EQ(y, 1);
}

TEST(BreakOp, breakNestedInWhile)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
	 *   algorithm
	 *     y := 0;
	 *     while true loop
	 *       if y == 0 then
	 *         y := 1;
	 *         break;
	 *       end if;
	 *     end while;
	 * end main
	 */

	SourcePosition location("-", 0, 0);

	Member yMember(location, "y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression yRef = Expression::reference(location, Type::Int(), "y");
	Expression condition = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::equal,
			yRef,
			Expression::constant(location, Type::Int(), 0));

	Statement ifStatement = IfStatement(location, IfStatement::Block(condition, {
			AssignmentStatement(location, yRef, Expression::constant(location, Type::Int(), 1)),
			BreakStatement(location)
	}));

	Statement whileStatement = WhileStatement(location, Expression::constant(location, Type::Bool(), true), ifStatement);

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, Type::Int(), 0)),
			whileStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int y = 0;
	runner.run("main", y);
	EXPECT_EQ(y, 1);
}

TEST(ReturnOp, earlyReturn)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer y;
	 *   algorithm
	 *     y := 1;
   *     if true then
   *       return;
   *     end if;
	 *     y := 0;
	 * end main
	 */

	SourcePosition location("-", 0, 0);

	Member yMember(location, "y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression yRef = Expression::reference(location, Type::Int(), "y");
	Statement ifStatement = IfStatement(location, IfStatement::Block(Expression::constant(location, Type::Bool(), true), { ReturnStatement(location) }));

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, Type::Int(), 1)),
			ifStatement,
			AssignmentStatement(location, yRef, Expression::constant(location, Type::Int(), 0)),
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ yMember },
			algorithm));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int y = 0;
	runner.run("main", y);
	EXPECT_EQ(y, 1);
}
