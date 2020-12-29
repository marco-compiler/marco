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

	Member xMember("x", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember("y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression condition = Expression(location, Type::Bool(), OperationKind::greater,
																		Expression(location, Type::Int(), ReferenceAccess("x")),
																		Expression(location, Type::Int(), Constant(0)));

	Statement thenStatement = AssignmentStatement(
			Expression(location, Type::Int(), ReferenceAccess("y")),
			Expression(location, Type::Int(), Constant(1)));

	Statement elseStatement = AssignmentStatement(
			Expression(location, Type::Int(), ReferenceAccess("y")),
			Expression(location, Type::Int(), Constant(2)));

	Statement ifStatement = IfStatement({
			IfStatement::Block(condition, { thenStatement }),
			IfStatement::Block(Expression::trueExp(location), { elseStatement })
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(ifStatement)));

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

	Member xMember("x", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember("y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression condition = Expression(location, Type::Bool(), OperationKind::greater,
																		Expression(location, Type::Int(), ReferenceAccess("x")),
																		Expression(location, Type::Int(), Constant(0)));

	Statement thenStatement = AssignmentStatement(
			Expression(location, Type::Int(), ReferenceAccess("y")),
			Expression(location, Type::Int(), Constant(1)));

	Statement elseStatement = AssignmentStatement(
			Expression(location, Type::Int(), ReferenceAccess("y")),
			Expression(location, Type::Int(), Constant(2)));

	Statement ifStatement = IfStatement({
																					IfStatement::Block(condition, { thenStatement }),
																					IfStatement::Block(Expression::trueExp(location), { elseStatement })
																			});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(ifStatement)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

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

	Member xMember("x", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember("y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression ifCondition = Expression(location, Type::Bool(), OperationKind::equal,
																		Expression(location, Type::Int(), ReferenceAccess("x")),
																		Expression(location, Type::Int(), Constant(0)));

	Statement thenStatement = AssignmentStatement(
			Expression(location, Type::Int(), ReferenceAccess("y")),
			Expression(location, Type::Int(), Constant(1)));

	Expression elseIfCondition = Expression(location, Type::Bool(), OperationKind::greater,
																				Expression(location, Type::Int(), ReferenceAccess("x")),
																				Expression(location, Type::Int(), Constant(0)));

	Statement elseIfStatement = AssignmentStatement(
			Expression(location, Type::Int(), ReferenceAccess("y")),
			Expression(location, Type::Int(), Constant(2)));

	Statement elseStatement = AssignmentStatement(
			Expression(location, Type::Int(), ReferenceAccess("y")),
			Expression(location, Type::Int(), Constant(3)));

	Statement ifStatement = IfStatement({
																					IfStatement::Block(ifCondition, { thenStatement }),
																					IfStatement::Block(elseIfCondition, { elseIfStatement }),
																					IfStatement::Block(Expression::trueExp(location), { elseStatement })
																			});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember },
			Algorithm(ifStatement)));

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

	Member xMember("x", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember("y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression condition = Expression(location, Type::Bool(), OperationKind::greater,
																		Expression(location, Type::Int(), ReferenceAccess("x")),
																		Expression(location, Type::Int(), Constant(0)));

	Expression xRef = Expression(location, Type::Int(), ReferenceAccess("x"));
	Expression yRef = Expression(location, Type::Int(), ReferenceAccess("y"));

	Statement whileStatement = WhileStatement(condition, {
			AssignmentStatement(yRef, Expression(location, Type::Int(), OperationKind::add, yRef, xRef)),
			AssignmentStatement(xRef, Expression(location, Type::Int(), OperationKind::subtract, xRef, Expression(location, Type::Int(), Constant(1))))
	});

	Algorithm algorithm = Algorithm({
			AssignmentStatement(yRef, Expression(location, Type::Int(), Constant(0))),
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

	Member yMember("y", Type::Int(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression condition = Expression::falseExp(location);
	Expression yRef = Expression(location, Type::Int(), ReferenceAccess("y"));

	Statement whileStatement = WhileStatement(condition, {
			AssignmentStatement(yRef, Expression(location, Type::Int(), Constant(0))),
			BreakStatement()
	});

	Algorithm algorithm = Algorithm({
																			AssignmentStatement(yRef, Expression(location, Type::Int(), Constant(1))),
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
