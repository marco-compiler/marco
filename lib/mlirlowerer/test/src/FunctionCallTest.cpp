#include <gtest/gtest.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/Dialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/runtime/ArrayDescriptor.h>
#include <modelica/utils/SourcePosition.h>

using namespace modelica;
using namespace frontend;
using namespace codegen;
using namespace std;

TEST(Function, callNoArguments)	 // NOLINT
{
	/**
	 * function foo
	 *   output Integer x;
	 *
	 *   algorithm
	 *   	 x := 1;
	 * end foo
	 *
	 * function main
	 *   output Integer x;
	 *
	 *   algorithm
	 *     x := foo();
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();
	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto xRef = Expression::reference(location, makeType<int>(), "x");

	auto fooAlgorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, xRef->clone(), Expression::constant(location, makeType<int>(), 1))
								}));

	auto foo = Class::standardFunction(location, true, "foo", llvm::None, xMember->clone(), std::move(fooAlgorithm));

	auto mainAlgorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, xRef->clone(),
																									 Expression::call(location, makeType<int>(), Expression::reference(location, makeType<int>(), "foo"), llvm::None))
								}));

	auto main = Class::standardFunction(location, true, "main", llvm::None, xMember->clone(), std::move(mainAlgorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(llvm::ArrayRef({ *foo, *main }));

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(y))));
	EXPECT_EQ(y, 1);
}

TEST(Function, recursiveCall)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[3] x;
	 *   input Integer i;
	 *   output Real y;
	 *
	 *   algorithm
	 *     y := 0
	 *     if i <= 3 then
	 *       y := x[i] + main(x, i + 1);
	 *     end if;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto iMember = Member::build(location, "i", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto xRef = Expression::reference(location, makeType<float>(3), "x");
	auto iRef = Expression::reference(location, makeType<int>(), "i");
	auto yRef = Expression::reference(location, makeType<float>(), "y");

	auto condition = Expression::operation(location, makeType<bool>(), OperationKind::lessEqual,
																				 llvm::ArrayRef({
																						 iRef->clone(),
																						 Expression::constant(location, makeType<int>(), 3)
																				 }));

	auto sum = Expression::operation(
			location, makeType<float>(), OperationKind::add,
			llvm::ArrayRef({
					Expression::operation(location, makeType<float>(), OperationKind::subscription,
																llvm::ArrayRef({
																		xRef->clone(),
																		Expression::operation(location, makeType<int>(), OperationKind::subtract,
																													llvm::ArrayRef({ iRef->clone(), Expression::constant(location, makeType<int>(), 1) }))
																})),
					Expression::call(location, makeType<float>(), Expression::reference(location, makeType<float>(), "main"),
													 llvm::ArrayRef({
															 xRef->clone(),
															 Expression::operation(location, makeType<int>(), OperationKind::add,
																										 llvm::ArrayRef({
																												 iRef->clone(),
																												 Expression::constant(location, makeType<int>(), 1)
																										 }))
													 }))
			}));

	auto ifStatement = Statement::ifStatement(
			location, IfStatement::Block(
										std::move(condition),
										llvm::ArrayRef({
												Statement::assignmentStatement(location, yRef->clone(), std::move(sum))
										})));

	auto algorithm = Algorithm::build(
			location, llvm::ArrayRef({
										Statement::assignmentStatement(location, yRef->clone(), Expression::constant(location, makeType<int>(), 0)),
										std::move(ifStatement)
								}));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(iMember) }),
			std::move(algorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<float, 3> x = { 1, 2, 3 };
	ArrayDescriptor<float, 1> xDesc(x);

	int i = 1;
	float y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, i, jit::Runner::result(y))));

	float expected = 0;

	for (float value : x)
		expected += value;

	EXPECT_EQ(y, expected);
}

TEST(Function, callWithStaticArrayAsOutput)	 // NOLINT
{
	/**
	 * function foo
	 *   output Integer[3] x;
	 *
	 *   algorithm
	 *   	 x[1] := 1;
	 *   	 x[2] := 2;
	 *   	 x[3] := 3;
	 * end foo
	 *
	 * function main
	 *   output Integer[3] x;
	 *
	 *   algorithm
	 *     x := foo();
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto xRef = Expression::reference(location, makeType<int>(3), "x");

	auto fooAlgorithm = Algorithm::build(
			location,
			llvm::ArrayRef({
					Statement::assignmentStatement(
							location,
							Expression::operation(location, makeType<int>(), OperationKind::subscription,
																		llvm::ArrayRef({
																				xRef->clone(),
																				Expression::constant(location, makeType<int>(), 0)
																		})),
							Expression::constant(location, makeType<int>(), 1)),
					Statement::assignmentStatement(
							location,
							Expression::operation(location, makeType<int>(), OperationKind::subscription,
																		llvm::ArrayRef({
																				xRef->clone(),
																				Expression::constant(location, makeType<int>(), 1)
																		})),
							Expression::constant(location, makeType<int>(), 2)),
					Statement::assignmentStatement(
							location,
							Expression::operation(location, makeType<int>(), OperationKind::subscription,
																		llvm::ArrayRef({
																				xRef->clone(),
																				Expression::constant(location, makeType<int>(), 2)
																		})),
							Expression::constant(location, makeType<int>(), 3))
			}));

	auto foo = Class::standardFunction(location, true, "foo", llvm::None, xMember->clone(), std::move(fooAlgorithm));

	auto mainAlgorithm = Algorithm::build(
			location,
			Statement::assignmentStatement(
					location,
					xRef->clone(),
					Expression::call(location, makeType<int>(3), Expression::reference(location, makeType<int>(3), "foo"), llvm::None)));

	auto main = Class::standardFunction(location, true, "main", llvm::None, xMember->clone(), std::move(mainAlgorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run({ *main, *foo });

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 0, 0, 0 };
	ArrayDescriptor<int, 1> xDesc(x);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc)));
	EXPECT_EQ(xDesc[0], 1);
	EXPECT_EQ(xDesc[1], 2);
	EXPECT_EQ(xDesc[2], 3);
}

TEST(Function, callWithDynamicArrayAsOutput)	 // NOLINT
{
	/**
	 * function foo
	 *   input Real x;
	 *   input Integer n;
	 *   output Real y[n];
	 *
	 *   algorithm
	 *     for i in 1:n loop
	 *   	 	 y[i] := x * i;
	 *   	 end for;
	 * end foo
	 *
	 * function main
	 *   input
	 *   output Real[3] x;
	 *
	 *   algorithm
	 *     x := foo(2, 3);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	llvm::SmallVector<std::unique_ptr<Member>, 3> fooMembers;
	fooMembers.push_back(Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input)));
	fooMembers.push_back(Member::build(location, "n", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input)));
	fooMembers.push_back(Member::build(location, "y", makeType<float>(Expression::reference(location, makeType<int>(), "n")), TypePrefix(ParameterQualifier::none, IOQualifier::output)));

	auto induction = Induction::build(
			location, "i",
			Expression::constant(location, makeType<int>(), 1),
			Expression::reference(location, makeType<int>(), "n"));

	auto fooAlgorithm = Algorithm::build(
			location,
			Statement::forStatement(
					location, std::move(induction),
					Statement::assignmentStatement(
							location,
							Expression::operation(location, makeType<float>(), OperationKind::subscription,
																		llvm::ArrayRef({
																				Expression::reference(location, makeType<float>(-1), "y"),
																				Expression::operation(location, makeType<int>(), OperationKind::subtract,
																															llvm::ArrayRef({
																																	Expression::reference(location, makeType<int>(), "i"),
																																	Expression::constant(location, makeType<int>(), 1)
																															}))
																		})),
							Expression::operation(location, makeType<float>(), OperationKind::multiply,
																		llvm::ArrayRef({
																				Expression::reference(location, makeType<float>(), "x"),
																				Expression::reference(location, makeType<int>(), "i")
																		})))
					));

	auto foo = Class::standardFunction(location, true, "foo", llvm::None, fooMembers, std::move(fooAlgorithm));

	llvm::SmallVector<std::unique_ptr<Member>, 3> mainMembers;
	mainMembers.push_back(Member::build(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output)));

	auto mainAlgorithm = Algorithm::build(
			location,
			Statement::assignmentStatement(
					location,
					Expression::reference(location, makeType<float>(-1), "x"),
					Expression::call(location, makeType<float>(-1),
													 Expression::reference(location, makeType<float>(-1), "foo"),
													 llvm::ArrayRef({
															 Expression::constant(location, makeType<float>(), 2),
															 Expression::constant(location, makeType<int>(), 3)
													 }))));

	auto main = Class::standardFunction(location, true, "main", llvm::None, mainMembers, std::move(mainAlgorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run({ *foo, *main });

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<float, 3> x = { 0, 0, 0 };
	ArrayDescriptor<float, 1> xDesc(x);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc)));
	EXPECT_FLOAT_EQ(xDesc[0], 2);
	EXPECT_FLOAT_EQ(xDesc[1], 4);
	EXPECT_FLOAT_EQ(xDesc[2], 6);
}

TEST(Function, callElementWise)	 // NOLINT
{
	/**
	 * function foo
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *   	 y := -x;
	 * end foo
	 *
	 * function main
	 *   input Integer[3] x;
	 *   output Integer[3] y;
	 *
	 *   algorithm
	 *     y := foo(x);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto fooAlgorithm = Algorithm::build(
			location,
			Statement::assignmentStatement(
					location,
					Expression::reference(location, makeType<int>(), "y"),
					Expression::operation(location, makeType<int>(), OperationKind::subtract,
																Expression::reference(location, makeType<int>(), "x"))));

	auto foo = Class::standardFunction(
			location, true, "foo", llvm::None,
			llvm::ArrayRef({
					Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input)),
					Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output))
			}),
			std::move(fooAlgorithm));

	auto mainAlgorithm = Algorithm::build(
			location,
			Statement::assignmentStatement(
					location,
					Expression::reference(location, makeType<int>(3), "y"),
					Expression::call(location, makeType<int>(3),
													 Expression::reference(location, makeType<int>(), "foo"),
													 Expression::reference(location, makeType<int>(3), "x"))));

	auto main = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({
					Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input)),
					Member::build(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output))
			}),
			std::move(mainAlgorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run({ *main, *foo });

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 1, 0, 0 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<int, 3> y = { 0, 0, 0 };
	ArrayDescriptor<int, 1> yDesc(y);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc)));
	EXPECT_EQ(yDesc[0], -1 * xDesc[0]);
	EXPECT_EQ(yDesc[1], -1 * xDesc[1]);
	EXPECT_EQ(yDesc[2], -1 * xDesc[2]);
}

TEST(Function, callWithMultipleOutputs)	 // NOLINT
{
	/**
	 * function foo
	 *   input Integer x;
	 *   output Integer y;
	 *   output Integer z;
	 *
	 *   algorithm
	 *   	 y := 2 * x;
	 *   	 z = 3 * x;
	 * end foo
	 *
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *   output Integer z;
	 *
	 *   algorithm
	 *     (y, z) := foo(x);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto xRef = Expression::reference(location, makeType<int>(), "x");

	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto yRef = Expression::reference(location, makeType<int>(), "y");

	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto zRef = Expression::reference(location, makeType<int>(), "z");

	auto fooAlgorithm = Algorithm::build(
			location,
			llvm::ArrayRef({
					Statement::assignmentStatement(
							location,
							yRef->clone(),
							Expression::operation(location, makeType<int>(), OperationKind::multiply,
																		llvm::ArrayRef({
																				xRef->clone(),
																				Expression::constant(location, makeType<int>(), 2)
																		}))),
					Statement::assignmentStatement(
							location,
							zRef->clone(),
							Expression::operation(location, makeType<int>(), OperationKind::multiply,
																		llvm::ArrayRef({
																				xRef->clone(),
																				Expression::constant(location, makeType<int>(), 3)
																		})))
			}));

	PackedType packedType({ makeType<int>(), makeType<int>() });

	auto foo = Class::standardFunction(
			location, true, "foo", llvm::None,
			llvm::ArrayRef({ xMember->clone(), yMember->clone(), zMember->clone() }),
			std::move(fooAlgorithm));

	auto mainAlgorithm = Algorithm::build(
			location,
			Statement::assignmentStatement(
					location,
					Expression::tuple(location, Type(PackedType(llvm::ArrayRef({ makeType<int>(), makeType<int>() }))),
														llvm::ArrayRef({ yRef->clone(), zRef->clone() })),
					Expression::call(location, packedType,
													 Expression::reference(location, packedType, "foo"),
													 Expression::reference(location, makeType<int>(), "x"))));

	auto main = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ xMember->clone(), yMember->clone(), zMember->clone() }),
			std::move(mainAlgorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run({ *main, *foo });

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	int x = 1;

	struct
	{
		int y = 0;
		int z = 0;
	} result;

	auto* resultPtr = &result;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(resultPtr), x)));
	EXPECT_EQ(result.y, x * 2);
	EXPECT_EQ(result.z, x * 3);
}