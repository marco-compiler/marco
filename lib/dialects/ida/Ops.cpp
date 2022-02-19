#include "marco/dialects/ida/IDADialect.h"
#include "marco/dialects/ida/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace ::mlir;
using namespace ::mlir::ida;

/*
/// Writes inside a function how to compute the monodimensional offset of a
/// variable needed by IDA, given the variable access and the indexes.
static mlir::Value computeVariableOffset(
    IdaBuilder& builder,
    const model::Variable& variable,
    const model::Expression& expression,
    int64_t varOffset,
    mlir::BlockArgument indexes)
{
  mlir::Location loc = expression.getOp()->getLoc();
  model::VectorAccess vectorAccess = model::AccessToVar::fromExp(expression).getAccess();
  marco::MultiDimInterval dimensions = variable.toMultiDimInterval();

  mlir::Value offset = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(0));

  // For every dimension of the variable
  for (size_t i = 0; i < vectorAccess.size(); i++)
  {
    // Compute the offset of the current dimension.
    model::SingleDimensionAccess acc = vectorAccess[i];
    int64_t accOffset = acc.isDirectAccess() ? acc.getOffset() : (acc.getOffset() + 1);
    mlir::Value accessOffset = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(accOffset));

    if (acc.isOffset())
    {
      // Add the offset that depends on the input indexes.
      mlir::Value indIndex = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(acc.getInductionVar()));
      mlir::Value indValue = builder.create<LoadPointerOp>(loc, indexes, indIndex);
      accessOffset = builder.create<modelica::AddOp>(loc, accessOffset.getType(), accessOffset, indValue);
    }

    // Multiply the previous offset by the width of the current dimension.
    mlir::Value dimension = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(dimensions[i].size()));
    offset = builder.create<modelica::MulOp>(loc, offset.getType(), offset, dimension);

    // Add the current dimension offset.
    offset = builder.create<modelica::AddOp>(loc, offset.getType(), offset, accessOffset);
  }

  // Add the offset from the start of the monodimensional variable array used by IDA.
  mlir::Value varOffsetValue = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(varOffset));
  return builder.create<modelica::AddOp>(loc, offset.getType(), offset, varOffsetValue);
}

/// Writes inside a function how to compute the given expression starting from
/// the given arguments (which are: time, vars, ders, indexes)
static mlir::Value getFunction(
    IdaBuilder& builder,
    model::Model& model,
    const model::Expression& expression,
    llvm::ArrayRef<mlir::BlockArgument> args)
{
  // Induction argument.
  if (expression.isInduction())
  {
    mlir::Location loc = expression.get<model::Induction>().getArgument().getLoc();
    unsigned int argNumber = expression.get<model::Induction>().getArgument().getArgNumber();
    mlir::Value indIndex = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(argNumber));
    mlir::Value indValue = builder.create<LoadPointerOp>(loc, args[3], indIndex);

    // Add one because Modelica is 1-indexed.
    mlir::Value one = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(1));
    return builder.create<modelica::AddOp>(loc, builder.getIntegerType(), indValue, one);
  }

  mlir::Operation* definingOp = expression.getOp();
  mlir::Location loc = definingOp->getLoc();

  // Constant value.
  if (mlir::isa<modelica::ConstantOp>(definingOp))
    return builder.clone(*definingOp)->getResult(0);

  // Variable reference.
  if (expression.isReferenceAccess())
  {
    model::Variable var = model.getVariable(expression.getReferredVectorAccess());

    // Time variable.
    if (var.isTime())
      return args[0];

    // Compute the IDA variable offset, which depends on the variable, the dimension and the access.
    mlir::Value varOffset = computeVariableOffset(builder, var, expression, var.getIdaOffset(), args[3]);

    // Access and return the correct variable value.
    mlir::BlockArgument argArray = var.isDerivative() ? args[2] : args[1];
    return builder.create<LoadPointerOp>(loc, argArray, varOffset);
  }

  // Operation.
  assert(expression.isOperation());

  // Recursively compute and map the value of all the children.
  mlir::BlockAndValueMapping mapping;
  for (size_t i : marco::irange(expression.childrenCount()))
    mapping.map(
        expression.getOp()->getOperand(i),
        getFunction(builder, model, expression.getChild(i), args));

  // Add to the residual function and return the correct mapped operation.
  return builder.clone(*definingOp, mapping)->getResult(0);
}

/// Writes inside a function how to compute the derivative of the given
/// expression starting from the given arguments (which are: time, vars, ders,
/// indexes, derVar, alpha)
static mlir::Value getDerFunction(
    IdaBuilder& builder,
    model::Model& model,
    const model::Expression& expression,
    llvm::ArrayRef<mlir::BlockArgument> args)
{
  // Induction argument.
  if (expression.isInduction())
  {
    mlir::Location loc = expression.get<model::Induction>().getArgument().getLoc();
    return builder.create<modelica::ConstantOp>(loc, builder.getRealAttribute(0.0));
  }

  mlir::Operation* definingOp = expression.getOp();
  mlir::Location loc = definingOp->getLoc();

  // Constant value.
  if (mlir::isa<modelica::ConstantOp>(definingOp))
    return builder.create<modelica::ConstantOp>(loc, builder.getRealAttribute(0.0));

  // Variable reference.
  if (expression.isReferenceAccess())
  {
    model::Variable var = model.getVariable(expression.getReferredVectorAccess());

    // Time variable.
    if (var.isTime())
      return builder.create<modelica::ConstantOp>(loc, builder.getRealAttribute(0.0));

    // Compute the IDA variable offset, which depends on the variable, the dimension and the access.
    mlir::Value varOffset = computeVariableOffset(builder, var, expression, var.getIdaOffset(), args[3]);

    // Check if the variable with respect to which we are currently derivating
    // is also the variable we are derivating.
    mlir::Value condition = builder.create<modelica::EqOp>(
        loc, builder.getBooleanType(), varOffset, args[5]);
    condition = builder.create<mlir::UnrealizedConversionCastOp>(loc, builder.getI1Type(), condition).getResult(0);

    // If yes, return alpha (if it is a derivative) or one (if it is a simple variable).
    mlir::Value thenValue = args[4];
    if (!var.isDerivative())
      thenValue = builder.create<modelica::ConstantOp>(loc, builder.getRealAttribute(1.0));

    // If no, return zero.
    mlir::Value elseValue = builder.create<modelica::ConstantOp>(loc, builder.getRealAttribute(0.0));
    return builder.create<mlir::SelectOp>(loc, builder.getRealType(), condition, thenValue, elseValue);
  }

  // Operation.
  assert(expression.isOperation());
  assert(definingOp->hasTrait<modelica::DerivativeInterface::Trait>());

  // Recursively compute and map the value of all the children.
  mlir::BlockAndValueMapping mapping;
  for (size_t i : marco::irange(expression.childrenCount()))
    mapping.map(
        expression.getOp()->getOperand(i),
        getFunction(builder, model, expression.getChild(i), args));

  // Clone the operation with the new operands.
  mlir::Operation* clonedOp = builder.clone(*definingOp, mapping);
  builder.setInsertionPoint(clonedOp);

  // Recursively compute and map the derivatives of all the children.
  mlir::BlockAndValueMapping derMapping;
  for (size_t i : marco::irange(expression.childrenCount()))
    derMapping.map(
        clonedOp->getOperand(i),
        getDerFunction(builder, model, expression.getChild(i), args));

  // Compute and return the derived operation.
  mlir::Value derivedOp = mlir::cast<modelica::DerivativeInterface>(clonedOp).derive(builder, derMapping).front();
  builder.setInsertionPointAfterValue(derivedOp);
  clonedOp->erase();
  return derivedOp;
}

static void foldConstants(mlir::OpBuilder& builder, mlir::Block& block)
{
  llvm::SmallVector<mlir::Operation*, 3> operations;

  for (mlir::Operation& operation : block.getOperations())
    operations.push_back(&operation);

  // If an operation has only constants as operands, we can substitute it with
  // the corresponding constant value and erase the old operation.
  for (mlir::Operation* operation : operations)
    if (operation->hasTrait<modelica::FoldableOpInterface::Trait>())
      mlir::cast<modelica::FoldableOpInterface>(operation).foldConstants(builder);
}

static void cleanOperation(mlir::Block& block)
{
  llvm::SmallVector<mlir::Operation*, 3> operations;

  for (mlir::Operation& operation : block.getOperations())
    if (!mlir::isa<FunctionTerminatorOp>(operation))
      operations.push_back(&operation);

  assert(llvm::all_of(operations,
      [](mlir::Operation* op) { return op->getNumResults() == 1; }));

  // If an operation has no uses, erase it.
  for (mlir::Operation* operation : llvm::reverse(operations))
    if (operation->use_empty())
      operation->erase();
}
*/

/*
void ResidualFunctionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, model::Model& model, model::Equation& equation)
{
  IdaBuilder idaBuilder(builder.getContext());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), idaBuilder.getStringAttr(name));

  // real residual_function(real time, real* variables, real* derivatives, int* indexes)
  llvm::SmallVector<mlir::Type, 4> argTypes = idaBuilder.getResidualArgTypes();
  mlir::Type returnType = { idaBuilder.getRealType() };
  state.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(idaBuilder.getFunctionType(argTypes, returnType)));

  mlir::Region* entryRegion = state.addRegion();
  mlir::Block& entryBlock = entryRegion->emplaceBlock();
  entryBlock.addArguments(argTypes);

  // Fill the only block of the function with how to compute the Residual of the given Equation.
  idaBuilder.setInsertionPointToStart(&entryBlock);

  mlir::Value lhsResidual = getFunction(idaBuilder, model, equation.lhs(), entryBlock.getArguments());
  mlir::Value rhsResidual = getFunction(idaBuilder, model, equation.rhs(), entryBlock.getArguments());

  mlir::Value returnValue = idaBuilder.create<modelica::SubOp>(equation.getOp().getLoc(), idaBuilder.getRealType(), rhsResidual, lhsResidual);
  idaBuilder.create<FunctionTerminatorOp>(equation.getOp().getLoc(), returnValue);

  // Fold the constants and clean the unused operations.
  foldConstants(builder, entryBlock);
  cleanOperation(entryBlock);
}
*/

/*
void JacobianFunctionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, model::Model& model, model::Equation& equation)
{
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));

  IdaBuilder idaBuilder(builder.getContext());

  // real jacobian_function(real time, real* variables, real* derivatives, int* indexes, real alpha, int der_var)
  llvm::SmallVector<mlir::Type, 6> argTypes = idaBuilder.getJacobianArgTypes();
  mlir::Type returnType = { idaBuilder.getRealType() };
  state.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(idaBuilder.getFunctionType(argTypes, returnType)));

  mlir::Region* entryRegion = state.addRegion();
  mlir::Block& entryBlock = entryRegion->emplaceBlock();
  entryBlock.addArguments(argTypes);

  // Fill the only block of the function with how to compute the Residual of the given Equation.
  idaBuilder.setInsertionPointToStart(&entryBlock);

  mlir::Value lhsJacobian = getDerFunction(idaBuilder, model, equation.lhs(), entryBlock.getArguments());
  mlir::Value rhsJacobian = getDerFunction(idaBuilder, model, equation.rhs(), entryBlock.getArguments());

  mlir::Value returnValue = idaBuilder.create<modelica::SubOp>(equation.getOp().getLoc(), idaBuilder.getRealType(), rhsJacobian, lhsJacobian);
  idaBuilder.create<FunctionTerminatorOp>(equation.getOp().getLoc(), returnValue);

  // Fold the constants and clean the unused operations.
  foldConstants(builder, entryBlock);
  cleanOperation(entryBlock);
}
*/

#define GET_OP_CLASSES
#include "marco/dialects/ida/IDA.cpp.inc"

namespace mlir::ida
{
  //===----------------------------------------------------------------------===//
  // InitOp
  //===----------------------------------------------------------------------===//

  void InitOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), instance(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // CreateOp
  //===----------------------------------------------------------------------===//

  void CreateOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
  }

  //===----------------------------------------------------------------------===//
  // FreeOp
  //===----------------------------------------------------------------------===//

  void FreeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Free::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // StepOp
  //===----------------------------------------------------------------------===//

  void StepOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), instance(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // SetTimesOp
  //===----------------------------------------------------------------------===//

  void SetTimesOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // SetToleranceOp
  //===----------------------------------------------------------------------===//

  void SetToleranceOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // GetCurrentTimeOp
  //===----------------------------------------------------------------------===//

  void GetCurrentTimeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // AddEquationOp
  //===----------------------------------------------------------------------===//

  void AddEquationOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), instance(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get(), equationRanges(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // AddResidualOp
  //===----------------------------------------------------------------------===//

  void AddResidualOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // AddJacobianOp
  //===----------------------------------------------------------------------===//

  void AddJacobianOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // AddVariableOp
  //===----------------------------------------------------------------------===//

  void AddVariableOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), instance(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get(), arrayDimensions(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // AddVarAccessOp
  //===----------------------------------------------------------------------===//

  void AddVarAccessOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // GetVariablesListOp
  //===----------------------------------------------------------------------===//

  void GetVariablesListOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // GetDerivativesListOp
  //===----------------------------------------------------------------------===//

  void GetDerivativesListOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // GetVariableOp
  //===----------------------------------------------------------------------===//

  void GetVariableOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), instance(), mlir::SideEffects::DefaultResource::get());
    // TODO allocation trait on result
  }

  //===----------------------------------------------------------------------===//
  // GetTimeOp
  //===----------------------------------------------------------------------===//

  void GetTimeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // PrintStatisticsOp
  //===----------------------------------------------------------------------===//

  void PrintStatisticsOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), instance(), mlir::SideEffects::DefaultResource::get());
  }
}
