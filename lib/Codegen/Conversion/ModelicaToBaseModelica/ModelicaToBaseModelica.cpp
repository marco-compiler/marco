#include "marco/Codegen/Conversion/ModelicaToBaseModelica/ModelicaToBaseModelica.h"
#include "marco/Dialect/BaseModelica/IR/Dialect.h"
#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "marco/Dialect/Modelica/Transforms/Evaluation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include <functional>

namespace mlir {
#define GEN_PASS_DEF_MODELICATOBASEMODELICACONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

namespace mo = ::mlir::modelica;
namespace bmo = ::mlir::bmodelica;

namespace {
/// The two sides of an equation, as values of Base Modelica.
struct EquationSides {
  llvm::SmallVector<mlir::Value, 1> lhs;
  llvm::SmallVector<mlir::Value, 1> rhs;
};

/// Function emitting, at the insertion point of the builder, the operations
/// computing the sides of an equation.
using SidesEmitter = std::function<mlir::FailureOr<EquationSides>()>;

/// Arithmetic progression of integers, with the last value included.
struct Progression {
  int64_t from;
  int64_t to;
  int64_t step;
};

class Converter {
public:
  explicit Converter(mlir::ModuleOp moduleOp);

  mlir::LogicalResult run();

private:
  //===-------------------------------------------------------------------===//
  // Types and attributes
  //===-------------------------------------------------------------------===//

  /// Convert a type of the Modelica dialect. Returns a null type, after
  /// emitting an error at the given location, if the type has no
  /// counterpart in Base Modelica.
  mlir::Type convertType(mlir::Type type, mlir::Location loc);

  /// Convert a literal of the Modelica dialect. Returns a null attribute,
  /// after emitting an error, in case of failure.
  mlir::TypedAttr convertAttribute(mlir::Attribute attr, mlir::Location loc);

  //===-------------------------------------------------------------------===//
  // Classes
  //===-------------------------------------------------------------------===//

  mlir::LogicalResult convertModel(mo::ModelOp modelOp);

  mlir::LogicalResult convertFunction(mo::FunctionOp functionOp);

  /// Collect the variables of a class, so that they can be referenced and
  /// evaluated.
  void collectVariables(mlir::Block &body);

  mlir::LogicalResult declareVariable(mo::VariableOp variableOp,
                                      bool inFunction);

  /// Create the operation computing the value of a modification of a
  /// variable, that is its binding or the value of one of its attributes.
  /// The values computed in the modification region of the variable are
  /// available to the modification.
  template <typename Op, typename... Args>
  mlir::LogicalResult createModificationOp(mo::VariableOp variableOp,
                                           mo::ModifyOp modifyOp,
                                           mlir::Value value, Args &&...args);

  mlir::LogicalResult convertStartAttribute(mo::VariableOp variableOp);

  mlir::LogicalResult convertExternal(mo::ExternalOp externalOp,
                                      mo::FunctionOp functionOp);

  //===-------------------------------------------------------------------===//
  // Equations
  //===-------------------------------------------------------------------===//

  mlir::LogicalResult convertEquations(mlir::Block &block);

  mlir::LogicalResult convertForEquation(mo::ForEquationOp op);

  /// Collect the emitters of the equations within a block, which may only
  /// contain equations and if-equations. If-equations whose condition can be
  /// evaluated are replaced by their selected branch, while the equations
  /// of the branches of the other ones are merged, so that each side of the
  /// resulting equations selects the side of the equation of the branch
  /// being active.
  mlir::LogicalResult
  collectEquations(mlir::Block &block,
                   llvm::SmallVectorImpl<SidesEmitter> &emitters);

  mlir::LogicalResult
  collectEquations(mlir::Operation *op,
                   llvm::SmallVectorImpl<SidesEmitter> &emitters);

  /// Create an equation whose sides are computed by the given emitter.
  mlir::LogicalResult emitEquation(const SidesEmitter &emitter,
                                   mlir::Location loc);

  /// Get whether the iteration range of a for-equation or the condition of
  /// an if-equation nested within the for-equation depend on its induction
  /// variable, in which case the for-equation has to be unrolled.
  static bool mustUnroll(mo::ForEquationOp op);

  //===-------------------------------------------------------------------===//
  // Statements
  //===-------------------------------------------------------------------===//

  mlir::LogicalResult convertStatements(mlir::Block &block);

  mlir::LogicalResult convertStatement(mlir::Operation *op);

  mlir::LogicalResult convertAssignment(mo::AssignmentOp op);

  /// Assign a value to the component reference computed by the destination
  /// of an assignment.
  mlir::LogicalResult assign(mlir::Value destination, mlir::Value value,
                             mlir::Location loc);

  mlir::LogicalResult convertForStatement(mo::ForOp op);

  mlir::LogicalResult convertAssert(mo::AssertOp op);

  //===-------------------------------------------------------------------===//
  // Expressions
  //===-------------------------------------------------------------------===//

  /// Convert the operations of a block, except its terminator and the
  /// operations to be skipped.
  mlir::LogicalResult
  convertExpressions(mlir::Block &block,
                     const llvm::DenseSet<mlir::Operation *> &skipped = {});

  mlir::LogicalResult convertExpression(mlir::Operation *op);

  /// Convert an operation into the Base Modelica operation with the same
  /// semantics, operands and results.
  template <typename Op>
  mlir::LogicalResult convertOneToOne(mlir::Operation *op);

  /// Get the converted value of a Modelica value.
  mlir::Value lookup(mlir::Value value);

  /// Convert the values yielded by a region, after converting its operations.
  mlir::FailureOr<llvm::SmallVector<mlir::Value>>
  convertYieldedValues(mlir::Region &region);

  mlir::Value createIndex(mlir::Location loc, int64_t value);

  /// Cast a value to the given type, if needed.
  mlir::Value castTo(mlir::Value value, mlir::Type type, mlir::Location loc);

  /// Convert a scalar index to a value of type 'index'.
  mlir::Value convertScalarIndex(mlir::Value index, mlir::Location loc);

  /// Convert a vector of indices to a range.
  mlir::Value convertVectorIndex(mlir::Value index, mlir::Location loc);

  /// Convert a vector to the range iterated by a reduction or a loop,
  /// keeping the values unchanged.
  mlir::Value convertIterable(mlir::Value vector, mlir::Location loc);

  mlir::LogicalResult convertSubscript(mo::SubscriptOp op);

  mlir::LogicalResult convertIfExpr(mo::IfExprOp op);

  mlir::LogicalResult convertArray(mo::ArrayOp op);

  mlir::LogicalResult convertArrayFor(mo::ArrayForOp op);

  mlir::LogicalResult convertReduction(mo::ReductionOp op);

  mlir::LogicalResult convertCall(mo::CallOp op);

  /// Compute the arguments of a call to the given function, ordering the
  /// named arguments as the inputs of the function.
  mlir::FailureOr<llvm::SmallVector<mlir::Value>>
  orderCallArguments(mlir::Operation *op, mlir::SymbolRefAttr callee,
                     mlir::ValueRange args,
                     std::optional<mlir::ArrayAttr> names);

  /// Get the result types of the function with the given name.
  mlir::FailureOr<llvm::SmallVector<mlir::Type>>
  getFunctionResultTypes(mlir::SymbolRefAttr callee, mlir::Location loc);

  /// Get the elements of a value, which is either a scalar or a tensor with
  /// static shape, in row-major order.
  mlir::LogicalResult getElements(mlir::Value value, mlir::Type elementType,
                                  mlir::Location loc,
                                  llvm::SmallVectorImpl<mlir::Value> &elements);

  //===-------------------------------------------------------------------===//
  // Evaluation
  //===-------------------------------------------------------------------===//

  /// Evaluate a value at compile time. Returns a null attribute if the value
  /// can't be evaluated.
  mlir::Attribute evaluate(mlir::Value value);

  /// Evaluate the values of a vector at compile time.
  std::optional<llvm::SmallVector<mlir::Attribute>>
  evaluateVector(mlir::Value value);

  /// Evaluate the value yielded by a region with a single block.
  mlir::Attribute evaluateYield(mlir::Region &region);

  std::optional<mlir::Attribute> evaluationHook(mlir::Value value);

  static std::optional<Progression>
  getProgression(llvm::ArrayRef<mlir::Attribute> values);

private:
  mlir::ModuleOp moduleOp;
  mlir::OpBuilder builder;

  /// The converted values.
  mlir::IRMapping mapping;

  /// The values of the iterators of the for-equations and of the array
  /// constructors that are being unrolled.
  llvm::DenseMap<mlir::Value, mlir::Attribute> iteratorValues;

  /// The variables of the class being converted.
  llvm::StringMap<mo::VariableOp> variables;

  /// The values of the parameters and constants of the class being
  /// converted, and the ones being computed.
  llvm::StringMap<mlir::Attribute> parameterValues;
  llvm::StringSet<> evaluatingParameters;

  /// The functions of the module, by name.
  llvm::StringMap<mo::FunctionOp> functions;
};
} // namespace

Converter::Converter(mlir::ModuleOp moduleOp)
    : moduleOp(moduleOp), builder(moduleOp.getContext()) {}

mlir::LogicalResult Converter::run() {
  llvm::SmallVector<mo::ModelOp> modelOps;

  for (mlir::Operation &op : moduleOp.getOps()) {
    if (auto functionOp = mlir::dyn_cast<mo::FunctionOp>(op)) {
      functions[functionOp.getSymName()] = functionOp;
    } else if (auto modelOp = mlir::dyn_cast<mo::ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  }

  bool failed = false;
  llvm::SmallVector<mlir::Operation *> convertedOps;

  for (mo::ModelOp modelOp : modelOps) {
    builder.setInsertionPoint(modelOp);
    failed |= mlir::failed(convertModel(modelOp));
    convertedOps.push_back(builder.getInsertionPoint()->getPrevNode());
  }

  // Only the functions that are called by the converted operations are
  // converted, so that the ones used only by the dropped operations, such
  // as the assertions within equations, are not required to be supported.
  llvm::StringSet<> convertedFunctions;

  for (size_t i = 0; i < convertedOps.size() && !failed; ++i) {
    llvm::SmallVector<mlir::StringRef> callees;

    convertedOps[i]->walk([&](bmo::CallOp callOp) {
      callees.push_back(callOp.getCallee().getRootReference().getValue());
    });

    for (llvm::StringRef callee : callees) {
      mo::FunctionOp functionOp = functions.lookup(callee);

      if (!functionOp || !convertedFunctions.insert(callee).second) {
        continue;
      }

      builder.setInsertionPoint(functionOp);
      failed |= mlir::failed(convertFunction(functionOp));
      convertedOps.push_back(builder.getInsertionPoint()->getPrevNode());
    }
  }

  if (failed) {
    return mlir::failure();
  }

  // Remove the results of the conversion that turned out to be unused, such
  // as the Integer values whose computation has been recreated on indices.
  // The users are visited before the operations defining their operands.
  for (mlir::Operation *convertedOp : convertedOps) {
    convertedOp->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
        [&](mlir::Operation *op) {
          if (op != convertedOp && mlir::isOpTriviallyDead(op)) {
            op->erase();
          }
        });
  }

  // Erase the converted operations, the functions that are not called, and
  // the remaining class-level operations, such as the classes from which the
  // models have been instantiated, which have no meaning in Base Modelica.
  for (mlir::Operation &op : llvm::make_early_inc_range(moduleOp.getOps())) {
    if (mlir::isa<mo::ModelicaDialect>(op.getDialect())) {
      op.erase();
    }
  }

  return mlir::success();
}

//===---------------------------------------------------------------------===//
// Types and attributes
//===---------------------------------------------------------------------===//

mlir::Type Converter::convertType(mlir::Type type, mlir::Location loc) {
  mlir::MLIRContext *context = type.getContext();

  if (mlir::isa<mo::BooleanType>(type)) {
    return bmo::BooleanType::get(context);
  }

  // Enumeration values are represented by their ordinal values.
  if (mlir::isa<mo::IntegerType, mo::EnumerationType>(type)) {
    return bmo::IntegerType::get(context);
  }

  if (mlir::isa<mo::RealType>(type)) {
    return bmo::RealType::get(context);
  }

  if (mlir::isa<mlir::IndexType>(type)) {
    return type;
  }

  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    mlir::Type elementType = convertType(tensorType.getElementType(), loc);

    if (!elementType) {
      return nullptr;
    }

    return tensorType.clone(elementType);
  }

  if (mlir::isa<mo::StringType>(type)) {
    mlir::emitError(loc) << "String values are not supported";
  } else if (mlir::isa<mo::ObjectType>(type)) {
    mlir::emitError(loc) << "records are not supported";
  } else {
    mlir::emitError(loc) << "type " << type << " is not supported";
  }

  return nullptr;
}

mlir::TypedAttr Converter::convertAttribute(mlir::Attribute attr,
                                            mlir::Location loc) {
  mlir::MLIRContext *context = attr.getContext();

  if (auto boolAttr = mlir::dyn_cast<mo::BooleanAttr>(attr)) {
    return bmo::BooleanAttr::get(context, boolAttr.getValue());
  }

  if (auto intAttr = mlir::dyn_cast<mo::IntegerAttr>(attr)) {
    return bmo::IntegerAttr::get(context, intAttr.getValue());
  }

  if (auto realAttr = mlir::dyn_cast<mo::RealAttr>(attr)) {
    return bmo::RealAttr::get(context, realAttr.getValue().convertToDouble());
  }

  if (auto enumAttr = mlir::dyn_cast<mo::EnumLiteralAttr>(attr)) {
    return bmo::IntegerAttr::get(context, enumAttr.getOrdinal());
  }

  if (auto indexAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
      indexAttr && mlir::isa<mlir::IndexType>(indexAttr.getType())) {
    return indexAttr;
  }

  if (auto arrayAttr = mlir::dyn_cast<mo::ArrayValueAttr>(attr)) {
    auto tensorType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
        convertType(arrayAttr.getType(), loc));

    if (!tensorType) {
      return nullptr;
    }

    mlir::Type elementType = tensorType.getElementType();

    if (mlir::isa<bmo::BooleanType>(elementType)) {
      llvm::SmallVector<bool> values;

      for (mlir::Attribute element : arrayAttr.getElements()) {
        values.push_back(mlir::cast<mo::BooleanAttr>(element).getValue());
      }

      return bmo::DenseBooleanElementsAttr::get(tensorType, values);
    }

    if (mlir::isa<bmo::IntegerType>(elementType)) {
      llvm::SmallVector<int64_t> values;

      for (mlir::Attribute element : arrayAttr.getElements()) {
        if (auto enumAttr = mlir::dyn_cast<mo::EnumLiteralAttr>(element)) {
          values.push_back(enumAttr.getOrdinal());
        } else {
          values.push_back(mlir::cast<mo::IntegerAttr>(element).getValue());
        }
      }

      return bmo::DenseIntegerElementsAttr::get(tensorType, values);
    }

    llvm::SmallVector<double> values;

    for (mlir::Attribute element : arrayAttr.getElements()) {
      if (auto intAttr = mlir::dyn_cast<mo::IntegerAttr>(element)) {
        values.push_back(static_cast<double>(intAttr.getValue()));
      } else {
        values.push_back(
            mlir::cast<mo::RealAttr>(element).getValue().convertToDouble());
      }
    }

    return bmo::DenseRealElementsAttr::get(tensorType, values);
  }

  if (mlir::isa<mo::StringValueAttr>(attr)) {
    mlir::emitError(loc) << "String values are not supported";
  } else {
    mlir::emitError(loc) << "literal " << attr << " is not supported";
  }

  return nullptr;
}

//===---------------------------------------------------------------------===//
// Classes
//===---------------------------------------------------------------------===//

void Converter::collectVariables(mlir::Block &body) {
  variables.clear();
  parameterValues.clear();
  evaluatingParameters.clear();

  for (auto variableOp : body.getOps<mo::VariableOp>()) {
    variables[variableOp.getSymName()] = variableOp;
  }
}

mlir::LogicalResult Converter::convertModel(mo::ModelOp modelOp) {
  for (mlir::Operation &op : modelOp.getBody()->getOperations()) {
    if (mlir::isa<mo::InstanceOp>(op)) {
      return op.emitError()
             << "the model must be flattened before being converted to "
                "Base Modelica";
    }
  }

  collectVariables(*modelOp.getBody());

  auto bmoModelOp =
      builder.create<bmo::ModelOp>(modelOp.getLoc(), modelOp.getSymName());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(
      builder.createBlock(&bmoModelOp.getBodyRegion()));

  bool failed = false;
  auto variableOps =
      llvm::to_vector(modelOp.getBody()->getOps<mo::VariableOp>());

  // The String variables of models, such as the names of files, are not
  // supported by Base Modelica. They are dropped, and their usages are
  // reported as errors.
  llvm::erase_if(variableOps, [](mo::VariableOp variableOp) {
    return mlir::isa<mo::StringType>(
        mlir::getElementTypeOrSelf(variableOp.getType()));
  });

  for (mo::VariableOp variableOp : variableOps) {
    failed |= mlir::failed(declareVariable(variableOp, false));
  }

  if (failed) {
    return mlir::failure();
  }

  // Binding equations and start values.
  for (mo::VariableOp variableOp : variableOps) {
    if (mlir::Value binding = variableOp.getBinding()) {
      failed |= mlir::failed(createModificationOp<bmo::BindingEquationOp>(
          variableOp, nullptr, binding, variableOp.getSymName()));
    }

    failed |= mlir::failed(convertStartAttribute(variableOp));
  }

  // Equations and algorithms.
  for (mlir::Operation &op : modelOp.getBody()->getOperations()) {
    if (auto sectionOp = mlir::dyn_cast<mo::EquationSectionOp>(op)) {
      mlir::OpBuilder::InsertionGuard sectionGuard(builder);

      if (sectionOp.getIsInitial()) {
        auto initialOp = builder.create<bmo::InitialOp>(sectionOp.getLoc());
        builder.setInsertionPointToStart(
            builder.createBlock(&initialOp.getBodyRegion()));
      } else {
        auto dynamicOp = builder.create<bmo::DynamicOp>(sectionOp.getLoc());
        builder.setInsertionPointToStart(
            builder.createBlock(&dynamicOp.getBodyRegion()));
      }

      failed |=
          mlir::failed(convertEquations(sectionOp.getBodyRegion().front()));
      continue;
    }

    if (auto sectionOp = mlir::dyn_cast<mo::AlgorithmSectionOp>(op)) {
      mlir::OpBuilder::InsertionGuard sectionGuard(builder);

      if (sectionOp.getIsInitial()) {
        auto initialOp = builder.create<bmo::InitialOp>(sectionOp.getLoc());
        builder.setInsertionPointToStart(
            builder.createBlock(&initialOp.getBodyRegion()));
      } else {
        auto dynamicOp = builder.create<bmo::DynamicOp>(sectionOp.getLoc());
        builder.setInsertionPointToStart(
            builder.createBlock(&dynamicOp.getBodyRegion()));
      }

      auto algorithmOp = builder.create<bmo::AlgorithmOp>(sectionOp.getLoc());
      builder.setInsertionPointToStart(
          builder.createBlock(&algorithmOp.getBodyRegion()));

      failed |=
          mlir::failed(convertStatements(sectionOp.getBodyRegion().front()));
    }
  }

  return mlir::success(!failed);
}

mlir::LogicalResult Converter::convertFunction(mo::FunctionOp functionOp) {
  collectVariables(*functionOp.getBody());

  auto bmoFunctionOp = builder.create<bmo::FunctionOp>(functionOp.getLoc(),
                                                       functionOp.getSymName());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(
      builder.createBlock(&bmoFunctionOp.getBodyRegion()));

  bool failed = false;
  auto variableOps =
      llvm::to_vector(functionOp.getBody()->getOps<mo::VariableOp>());

  for (mo::VariableOp variableOp : variableOps) {
    failed |= mlir::failed(declareVariable(variableOp, true));
  }

  if (failed) {
    return mlir::failure();
  }

  // Default values.
  for (mo::VariableOp variableOp : variableOps) {
    if (mlir::Value binding = variableOp.getBinding()) {
      failed |= mlir::failed(createModificationOp<bmo::DefaultOp>(
          variableOp, nullptr, binding, variableOp.getSymName()));
    }
  }

  for (mlir::Operation &op : functionOp.getBody()->getOperations()) {
    if (auto sectionOp = mlir::dyn_cast<mo::AlgorithmSectionOp>(op)) {
      mlir::OpBuilder::InsertionGuard sectionGuard(builder);
      auto algorithmOp = builder.create<bmo::AlgorithmOp>(sectionOp.getLoc());
      builder.setInsertionPointToStart(
          builder.createBlock(&algorithmOp.getBodyRegion()));

      failed |=
          mlir::failed(convertStatements(sectionOp.getBodyRegion().front()));
    } else if (auto externalOp = mlir::dyn_cast<mo::ExternalOp>(op)) {
      failed |= mlir::failed(convertExternal(externalOp, functionOp));
    }
  }

  return mlir::success(!failed);
}

mlir::LogicalResult Converter::declareVariable(mo::VariableOp variableOp,
                                               bool inFunction) {
  mlir::Location loc = variableOp.getLoc();
  mlir::Type type = convertType(variableOp.getType(), loc);

  if (!type) {
    return mlir::failure();
  }

  llvm::SmallVector<int64_t> shape;
  mlir::Type elementType = type;

  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    llvm::append_range(shape, tensorType.getShape());
    elementType = tensorType.getElementType();
  }

  auto variability = bmo::VariabilityProperty::none;

  if (auto value = variableOp.getVariability()) {
    switch (*value) {
    case mo::Variability::discrete:
      variability = bmo::VariabilityProperty::discrete;
      break;
    case mo::Variability::parameter:
      variability = bmo::VariabilityProperty::parameter;
      break;
    case mo::Variability::constant:
      variability = bmo::VariabilityProperty::constant;
      break;
    }
  }

  auto causality = bmo::IOProperty::none;

  if (auto value = variableOp.getCausality()) {
    causality = *value == mo::Causality::input ? bmo::IOProperty::input
                                               : bmo::IOProperty::output;
  }

  auto variableType =
      bmo::VariableType::get(shape, elementType, variability, causality);

  // The dynamic dimensions of the variables of functions are either computed
  // by the dimensions region, or determined by the value of the variable.
  llvm::SmallVector<bmo::DimensionConstraint> constraints;
  mlir::Region &dimensionsRegion = variableOp.getDimensionsRegion();
  bool fixedDimensions = !dimensionsRegion.empty();

  for (int64_t size : shape) {
    if (size == mlir::ShapedType::kDynamic) {
      constraints.push_back(fixedDimensions ? bmo::DimensionConstraint::Fixed
                                            : bmo::DimensionConstraint::Free);
    }
  }

  if (!inFunction && !constraints.empty()) {
    return variableOp.emitError()
           << "the dimensions of the variables of models must be known at "
              "compile time";
  }

  auto bmoVariableOp = builder.create<bmo::VariableOp>(
      loc, variableOp.getSymName(), variableType, constraints);

  if (fixedDimensions) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(
        builder.createBlock(&bmoVariableOp.getConstraintsRegion()));

    auto sizes = convertYieldedValues(dimensionsRegion);

    if (mlir::failed(sizes)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> indices;

    for (mlir::Value size : *sizes) {
      indices.push_back(castTo(size, builder.getIndexType(), loc));
    }

    builder.create<bmo::YieldOp>(loc, indices);
  }

  return mlir::success();
}

template <typename Op, typename... Args>
mlir::LogicalResult Converter::createModificationOp(mo::VariableOp variableOp,
                                                    mo::ModifyOp modifyOp,
                                                    mlir::Value value,
                                                    Args &&...args) {
  mlir::Location loc = value.getLoc();
  auto op = builder.create<Op>(loc, std::forward<Args>(args)...);

  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Block *body = builder.createBlock(&op.getBodyRegion());
  builder.setInsertionPointToStart(body);

  // Within the flat models, the modification regions have no arguments.
  mlir::Block &modificationBlock = variableOp.getModificationRegion().front();

  if (modificationBlock.getNumArguments() != 0) {
    return variableOp.emitError()
           << "the model must be flattened before being converted to "
              "Base Modelica";
  }

  llvm::DenseSet<mlir::Operation *> skipped;

  for (mlir::Operation &nestedOp : modificationBlock) {
    if (mlir::isa<mo::ModifyOp, mo::BindOp>(nestedOp)) {
      skipped.insert(&nestedOp);
    }
  }

  if (mlir::failed(convertExpressions(modificationBlock, skipped))) {
    return mlir::failure();
  }

  if (modifyOp) {
    mlir::Block &modifyBlock = modifyOp.getModificationRegion().front();
    llvm::DenseSet<mlir::Operation *> modifySkipped;

    for (mlir::Operation &nestedOp : modifyBlock) {
      if (mlir::isa<mo::ModifyOp, mo::BindOp>(nestedOp)) {
        modifySkipped.insert(&nestedOp);
      }
    }

    if (mlir::failed(convertExpressions(modifyBlock, modifySkipped))) {
      return mlir::failure();
    }
  }

  mlir::Value result = lookup(value);

  if (!result) {
    return mlir::failure();
  }

  builder.create<bmo::YieldOp>(loc, result);

  // Remove the operations computing the values of other modifications.
  for (mlir::Operation &nestedOp :
       llvm::make_early_inc_range(llvm::reverse(*body))) {
    if (mlir::isOpTriviallyDead(&nestedOp)) {
      nestedOp.erase();
    }
  }

  return mlir::success();
}

static mlir::Value getModificationValue(mo::ModifyOp modifyOp) {
  if (!modifyOp || modifyOp.getModificationRegion().empty()) {
    return nullptr;
  }

  for (auto bindOp :
       modifyOp.getModificationRegion().front().getOps<mo::BindOp>()) {
    return bindOp.getValue();
  }

  return nullptr;
}

mlir::LogicalResult
Converter::convertStartAttribute(mo::VariableOp variableOp) {
  mo::ModifyOp startOp = variableOp.getAttributeModification("start");
  mlir::Value start = getModificationValue(startOp);

  if (!start) {
    return mlir::success();
  }

  // The 'fixed' attribute is true by default only for parameters and
  // constants, whose start value is relevant only if they
  // have no binding.
  bool fixed = variableOp.getVariability() &&
               *variableOp.getVariability() != mo::Variability::discrete;

  if (fixed && variableOp.getBinding()) {
    return mlir::success();
  }

  if (mlir::Value fixedValue =
          getModificationValue(variableOp.getAttributeModification("fixed"))) {
    auto fixedAttr =
        mlir::dyn_cast_or_null<mo::BooleanAttr>(evaluate(fixedValue));

    if (!fixedAttr) {
      return variableOp.emitError()
             << "the 'fixed' attribute must be known at compile time";
    }

    fixed = fixedAttr.getValue();
  }

  bool each = mlir::isa<mlir::RankedTensorType>(variableOp.getType()) &&
              !mlir::isa<mlir::RankedTensorType>(start.getType());

  return createModificationOp<bmo::StartOp>(
      variableOp, startOp, start,
      mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr()), fixed, each);
}

mlir::LogicalResult Converter::convertExternal(mo::ExternalOp externalOp,
                                               mo::FunctionOp functionOp) {
  mlir::Location loc = externalOp.getLoc();
  llvm::StringRef language = externalOp.getLanguage();

  if (language != "C" && language != "FORTRAN 77") {
    return externalOp.emitError() << "external functions with language '"
                                  << language << "' are not supported";
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  auto algorithmOp = builder.create<bmo::AlgorithmOp>(loc);
  builder.setInsertionPointToStart(
      builder.createBlock(&algorithmOp.getBodyRegion()));

  llvm::SmallVector<mlir::Value> args;
  llvm::SmallVector<mlir::Type> resultTypes;
  mlir::Value destination;

  // Arguments that may be modified by the external function, which are
  // passed by means of temporary memory.
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> temporaries;

  if (externalOp.getBodyRegion().empty()) {
    // Default mapping of the external function: the inputs are the
    // arguments, and the only output is the result.
    llvm::SmallVector<mo::VariableOp> outputs;

    for (auto variableOp : functionOp.getBody()->getOps<mo::VariableOp>()) {
      auto causality = variableOp.getCausality();
      mlir::Type type = convertType(variableOp.getType(), variableOp.getLoc());

      if (!type) {
        return mlir::failure();
      }

      if (causality == mo::Causality::input) {
        args.push_back(builder.create<bmo::VariableGetOp>(
            loc, type, variableOp.getSymName()));
      } else if (causality == mo::Causality::output) {
        outputs.push_back(variableOp);
        resultTypes.push_back(type);
      }
    }

    if (outputs.size() > 1) {
      return externalOp.emitError()
             << "external functions with multiple outputs require an "
                "explicit external call";
    }

    auto callOp = builder.create<bmo::ExternalCallOp>(
        loc, resultTypes, functionOp.getSymName(), args);

    callOp.setLanguage(language);

    if (!outputs.empty()) {
      builder.create<bmo::VariableSetOp>(loc, outputs[0].getSymNameAttr(),
                                         mlir::ValueRange(),
                                         callOp.getResult(0));
    }

    return mlir::success();
  }

  mlir::Block &body = externalOp.getBodyRegion().front();
  auto yieldOp = mlir::cast<mo::YieldOp>(body.getTerminator());
  mlir::ValueRange yielded = yieldOp.getValues();

  if (externalOp.getHasOutput()) {
    destination = yielded.front();
    yielded = yielded.drop_front();

    mlir::Type type = convertType(destination.getType(), loc);

    if (!type) {
      return mlir::failure();
    }

    resultTypes.push_back(type);
  }

  // Arguments that are references to the outputs or protected variables of
  // the function can be modified by the external function.
  llvm::DenseSet<mlir::Operation *> skipped;

  if (destination) {
    if (auto refOp = destination.getDefiningOp<mo::RefOp>();
        refOp && refOp->hasOneUse()) {
      skipped.insert(refOp);
    }
  }

  if (mlir::failed(convertExpressions(body, skipped))) {
    return mlir::failure();
  }

  for (mlir::Value arg : yielded) {
    mlir::Value value = lookup(arg);

    if (!value) {
      return mlir::failure();
    }

    auto refOp = arg.getDefiningOp<mo::RefOp>();
    mo::VariableOp variableOp = refOp && refOp.hasSimpleName()
                                    ? variables.lookup(refOp.getSimpleName())
                                    : nullptr;

    if (variableOp && variableOp.getCausality() != mo::Causality::input &&
        !mlir::isa<mlir::RankedTensorType>(value.getType())) {
      auto allocOp = builder.create<bmo::ArrayAllocOp>(
          loc, bmo::ArrayType::get({}, value.getType()), mlir::ValueRange());

      builder.create<bmo::StoreOp>(loc, value, allocOp);
      args.push_back(allocOp);
      temporaries.emplace_back(arg, allocOp);
      continue;
    }

    args.push_back(value);
  }

  auto callOp = builder.create<bmo::ExternalCallOp>(
      loc, resultTypes,
      externalOp.getFunction().value_or(functionOp.getSymName()), args);

  callOp.setLanguage(language);

  for (auto [arg, allocOp] : temporaries) {
    mlir::Value value = builder.create<bmo::LoadOp>(loc, allocOp);

    if (mlir::failed(assign(arg, value, loc))) {
      return mlir::failure();
    }
  }

  if (destination) {
    return assign(destination, callOp.getResult(0), loc);
  }

  return mlir::success();
}

//===---------------------------------------------------------------------===//
// Equations
//===---------------------------------------------------------------------===//

mlir::LogicalResult Converter::convertEquations(mlir::Block &block) {
  for (mlir::Operation &op : block) {
    if (auto forOp = mlir::dyn_cast<mo::ForEquationOp>(op)) {
      if (mlir::failed(convertForEquation(forOp))) {
        return mlir::failure();
      }

      continue;
    }

    if (auto ifOp = mlir::dyn_cast<mo::IfEquationOp>(op)) {
      // The branch selected by a condition known at compile time may contain
      // any kind of equation.
      if (auto condition = mlir::dyn_cast_or_null<mo::BooleanAttr>(
              evaluateYield(ifOp.getConditionRegion()))) {
        mlir::Region &branch =
            condition.getValue() ? ifOp.getThenRegion() : ifOp.getElseRegion();

        if (!branch.empty() && mlir::failed(convertEquations(branch.front()))) {
          return mlir::failure();
        }

        continue;
      }
    }

    if (mlir::isa<mo::EquationOp, mo::IfEquationOp>(op)) {
      llvm::SmallVector<SidesEmitter> emitters;

      if (mlir::failed(collectEquations(&op, emitters))) {
        return mlir::failure();
      }

      for (const SidesEmitter &emitter : emitters) {
        if (mlir::failed(emitEquation(emitter, op.getLoc()))) {
          return mlir::failure();
        }
      }

      continue;
    }

    if (mlir::isa<mo::AssertOp>(op)) {
      mlir::emitWarning(op.getLoc())
          << "assertions within equations are not supported, and are ignored";

      continue;
    }

    if (mlir::isa<mo::WhenEquationOp>(op)) {
      return op.emitError() << "when-equations are not supported yet";
    }

    if (mlir::isa<mo::ConnectOp>(op)) {
      return op.emitError() << "the connections must be expanded before the "
                               "conversion to Base Modelica";
    }

    return op.emitError() << "'" << op.getName()
                          << "' is not supported by the conversion to Base "
                             "Modelica";
  }

  return mlir::success();
}

bool Converter::mustUnroll(mo::ForEquationOp op) {
  mlir::Value induction = op.getInductionVariable();
  bool result = false;

  auto usesInduction = [&](mlir::Region &region) {
    return region
        .walk([&](mlir::Operation *nestedOp) {
          return llvm::is_contained(nestedOp->getOperands(), induction)
                     ? mlir::WalkResult::interrupt()
                     : mlir::WalkResult::advance();
        })
        .wasInterrupted();
  };

  op.getBodyRegion().walk([&](mlir::Operation *nestedOp) {
    if (auto forOp = mlir::dyn_cast<mo::ForEquationOp>(nestedOp)) {
      result |= usesInduction(forOp.getRangeRegion());
    } else if (auto ifOp = mlir::dyn_cast<mo::IfEquationOp>(nestedOp)) {
      result |= usesInduction(ifOp.getConditionRegion());
    }
  });

  return result;
}

mlir::LogicalResult Converter::convertForEquation(mo::ForEquationOp op) {
  if (op.hasImplicitRange()) {
    return op.emitError()
           << "for-equations with implicit iteration range are not supported";
  }

  auto yieldOp =
      mlir::cast<mo::YieldOp>(op.getRangeRegion().front().getTerminator());

  auto values = evaluateVector(yieldOp.getValues()[0]);

  if (!values) {
    return op.emitError() << "the iteration range of the for-equation must be "
                             "known at compile time";
  }

  if (values->empty()) {
    return mlir::success();
  }

  mlir::BlockArgument induction = op.getInductionVariable();
  mlir::Block &body = op.getBodyRegion().front();

  if (auto progression = getProgression(*values);
      progression && !mustUnroll(op)) {
    // Base Modelica iterates with a positive step.
    if (progression->step < 0) {
      std::swap(progression->from, progression->to);
      progression->step = -progression->step;
    }

    auto forOp = builder.create<bmo::ForEquationOp>(
        op.getLoc(), progression->from, progression->to, progression->step);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    mapping.map(induction, forOp.getInduction());

    return convertEquations(body);
  }

  for (mlir::Attribute value : *values) {
    iteratorValues[induction] = value;

    if (mlir::failed(convertEquations(body))) {
      iteratorValues.erase(induction);
      return mlir::failure();
    }
  }

  iteratorValues.erase(induction);
  return mlir::success();
}

mlir::LogicalResult
Converter::collectEquations(mlir::Block &block,
                            llvm::SmallVectorImpl<SidesEmitter> &emitters) {
  for (mlir::Operation &op : block) {
    if (mlir::isa<mo::AssertOp>(op)) {
      mlir::emitWarning(op.getLoc())
          << "assertions within equations are not supported, and are ignored";

      continue;
    }

    if (mlir::failed(collectEquations(&op, emitters))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

/// Get the most generic type between two types of Base Modelica values.
static mlir::Type getMostGenericType(mlir::Type first, mlir::Type second) {
  if (first == second) {
    return first;
  }

  auto firstTensor = mlir::dyn_cast<mlir::RankedTensorType>(first);
  auto secondTensor = mlir::dyn_cast<mlir::RankedTensorType>(second);

  if (firstTensor && secondTensor) {
    mlir::Type elementType = getMostGenericType(firstTensor.getElementType(),
                                                secondTensor.getElementType());

    return firstTensor.clone(elementType);
  }

  if (mlir::isa<bmo::RealType>(first) || mlir::isa<bmo::RealType>(second)) {
    return bmo::RealType::get(first.getContext());
  }

  if (mlir::isa<bmo::IntegerType>(first) ||
      mlir::isa<bmo::IntegerType>(second)) {
    return bmo::IntegerType::get(first.getContext());
  }

  return first;
}

mlir::LogicalResult
Converter::collectEquations(mlir::Operation *op,
                            llvm::SmallVectorImpl<SidesEmitter> &emitters) {
  if (auto equationOp = mlir::dyn_cast<mo::EquationOp>(op)) {
    emitters.push_back(
        [this, equationOp]() mutable -> mlir::FailureOr<EquationSides> {
          mlir::Block &body = equationOp.getBodyRegion().front();

          if (mlir::failed(convertExpressions(body))) {
            return mlir::failure();
          }

          auto sidesOp = mlir::cast<mo::EquationSidesOp>(body.getTerminator());
          EquationSides sides;

          for (mlir::Value value : sidesOp.getLhs()) {
            sides.lhs.push_back(lookup(value));
          }

          for (mlir::Value value : sidesOp.getRhs()) {
            sides.rhs.push_back(lookup(value));
          }

          if (llvm::is_contained(sides.lhs, nullptr) ||
              llvm::is_contained(sides.rhs, nullptr)) {
            return mlir::failure();
          }

          return sides;
        });

    return mlir::success();
  }

  auto ifOp = mlir::dyn_cast<mo::IfEquationOp>(op);

  if (!ifOp) {
    if (mlir::isa<mo::ForEquationOp>(op)) {
      return op->emitError()
             << "for-equations within if-equations whose condition is not "
                "known at compile time are not supported";
    }

    if (mlir::isa<mo::WhenEquationOp>(op)) {
      return op->emitError() << "when-equations are not supported yet";
    }

    return op->emitError() << "'" << op->getName()
                           << "' is not supported by the conversion to Base "
                              "Modelica";
  }

  if (auto condition = mlir::dyn_cast_or_null<mo::BooleanAttr>(
          evaluateYield(ifOp.getConditionRegion()))) {
    mlir::Region &branch =
        condition.getValue() ? ifOp.getThenRegion() : ifOp.getElseRegion();

    if (branch.empty()) {
      return mlir::success();
    }

    return collectEquations(branch.front(), emitters);
  }

  // The branches must have the same number of equations,
  // which are merged pairwise.
  llvm::SmallVector<SidesEmitter> thenEmitters;
  llvm::SmallVector<SidesEmitter> elseEmitters;

  if (mlir::failed(
          collectEquations(ifOp.getThenRegion().front(), thenEmitters))) {
    return mlir::failure();
  }

  if (!ifOp.getElseRegion().empty() &&
      mlir::failed(
          collectEquations(ifOp.getElseRegion().front(), elseEmitters))) {
    return mlir::failure();
  }

  if (thenEmitters.size() != elseEmitters.size()) {
    return ifOp.emitError()
           << "the branches of an if-equation whose condition is not known at "
              "compile time must have the same number of equations";
  }

  for (auto [thenEmitter, elseEmitter] :
       llvm::zip_equal(thenEmitters, elseEmitters)) {
    emitters.push_back([this, ifOp, thenEmitter, elseEmitter]() mutable
                           -> mlir::FailureOr<EquationSides> {
      auto condition = convertYieldedValues(ifOp.getConditionRegion());

      if (mlir::failed(condition)) {
        return mlir::failure();
      }

      mlir::FailureOr<EquationSides> thenSides = thenEmitter();

      if (mlir::failed(thenSides)) {
        return mlir::failure();
      }

      mlir::FailureOr<EquationSides> elseSides = elseEmitter();

      if (mlir::failed(elseSides)) {
        return mlir::failure();
      }

      if (thenSides->lhs.size() != elseSides->lhs.size() ||
          thenSides->rhs.size() != elseSides->rhs.size()) {
        ifOp.emitError() << "the equations of the branches of the if-equation "
                            "have different numbers of values";

        return mlir::failure();
      }

      mlir::Location loc = ifOp.getLoc();
      mlir::Value conditionValue = (*condition)[0];

      auto select = [&](llvm::ArrayRef<mlir::Value> thenValues,
                        llvm::ArrayRef<mlir::Value> elseValues,
                        llvm::SmallVectorImpl<mlir::Value> &result) {
        for (auto [thenValue, elseValue] :
             llvm::zip_equal(thenValues, elseValues)) {
          mlir::Type type =
              getMostGenericType(thenValue.getType(), elseValue.getType());

          result.push_back(
              builder
                  .create<bmo::SelectOp>(loc, type, conditionValue,
                                         castTo(thenValue, type, loc),
                                         castTo(elseValue, type, loc))
                  .getResult(0));
        }
      };

      EquationSides sides;
      select(thenSides->lhs, elseSides->lhs, sides.lhs);
      select(thenSides->rhs, elseSides->rhs, sides.rhs);
      return sides;
    });
  }

  return mlir::success();
}

mlir::LogicalResult Converter::emitEquation(const SidesEmitter &emitter,
                                            mlir::Location loc) {
  auto equationOp = builder.create<bmo::EquationOp>(loc);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(
      builder.createBlock(&equationOp.getBodyRegion()));

  mlir::FailureOr<EquationSides> sides = emitter();

  if (mlir::failed(sides)) {
    return mlir::failure();
  }

  mlir::Value lhs = builder.create<bmo::EquationSideOp>(loc, sides->lhs);
  mlir::Value rhs = builder.create<bmo::EquationSideOp>(loc, sides->rhs);
  builder.create<bmo::EquationSidesOp>(loc, lhs, rhs);
  return mlir::success();
}

//===---------------------------------------------------------------------===//
// Statements
//===---------------------------------------------------------------------===//

mlir::LogicalResult Converter::convertStatements(mlir::Block &block) {
  for (mlir::Operation &op : block) {
    if (mlir::failed(convertStatement(&op))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult Converter::convertStatement(mlir::Operation *op) {
  mlir::Location loc = op->getLoc();

  if (auto assignmentOp = mlir::dyn_cast<mo::AssignmentOp>(op)) {
    return convertAssignment(assignmentOp);
  }

  if (auto callOp = mlir::dyn_cast<mo::CallStatementOp>(op)) {
    auto args = convertYieldedValues(callOp.getBodyRegion());

    if (mlir::failed(args)) {
      return mlir::failure();
    }

    auto orderedArgs =
        orderCallArguments(op, callOp.getCallee(), *args, callOp.getArgNames());

    if (mlir::failed(orderedArgs)) {
      return mlir::failure();
    }

    auto resultTypes = getFunctionResultTypes(callOp.getCallee(), loc);

    if (mlir::failed(resultTypes)) {
      return mlir::failure();
    }

    builder.create<bmo::CallOp>(loc, callOp.getCallee(), *resultTypes,
                                *orderedArgs);

    return mlir::success();
  }

  if (auto ifOp = mlir::dyn_cast<mo::IfOp>(op)) {
    auto condition = convertYieldedValues(ifOp.getConditionRegion());

    if (mlir::failed(condition)) {
      return mlir::failure();
    }

    bool hasElse = !ifOp.getElseRegion().empty();
    auto bmoIfOp = builder.create<bmo::IfOp>(loc, (*condition)[0], hasElse);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bmoIfOp.thenBlock());

    if (mlir::failed(convertStatements(ifOp.getThenRegion().front()))) {
      return mlir::failure();
    }

    if (hasElse) {
      builder.setInsertionPointToStart(bmoIfOp.elseBlock());

      if (mlir::failed(convertStatements(ifOp.getElseRegion().front()))) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }

  if (auto forOp = mlir::dyn_cast<mo::ForOp>(op)) {
    return convertForStatement(forOp);
  }

  if (auto whileOp = mlir::dyn_cast<mo::WhileOp>(op)) {
    auto bmoWhileOp = builder.create<bmo::WhileOp>(loc);
    mlir::OpBuilder::InsertionGuard guard(builder);

    builder.setInsertionPointToStart(
        builder.createBlock(&bmoWhileOp.getConditionRegion()));

    auto condition = convertYieldedValues(whileOp.getConditionRegion());

    if (mlir::failed(condition)) {
      return mlir::failure();
    }

    builder.create<bmo::ConditionOp>(loc, (*condition)[0]);

    builder.setInsertionPointToStart(
        builder.createBlock(&bmoWhileOp.getBodyRegion()));

    if (mlir::failed(convertStatements(whileOp.getBodyRegion().front()))) {
      return mlir::failure();
    }

    builder.create<bmo::YieldOp>(loc);
    return mlir::success();
  }

  if (mlir::isa<mo::BreakOp>(op)) {
    builder.create<bmo::BreakOp>(loc);
    return mlir::success();
  }

  if (mlir::isa<mo::ReturnOp>(op)) {
    builder.create<bmo::ReturnOp>(loc);
    return mlir::success();
  }

  if (auto assertOp = mlir::dyn_cast<mo::AssertOp>(op)) {
    return convertAssert(assertOp);
  }

  if (mlir::isa<mo::WhenOp>(op)) {
    return op->emitError() << "when-statements are not supported yet";
  }

  return op->emitError() << "'" << op->getName()
                         << "' is not supported by the conversion to Base "
                            "Modelica";
}

/// Collect the operations computing the component reference of the
/// destination of an assignment, which are not to be converted as reads.
static void collectDestinationOps(mlir::Value destination,
                                  llvm::DenseSet<mlir::Operation *> &ops) {
  mlir::Value current = destination;

  while (mlir::Operation *op = current.getDefiningOp()) {
    if (!op->hasOneUse()) {
      return;
    }

    if (auto subscriptOp = mlir::dyn_cast<mo::SubscriptOp>(op)) {
      ops.insert(op);
      current = subscriptOp.getArray();
      continue;
    }

    if (mlir::isa<mo::RefOp>(op)) {
      ops.insert(op);
    }

    return;
  }
}

mlir::LogicalResult Converter::convertAssignment(mo::AssignmentOp op) {
  mlir::Block &body = op.getBodyRegion().front();
  auto sidesOp = mlir::cast<mo::AssignmentSidesOp>(body.getTerminator());

  llvm::DenseSet<mlir::Operation *> skipped;

  for (mlir::Value destination : sidesOp.getDestinations()) {
    collectDestinationOps(destination, skipped);
  }

  if (mlir::failed(convertExpressions(body, skipped))) {
    return mlir::failure();
  }

  for (auto [destination, value] :
       llvm::zip_equal(sidesOp.getDestinations(), sidesOp.getValues())) {
    mlir::Value convertedValue = lookup(value);

    if (!convertedValue ||
        mlir::failed(assign(destination, convertedValue, op.getLoc()))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult Converter::assign(mlir::Value destination,
                                      mlir::Value value, mlir::Location loc) {
  llvm::SmallVector<mo::SubscriptOp> subscriptOps;
  mlir::Value current = destination;

  while (auto subscriptOp = current.getDefiningOp<mo::SubscriptOp>()) {
    subscriptOps.push_back(subscriptOp);
    current = subscriptOp.getArray();
  }

  auto refOp = current.getDefiningOp<mo::RefOp>();

  if (!refOp || !refOp.hasSimpleName()) {
    return mlir::emitError(loc) << "only component references can be assigned";
  }

  mo::VariableOp variableOp = variables.lookup(refOp.getSimpleName());

  if (!variableOp) {
    return mlir::emitError(loc) << "unknown variable " << refOp.getName();
  }

  mlir::Type variableType = convertType(variableOp.getType(), loc);

  if (!variableType) {
    return mlir::failure();
  }

  mlir::StringAttr name = variableOp.getSymNameAttr();

  if (subscriptOps.empty()) {
    if (!mlir::isa<mlir::RankedTensorType>(variableType)) {
      value = castTo(value, variableType, loc);
    }

    builder.create<bmo::VariableSetOp>(loc, name, mlir::ValueRange(), value);
    return mlir::success();
  }

  if (subscriptOps.size() != 1) {
    return mlir::emitError(loc)
           << "assignments to multiple levels of subscripts are not supported";
  }

  mo::SubscriptOp subscriptOp = subscriptOps.front();
  auto tensorType = mlir::cast<mlir::RankedTensorType>(variableType);
  llvm::SmallVector<mlir::Value> subscripts;
  bool allScalar = true;
  auto indices = subscriptOp.getIndices();
  size_t nextIndex = 0;

  for (bool isColon : subscriptOp.getColons()) {
    if (isColon) {
      allScalar = false;

      subscripts.push_back(builder.create<bmo::RangeUnboundedOp>(
          loc,
          bmo::RangeType::get(builder.getContext(), builder.getIndexType())));

      continue;
    }

    mlir::Value index = indices[nextIndex++];
    mlir::Value converted;

    if (mlir::isa<mlir::RankedTensorType>(index.getType())) {
      allScalar = false;
      converted = convertVectorIndex(index, loc);
    } else {
      converted = convertScalarIndex(index, loc);
    }

    if (!converted) {
      return mlir::failure();
    }

    subscripts.push_back(converted);
  }

  if (allScalar &&
      static_cast<int64_t>(subscripts.size()) == tensorType.getRank()) {
    builder.create<bmo::VariableSetOp>(
        loc, name, subscripts, castTo(value, tensorType.getElementType(), loc));

    return mlir::success();
  }

  mlir::Value tensor =
      builder.create<bmo::VariableGetOp>(loc, tensorType, name.getValue());

  mlir::Value updated = builder.create<bmo::TensorInsertSliceOp>(
      loc, tensorType, value, tensor, subscripts);

  builder.create<bmo::VariableSetOp>(loc, name, mlir::ValueRange(), updated);
  return mlir::success();
}

mlir::LogicalResult Converter::convertForStatement(mo::ForOp op) {
  mlir::Location loc = op.getLoc();
  mlir::Block &rangeBlock = op.getRangeRegion().front();
  mlir::Value range =
      mlir::cast<mo::YieldOp>(rangeBlock.getTerminator()).getValues()[0];

  if (mlir::failed(convertExpressions(rangeBlock))) {
    return mlir::failure();
  }

  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
  mlir::Value vector;
  bool decreasing = false;

  if (auto rangeOp = range.getDefiningOp<mo::RangeOp>()) {
    // Iteration over the values of the range.
    lowerBound = lookup(rangeOp.getStart());
    upperBound = lookup(rangeOp.getStop());

    if (!lowerBound || !upperBound) {
      return mlir::failure();
    }

    if (mlir::Value rangeStep = rangeOp.getStep()) {
      step = lookup(rangeStep);

      if (!step) {
        return mlir::failure();
      }

      auto stepValue =
          mlir::dyn_cast_or_null<mo::IntegerAttr>(evaluate(rangeStep));

      if (!stepValue) {
        return op.emitError()
               << "the step of the range of a for-statement must be known at "
                  "compile time";
      }

      decreasing = stepValue.getValue() < 0;
    } else {
      step = createIndex(loc, 1);
    }

    if (!mlir::isa<bmo::IntegerType, mlir::IndexType>(lowerBound.getType())) {
      return op.emitError()
             << "for-statements are supported only on Integer ranges";
    }

    lowerBound = castTo(lowerBound, builder.getIndexType(), loc);
    upperBound = castTo(upperBound, builder.getIndexType(), loc);
    step = castTo(step, builder.getIndexType(), loc);
  } else {
    // Iteration over the elements of a vector.
    vector = lookup(range);

    if (!vector) {
      return mlir::failure();
    }

    lowerBound = createIndex(loc, 0);

    mlir::Value size = builder.create<bmo::SizeOp>(
        loc, bmo::IntegerType::get(builder.getContext()), vector,
        createIndex(loc, 0));

    upperBound = builder.create<bmo::SubOp>(loc, builder.getIndexType(), size,
                                            createIndex(loc, 1));

    step = createIndex(loc, 1);
  }

  mlir::Value inductionVar = builder.create<bmo::ArrayAllocaOp>(
      loc, bmo::ArrayType::get({}, builder.getIndexType()), mlir::ValueRange());

  builder.create<bmo::StoreOp>(loc, lowerBound, inductionVar);

  auto forOp = builder.create<bmo::ForOp>(loc, mlir::ValueRange());
  mlir::OpBuilder::InsertionGuard guard(builder);

  mlir::Block *conditionBlock =
      builder.createBlock(&forOp.getConditionRegion());
  mlir::Block *bodyBlock = builder.createBlock(&forOp.getBodyRegion());
  mlir::Block *stepBlock = builder.createBlock(&forOp.getStepRegion());

  // Condition.
  builder.setInsertionPointToStart(conditionBlock);
  mlir::Value inductionValue = builder.create<bmo::LoadOp>(loc, inductionVar);
  mlir::Type boolType = bmo::BooleanType::get(builder.getContext());
  mlir::Value condition;

  if (decreasing) {
    condition =
        builder.create<bmo::GteOp>(loc, boolType, inductionValue, upperBound);
  } else {
    condition =
        builder.create<bmo::LteOp>(loc, boolType, inductionValue, upperBound);
  }

  builder.create<bmo::ConditionOp>(loc, condition);

  // Body.
  builder.setInsertionPointToStart(bodyBlock);
  inductionValue = builder.create<bmo::LoadOp>(loc, inductionVar);

  if (vector) {
    mapping.map(
        op.getInductionVariable(),
        builder.create<bmo::TensorExtractOp>(loc, vector, inductionValue)
            .getResult());
  } else {
    mapping.map(op.getInductionVariable(), inductionValue);
  }

  if (mlir::failed(convertStatements(op.getBodyRegion().front()))) {
    return mlir::failure();
  }

  builder.create<bmo::YieldOp>(loc);

  // Step.
  builder.setInsertionPointToStart(stepBlock);
  inductionValue = builder.create<bmo::LoadOp>(loc, inductionVar);

  mlir::Value incremented = builder.create<bmo::AddOp>(
      loc, builder.getIndexType(), inductionValue, step);

  builder.create<bmo::StoreOp>(loc, incremented, inductionVar);
  builder.create<bmo::YieldOp>(loc);

  return mlir::success();
}

mlir::LogicalResult Converter::convertAssert(mo::AssertOp op) {
  mlir::Location loc = op.getLoc();
  mlir::Block &body = op.getBodyRegion().front();
  auto yieldOp = mlir::cast<mo::YieldOp>(body.getTerminator());
  mlir::ValueRange values = yieldOp.getValues();

  // The message and the level are attributes of the Base Modelica operation,
  // and thus must be known at compile time.
  std::string message = "Assertion failed";

  if (auto messageAttr =
          mlir::dyn_cast_or_null<mo::StringValueAttr>(evaluate(values[1]))) {
    message = messageAttr.getValue().str();
  }

  // AssertionLevel = enumeration(error, warning).
  auto level = bmo::AssertionLevel::Error;

  if (values.size() > 2) {
    if (auto levelAttr =
            mlir::dyn_cast_or_null<mo::EnumLiteralAttr>(evaluate(values[2]));
        levelAttr && levelAttr.getLiteral() == "warning") {
      level = bmo::AssertionLevel::Warning;
    }
  }

  // Skip the operations computing only the message and the level.
  llvm::DenseSet<mlir::Operation *> conditionOps;
  llvm::SmallVector<mlir::Value> worklist = {values[0]};

  while (!worklist.empty()) {
    mlir::Operation *definingOp = worklist.pop_back_val().getDefiningOp();

    if (definingOp && conditionOps.insert(definingOp).second) {
      llvm::append_range(worklist, definingOp->getOperands());

      definingOp->walk([&](mlir::Operation *nestedOp) {
        for (mlir::Value operand : nestedOp->getOperands()) {
          if (operand.getParentBlock() == &body) {
            worklist.push_back(operand);
          }
        }
      });
    }
  }

  llvm::DenseSet<mlir::Operation *> skipped;

  for (mlir::Operation &nestedOp : body) {
    if (!conditionOps.contains(&nestedOp)) {
      skipped.insert(&nestedOp);
    }
  }

  auto bmoAssertOp =
      builder.create<bmo::AssertOp>(loc, builder.getStringAttr(message), level);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(
      builder.createBlock(&bmoAssertOp.getConditionRegion()));

  if (mlir::failed(convertExpressions(body, skipped))) {
    return mlir::failure();
  }

  mlir::Value condition = lookup(values[0]);

  if (!condition) {
    return mlir::failure();
  }

  builder.create<bmo::YieldOp>(loc, condition);
  return mlir::success();
}

//===---------------------------------------------------------------------===//
// Expressions
//===---------------------------------------------------------------------===//

mlir::LogicalResult Converter::convertExpressions(
    mlir::Block &block, const llvm::DenseSet<mlir::Operation *> &skipped) {
  for (mlir::Operation &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>() || skipped.contains(&op)) {
      continue;
    }

    if (mlir::failed(convertExpression(&op))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

template <typename Op>
mlir::LogicalResult Converter::convertOneToOne(mlir::Operation *op) {
  llvm::SmallVector<mlir::Type, 1> resultTypes;
  llvm::SmallVector<mlir::Value> operands;

  for (mlir::Type type : op->getResultTypes()) {
    mlir::Type converted = convertType(type, op->getLoc());

    if (!converted) {
      return mlir::failure();
    }

    resultTypes.push_back(converted);
  }

  for (mlir::Value operand : op->getOperands()) {
    mlir::Value converted = lookup(operand);

    if (!converted) {
      return mlir::failure();
    }

    operands.push_back(converted);
  }

  auto newOp = builder.create<Op>(op->getLoc(), mlir::TypeRange(resultTypes),
                                  mlir::ValueRange(operands));

  for (auto [oldResult, newResult] :
       llvm::zip_equal(op->getResults(), newOp->getResults())) {
    mapping.map(oldResult, newResult);
  }

  return mlir::success();
}

mlir::LogicalResult Converter::convertExpression(mlir::Operation *op) {
  mlir::Location loc = op->getLoc();

  auto passThrough = [&](mlir::Value operand) -> mlir::LogicalResult {
    mlir::Value converted = lookup(operand);

    if (!converted) {
      return mlir::failure();
    }

    mapping.map(op->getResult(0), converted);
    return mlir::success();
  };

  if (auto constantOp = mlir::dyn_cast<mo::ConstantOp>(op)) {
    mlir::TypedAttr attr = convertAttribute(constantOp.getValue(), loc);

    if (!attr) {
      return mlir::failure();
    }

    mapping.map(constantOp.getResult(),
                builder.create<bmo::ConstantOp>(loc, attr).getResult());

    return mlir::success();
  }

  if (auto refOp = mlir::dyn_cast<mo::RefOp>(op)) {
    if (!refOp.hasSimpleName() ||
        !variables.contains(refOp.getSimpleName())) {
      return refOp.emitError() << "unknown variable " << refOp.getName();
    }

    mlir::Type type = convertType(refOp.getResult().getType(), loc);

    if (!type) {
      return mlir::failure();
    }

    mapping.map(
        refOp.getResult(),
        builder.create<bmo::VariableGetOp>(loc, type, refOp.getSimpleName())
            .getResult());

    return mlir::success();
  }

  // The ranges are converted depending on their usage.
  if (mlir::isa<mo::RangeOp>(op)) {
    return mlir::success();
  }

  if (mlir::isa<mo::TimeOp>(op)) {
    mapping.map(op->getResult(0), builder.create<bmo::TimeOp>(loc).getResult());
    return mlir::success();
  }

  if (auto subscriptOp = mlir::dyn_cast<mo::SubscriptOp>(op)) {
    return convertSubscript(subscriptOp);
  }

  if (auto ifExprOp = mlir::dyn_cast<mo::IfExprOp>(op)) {
    return convertIfExpr(ifExprOp);
  }

  if (auto arrayOp = mlir::dyn_cast<mo::ArrayOp>(op)) {
    return convertArray(arrayOp);
  }

  if (auto arrayForOp = mlir::dyn_cast<mo::ArrayForOp>(op)) {
    return convertArrayFor(arrayForOp);
  }

  if (auto reductionOp = mlir::dyn_cast<mo::ReductionOp>(op)) {
    return convertReduction(reductionOp);
  }

  if (auto callOp = mlir::dyn_cast<mo::CallOp>(op)) {
    return convertCall(callOp);
  }

  if (mlir::isa<mo::FillOp, mo::ZerosOp, mo::OnesOp>(op)) {
    auto resultType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
        convertType(op->getResult(0).getType(), loc));

    if (!resultType) {
      return mlir::failure();
    }

    mlir::Type elementType = resultType.getElementType();
    mlir::ValueRange sizes = op->getOperands();
    mlir::Value value;

    if (auto fillOp = mlir::dyn_cast<mo::FillOp>(op)) {
      value = lookup(fillOp.getValue());
      sizes = fillOp.getSizes();

      if (!value) {
        return mlir::failure();
      }

      if (mlir::isa<mlir::RankedTensorType>(value.getType())) {
        return op->emitError() << "fill of arrays is not supported";
      }

      value = castTo(value, elementType, loc);
    } else {
      int64_t constant = mlir::isa<mo::OnesOp>(op) ? 1 : 0;

      value = builder.create<bmo::ConstantOp>(
          loc, bmo::IntegerAttr::get(builder.getContext(), constant));

      value = castTo(value, elementType, loc);
    }

    // The dimensions known at compile time become static.
    llvm::SmallVector<int64_t> shape(resultType.getShape());
    llvm::SmallVector<mlir::Value> dynamicSizes;

    for (auto [size, dimension] : llvm::zip_equal(sizes, shape)) {
      if (dimension == mlir::ShapedType::kDynamic) {
        if (auto sizeAttr =
                mlir::dyn_cast_or_null<mo::IntegerAttr>(evaluate(size))) {
          dimension = sizeAttr.getValue();
          continue;
        }

        mlir::Value converted = lookup(size);

        if (!converted) {
          return mlir::failure();
        }

        dynamicSizes.push_back(castTo(converted, builder.getIndexType(), loc));
      }
    }

    mapping.map(op->getResult(0),
                builder
                    .create<bmo::TensorBroadcastOp>(
                        loc, resultType.clone(shape), value, dynamicSizes)
                    .getResult());

    return mlir::success();
  }

  if (auto reduceOp = mlir::dyn_cast<mo::ArrayReduceOp>(op)) {
    switch (reduceOp.getKind()) {
    case mo::ReductionKind::sum:
      return convertOneToOne<bmo::SumOp>(op);
    case mo::ReductionKind::product:
      return convertOneToOne<bmo::ProductOp>(op);
    case mo::ReductionKind::min:
      return convertOneToOne<bmo::MinOp>(op);
    case mo::ReductionKind::max:
      return convertOneToOne<bmo::MaxOp>(op);
    }
  }

  if (auto sizeOp = mlir::dyn_cast<mo::SizeOp>(op)) {
    mlir::Value array = lookup(sizeOp.getArray());
    mlir::Type resultType = convertType(sizeOp.getResult().getType(), loc);

    if (!array || !resultType) {
      return mlir::failure();
    }

    mlir::Value result;

    if (mlir::Value dimension = sizeOp.getDim()) {
      mlir::Value index = convertScalarIndex(dimension, loc);

      if (!index) {
        return mlir::failure();
      }

      result = builder.create<bmo::SizeOp>(loc, resultType, array, index);
    } else {
      result =
          builder.create<bmo::SizeOp>(loc, resultType, array, mlir::Value());
    }

    mapping.map(sizeOp.getResult(), result);
    return mlir::success();
  }

  if (auto promoteOp = mlir::dyn_cast<mo::PromoteOp>(op)) {
    // Appending dimensions of size 1 doesn't change the order of the
    // elements.
    mlir::Value operand = lookup(promoteOp.getOperand());
    auto resultType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
        convertType(promoteOp.getResult().getType(), loc));

    if (!operand || !resultType) {
      return mlir::failure();
    }

    if (!resultType.hasStaticShape()) {
      return op->emitError() << "promote of arrays with dimensions not known "
                                "at compile time is not supported";
    }

    llvm::SmallVector<mlir::Value> elements;

    if (mlir::failed(
            getElements(operand, resultType.getElementType(), loc, elements))) {
      return mlir::failure();
    }

    mapping.map(
        promoteOp.getResult(),
        builder.create<bmo::TensorFromElementsOp>(loc, resultType, elements)
            .getResult());

    return mlir::success();
  }

  if (auto semiLinearOp = mlir::dyn_cast<mo::SemiLinearOp>(op)) {
    // semiLinear(x, kPos, kNeg) = if x >= 0 then kPos * x else kNeg * x.
    mlir::Type resultType =
        convertType(semiLinearOp.getResult().getType(), loc);

    if (!resultType) {
      return mlir::failure();
    }

    if (mlir::isa<mlir::RankedTensorType>(resultType)) {
      return op->emitError() << "semiLinear of arrays is not supported";
    }

    llvm::SmallVector<mlir::Value, 3> operands;

    for (mlir::Value operand : semiLinearOp.getValues()) {
      mlir::Value converted = lookup(operand);

      if (!converted) {
        return mlir::failure();
      }

      operands.push_back(converted);
    }

    mlir::Value zero = builder.create<bmo::ConstantOp>(
        loc, bmo::IntegerAttr::get(builder.getContext(), 0));

    mlir::Value condition = builder.create<bmo::GteOp>(
        loc, bmo::BooleanType::get(builder.getContext()), operands[0], zero);

    mlir::Value positive =
        builder.create<bmo::MulOp>(loc, resultType, operands[1], operands[0]);

    mlir::Value negative =
        builder.create<bmo::MulOp>(loc, resultType, operands[2], operands[0]);

    mapping.map(semiLinearOp.getResult(),
                builder
                    .create<bmo::SelectOp>(loc, resultType, condition, positive,
                                           negative)
                    .getResult(0));

    return mlir::success();
  }

  if (auto scalarOp = mlir::dyn_cast<mo::ScalarOp>(op)) {
    mlir::Value array = lookup(scalarOp.getOperand());

    if (!array) {
      return mlir::failure();
    }

    auto tensorType = mlir::cast<mlir::RankedTensorType>(array.getType());

    llvm::SmallVector<mlir::Value> indices(tensorType.getRank(),
                                           createIndex(loc, 0));

    mapping.map(
        scalarOp.getResult(),
        builder.create<bmo::TensorExtractOp>(loc, array, indices).getResult());

    return mlir::success();
  }

  // Operations without a counterpart in Base Modelica, whose result is the
  // value of one of their operands.
  if (mlir::isa<mo::EnumToIntegerOp, mo::IntegerToEnumOp, mo::NoEventOp>(op)) {
    return passThrough(op->getOperand(0));
  }

  if (auto smoothOp = mlir::dyn_cast<mo::SmoothOp>(op)) {
    return passThrough(smoothOp.getExpr());
  }

  if (auto homotopyOp = mlir::dyn_cast<mo::HomotopyOp>(op)) {
    return passThrough(homotopyOp.getActual());
  }

  return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
      .Case<mo::NegateOp>(
          [&](auto) { return convertOneToOne<bmo::NegateOp>(op); })
      .Case<mo::AddOp>([&](auto) { return convertOneToOne<bmo::AddOp>(op); })
      .Case<mo::SubOp>([&](auto) { return convertOneToOne<bmo::SubOp>(op); })
      .Case<mo::MulOp>([&](auto) { return convertOneToOne<bmo::MulOp>(op); })
      .Case<mo::DivOp>([&](auto) { return convertOneToOne<bmo::DivOp>(op); })
      .Case<mo::PowOp>([&](auto) { return convertOneToOne<bmo::PowOp>(op); })
      .Case<mo::AddEWOp>(
          [&](auto) { return convertOneToOne<bmo::AddEWOp>(op); })
      .Case<mo::SubEWOp>(
          [&](auto) { return convertOneToOne<bmo::SubEWOp>(op); })
      .Case<mo::MulEWOp>(
          [&](auto) { return convertOneToOne<bmo::MulEWOp>(op); })
      .Case<mo::DivEWOp>(
          [&](auto) { return convertOneToOne<bmo::DivEWOp>(op); })
      .Case<mo::PowEWOp>(
          [&](auto) { return convertOneToOne<bmo::PowEWOp>(op); })
      .Case<mo::EqOp>([&](auto) { return convertOneToOne<bmo::EqOp>(op); })
      .Case<mo::NotEqOp>(
          [&](auto) { return convertOneToOne<bmo::NotEqOp>(op); })
      .Case<mo::LtOp>([&](auto) { return convertOneToOne<bmo::LtOp>(op); })
      .Case<mo::LteOp>([&](auto) { return convertOneToOne<bmo::LteOp>(op); })
      .Case<mo::GtOp>([&](auto) { return convertOneToOne<bmo::GtOp>(op); })
      .Case<mo::GteOp>([&](auto) { return convertOneToOne<bmo::GteOp>(op); })
      .Case<mo::NotOp>([&](auto) { return convertOneToOne<bmo::NotOp>(op); })
      .Case<mo::AndOp>([&](auto) { return convertOneToOne<bmo::AndOp>(op); })
      .Case<mo::OrOp>([&](auto) { return convertOneToOne<bmo::OrOp>(op); })
      .Case<mo::AbsOp>([&](auto) { return convertOneToOne<bmo::AbsOp>(op); })
      .Case<mo::SignOp>([&](auto) { return convertOneToOne<bmo::SignOp>(op); })
      .Case<mo::SqrtOp>([&](auto) { return convertOneToOne<bmo::SqrtOp>(op); })
      .Case<mo::CeilOp>([&](auto) { return convertOneToOne<bmo::CeilOp>(op); })
      .Case<mo::FloorOp>(
          [&](auto) { return convertOneToOne<bmo::FloorOp>(op); })
      .Case<mo::IntegerOp>(
          [&](auto) { return convertOneToOne<bmo::IntegerOp>(op); })
      .Case<mo::DivTruncOp>(
          [&](auto) { return convertOneToOne<bmo::DivTruncOp>(op); })
      .Case<mo::ModOp>([&](auto) { return convertOneToOne<bmo::ModOp>(op); })
      .Case<mo::RemOp>([&](auto) { return convertOneToOne<bmo::RemOp>(op); })
      .Case<mo::MinOp>([&](auto) { return convertOneToOne<bmo::MinOp>(op); })
      .Case<mo::MaxOp>([&](auto) { return convertOneToOne<bmo::MaxOp>(op); })
      .Case<mo::SinOp>([&](auto) { return convertOneToOne<bmo::SinOp>(op); })
      .Case<mo::CosOp>([&](auto) { return convertOneToOne<bmo::CosOp>(op); })
      .Case<mo::TanOp>([&](auto) { return convertOneToOne<bmo::TanOp>(op); })
      .Case<mo::AsinOp>([&](auto) { return convertOneToOne<bmo::AsinOp>(op); })
      .Case<mo::AcosOp>([&](auto) { return convertOneToOne<bmo::AcosOp>(op); })
      .Case<mo::AtanOp>([&](auto) { return convertOneToOne<bmo::AtanOp>(op); })
      .Case<mo::Atan2Op>(
          [&](auto) { return convertOneToOne<bmo::Atan2Op>(op); })
      .Case<mo::SinhOp>([&](auto) { return convertOneToOne<bmo::SinhOp>(op); })
      .Case<mo::CoshOp>([&](auto) { return convertOneToOne<bmo::CoshOp>(op); })
      .Case<mo::TanhOp>([&](auto) { return convertOneToOne<bmo::TanhOp>(op); })
      .Case<mo::ExpOp>([&](auto) { return convertOneToOne<bmo::ExpOp>(op); })
      .Case<mo::LogOp>([&](auto) { return convertOneToOne<bmo::LogOp>(op); })
      .Case<mo::Log10Op>(
          [&](auto) { return convertOneToOne<bmo::Log10Op>(op); })
      .Case<mo::TransposeOp>(
          [&](auto) { return convertOneToOne<bmo::TransposeOp>(op); })
      .Case<mo::SymmetricOp>(
          [&](auto) { return convertOneToOne<bmo::SymmetricOp>(op); })
      .Case<mo::IdentityOp>(
          [&](auto) { return convertOneToOne<bmo::IdentityOp>(op); })
      .Case<mo::DiagonalOp>(
          [&](auto) { return convertOneToOne<bmo::DiagonalOp>(op); })
      .Case<mo::LinspaceOp>(
          [&](auto) { return convertOneToOne<bmo::LinspaceOp>(op); })
      .Case<mo::NDimsOp>(
          [&](auto) { return convertOneToOne<bmo::NDimsOp>(op); })
      .Case<mo::DerOp>([&](auto) { return convertOneToOne<bmo::DerOp>(op); })
      .Default([&](mlir::Operation *) {
        return op->emitError() << "'" << op->getName()
                               << "' is not supported by the conversion to "
                                  "Base Modelica";
      });
}

mlir::Value Converter::lookup(mlir::Value value) {
  mlir::Location loc = value.getLoc();

  if (auto it = iteratorValues.find(value); it != iteratorValues.end()) {
    mlir::TypedAttr attr = convertAttribute(it->second, loc);

    if (!attr) {
      return nullptr;
    }

    return builder.create<bmo::ConstantOp>(loc, attr);
  }

  if (mlir::Value converted = mapping.lookupOrNull(value)) {
    return converted;
  }

  if (value.getDefiningOp<mo::RangeOp>()) {
    // Ranges used as arrays.
    mlir::Attribute attr = evaluate(value);

    if (!attr) {
      mlir::emitError(loc) << "ranges used as arrays must have bounds known at "
                              "compile time";

      return nullptr;
    }

    mlir::TypedAttr converted = convertAttribute(attr, loc);

    if (!converted) {
      return nullptr;
    }

    return builder.create<bmo::ConstantOp>(loc, converted);
  }

  mlir::emitError(loc) << "value not converted";
  return nullptr;
}

mlir::FailureOr<llvm::SmallVector<mlir::Value>>
Converter::convertYieldedValues(mlir::Region &region) {
  mlir::Block &block = region.front();

  if (mlir::failed(convertExpressions(block))) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> result;

  for (mlir::Value value : block.getTerminator()->getOperands()) {
    mlir::Value converted = lookup(value);

    if (!converted) {
      return mlir::failure();
    }

    result.push_back(converted);
  }

  return result;
}

mlir::Value Converter::createIndex(mlir::Location loc, int64_t value) {
  return builder.create<bmo::ConstantOp>(loc, builder.getIndexAttr(value));
}

mlir::Value Converter::castTo(mlir::Value value, mlir::Type type,
                              mlir::Location loc) {
  if (value.getType() == type || mlir::isa<mlir::TensorType>(type) ||
      mlir::isa<mlir::TensorType>(value.getType())) {
    return value;
  }

  return builder.create<bmo::CastOp>(loc, type, value);
}

mlir::Value Converter::convertScalarIndex(mlir::Value index,
                                          mlir::Location loc) {
  if (auto it = iteratorValues.find(index); it != iteratorValues.end()) {
    if (auto intAttr = mlir::dyn_cast<mo::IntegerAttr>(it->second)) {
      return createIndex(loc, intAttr.getValue());
    }

    // Enumeration literals select their position within the enumeration.
    if (auto enumAttr = mlir::dyn_cast<mo::EnumLiteralAttr>(it->second)) {
      return createIndex(loc, enumAttr.getOrdinal() - 1);
    }
  }

  mlir::Type indexType = builder.getIndexType();

  // Integer arithmetic is recreated on values of type 'index', so that the
  // accesses depending on induction variables are kept affine.
  if (mlir::isa<mo::IntegerType>(index.getType())) {
    if (auto constantOp = index.getDefiningOp<mo::ConstantOp>()) {
      if (auto intAttr =
              mlir::dyn_cast<mo::IntegerAttr>(constantOp.getValue())) {
        return createIndex(loc, intAttr.getValue());
      }
    }

    mlir::Operation *op = index.getDefiningOp();

    if (op && mlir::isa<mo::AddOp, mo::SubOp, mo::MulOp, mo::NegateOp>(op) &&
        llvm::all_of(op->getOperandTypes(), [](mlir::Type type) {
          return mlir::isa<mo::IntegerType>(type);
        })) {
      llvm::SmallVector<mlir::Value, 2> operands;

      for (mlir::Value operand : op->getOperands()) {
        mlir::Value converted = convertScalarIndex(operand, loc);

        if (!converted) {
          return nullptr;
        }

        operands.push_back(converted);
      }

      if (mlir::isa<mo::AddOp>(op)) {
        return builder.createOrFold<bmo::AddOp>(loc, indexType, operands[0],
                                                operands[1]);
      }

      if (mlir::isa<mo::SubOp>(op)) {
        return builder.createOrFold<bmo::SubOp>(loc, indexType, operands[0],
                                                operands[1]);
      }

      if (mlir::isa<mo::MulOp>(op)) {
        return builder.createOrFold<bmo::MulOp>(loc, indexType, operands[0],
                                                operands[1]);
      }

      return builder.createOrFold<bmo::NegateOp>(loc, indexType, operands[0]);
    }
  }

  mlir::Value converted = lookup(index);

  if (!converted) {
    return nullptr;
  }

  // Boolean dimensions are indexed by false and true.
  return castTo(converted, indexType, loc);
}

mlir::Value Converter::convertVectorIndex(mlir::Value index,
                                          mlir::Location loc) {
  auto rangeType =
      bmo::RangeType::get(builder.getContext(), builder.getIndexType());

  if (auto rangeOp = index.getDefiningOp<mo::RangeOp>()) {
    mlir::Value start = convertScalarIndex(rangeOp.getStart(), loc);
    mlir::Value stop = convertScalarIndex(rangeOp.getStop(), loc);
    mlir::Value step = createIndex(loc, 1);

    if (mlir::Value rangeStep = rangeOp.getStep()) {
      step = lookup(rangeStep);

      if (!step) {
        return nullptr;
      }

      step = castTo(step, builder.getIndexType(), loc);
    }

    if (!start || !stop) {
      return nullptr;
    }

    return builder.create<bmo::RangeBoundedOp>(loc, rangeType, start, stop,
                                               step);
  }

  auto values = evaluateVector(index);
  std::optional<Progression> progression;

  if (values) {
    progression = getProgression(*values);
  }

  if (!progression) {
    mlir::emitError(loc) << "vector subscripts are supported only if they are "
                            "ranges";

    return nullptr;
  }

  return builder.create<bmo::RangeBoundedOp>(
      loc, rangeType, createIndex(loc, progression->from),
      createIndex(loc, progression->to),
      createIndex(loc, progression->step));
}

mlir::Value Converter::convertIterable(mlir::Value vector, mlir::Location loc) {
  auto vectorType = mlir::cast<mlir::RankedTensorType>(vector.getType());
  mlir::Type elementType = convertType(vectorType.getElementType(), loc);

  if (!elementType) {
    return nullptr;
  }

  if (!mlir::isa<bmo::IntegerType>(elementType)) {
    mlir::emitError(loc) << "iterations are supported only on Integer ranges";
    return nullptr;
  }

  auto rangeType = bmo::RangeType::get(builder.getContext(), elementType);

  if (auto rangeOp = vector.getDefiningOp<mo::RangeOp>()) {
    mlir::Value start = lookup(rangeOp.getStart());
    mlir::Value stop = lookup(rangeOp.getStop());
    mlir::Value step;

    if (mlir::Value rangeStep = rangeOp.getStep()) {
      step = lookup(rangeStep);
    } else {
      step = builder.create<bmo::ConstantOp>(
          loc, bmo::IntegerAttr::get(builder.getContext(), 1));
    }

    if (!start || !stop || !step) {
      return nullptr;
    }

    return builder.create<bmo::RangeBoundedOp>(loc, rangeType, start, stop,
                                               step);
  }

  auto values = evaluateVector(vector);
  std::optional<Progression> progression;

  if (values) {
    progression = getProgression(*values);
  }

  if (!progression) {
    mlir::emitError(loc)
        << "iterations are supported only on ranges and on vectors forming "
           "arithmetic progressions known at compile time";

    return nullptr;
  }

  auto createInteger = [&](int64_t value) -> mlir::Value {
    return builder.create<bmo::ConstantOp>(
        loc, bmo::IntegerAttr::get(builder.getContext(), value));
  };

  return builder.create<bmo::RangeBoundedOp>(
      loc, rangeType, createInteger(progression->from),
      createInteger(progression->to), createInteger(progression->step));
}

mlir::LogicalResult Converter::convertSubscript(mo::SubscriptOp op) {
  mlir::Location loc = op.getLoc();
  mlir::Value array = lookup(op.getArray());

  if (!array) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> subscripts;
  bool allScalar = true;
  auto indices = op.getIndices();
  size_t nextIndex = 0;

  for (bool isColon : op.getColons()) {
    if (isColon) {
      allScalar = false;

      subscripts.push_back(builder.create<bmo::RangeUnboundedOp>(
          loc,
          bmo::RangeType::get(builder.getContext(), builder.getIndexType())));

      continue;
    }

    mlir::Value index = indices[nextIndex++];
    mlir::Value converted;

    if (mlir::isa<mlir::RankedTensorType>(index.getType())) {
      allScalar = false;
      converted = convertVectorIndex(index, loc);
    } else {
      converted = convertScalarIndex(index, loc);
    }

    if (!converted) {
      return mlir::failure();
    }

    subscripts.push_back(converted);
  }

  auto tensorType = mlir::cast<mlir::RankedTensorType>(array.getType());
  mlir::Value result;

  if (allScalar &&
      static_cast<int64_t>(subscripts.size()) == tensorType.getRank()) {
    result = builder.create<bmo::TensorExtractOp>(loc, array, subscripts);
  } else {
    result = builder.create<bmo::TensorViewOp>(loc, array, subscripts);
  }

  mapping.map(op.getResult(), result);
  return mlir::success();
}

mlir::LogicalResult Converter::convertIfExpr(mo::IfExprOp op) {
  mlir::Location loc = op.getLoc();
  mlir::Value condition = lookup(op.getCondition());
  mlir::Type resultType = convertType(op.getResult().getType(), loc);

  if (!condition || !resultType) {
    return mlir::failure();
  }

  // Both the values are computed, and the selected one is used.
  auto thenValues = convertYieldedValues(op.getThenRegion());

  if (mlir::failed(thenValues)) {
    return mlir::failure();
  }

  auto elseValues = convertYieldedValues(op.getElseRegion());

  if (mlir::failed(elseValues)) {
    return mlir::failure();
  }

  auto selectOp = builder.create<bmo::SelectOp>(
      loc, resultType, condition, castTo((*thenValues)[0], resultType, loc),
      castTo((*elseValues)[0], resultType, loc));

  mapping.map(op.getResult(), selectOp.getResult(0));
  return mlir::success();
}

mlir::LogicalResult
Converter::getElements(mlir::Value value, mlir::Type elementType,
                       mlir::Location loc,
                       llvm::SmallVectorImpl<mlir::Value> &elements) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());

  if (!tensorType) {
    elements.push_back(castTo(value, elementType, loc));
    return mlir::success();
  }

  if (!tensorType.hasStaticShape()) {
    return mlir::emitError(loc)
           << "arrays with dimensions not known at compile time are not "
              "supported within array constructors";
  }

  llvm::ArrayRef<int64_t> shape = tensorType.getShape();
  llvm::SmallVector<int64_t> position(shape.size(), 0);

  for (int64_t i = 0, e = tensorType.getNumElements(); i < e; ++i) {
    llvm::SmallVector<mlir::Value> indices;

    for (int64_t index : position) {
      indices.push_back(createIndex(loc, index));
    }

    mlir::Value element =
        builder.create<bmo::TensorExtractOp>(loc, value, indices);

    elements.push_back(castTo(element, elementType, loc));

    // Advance the position in row-major order.
    for (size_t dim = shape.size(); dim-- > 0;) {
      if (++position[dim] < shape[dim]) {
        break;
      }

      position[dim] = 0;
    }
  }

  return mlir::success();
}

mlir::LogicalResult Converter::convertArray(mo::ArrayOp op) {
  mlir::Location loc = op.getLoc();
  auto resultType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
      convertType(op.getResult().getType(), loc));

  if (!resultType) {
    return mlir::failure();
  }

  if (!resultType.hasStaticShape()) {
    return op.emitError() << "arrays with dimensions not known at compile "
                             "time are not supported";
  }

  llvm::SmallVector<mlir::Value> elements;

  for (mlir::Value value : op.getValues()) {
    mlir::Value converted = lookup(value);

    if (!converted ||
        mlir::failed(getElements(converted, resultType.getElementType(), loc,
                                 elements))) {
      return mlir::failure();
    }
  }

  mapping.map(
      op.getResult(),
      builder.create<bmo::TensorFromElementsOp>(loc, resultType, elements)
          .getResult());

  return mlir::success();
}

mlir::LogicalResult Converter::convertArrayFor(mo::ArrayForOp op) {
  mlir::Location loc = op.getLoc();
  auto resultType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
      convertType(op.getResult().getType(), loc));

  if (!resultType) {
    return mlir::failure();
  }

  // Arrays known at compile time become constants.
  if (resultType.hasStaticShape()) {
    if (mlir::Attribute value = evaluate(op.getResult())) {
      mlir::TypedAttr attr = convertAttribute(value, loc);

      if (!attr) {
        return mlir::failure();
      }

      mapping.map(op.getResult(),
                  builder.create<bmo::ConstantOp>(loc, attr).getResult());

      return mlir::success();
    }
  }

  // Otherwise, the array is built element by element.
  llvm::SmallVector<llvm::SmallVector<mlir::Attribute>> ranges;

  for (mlir::Value range : op.getRanges()) {
    auto values = evaluateVector(range);

    if (!values) {
      return op.emitError() << "the iteration ranges of array constructors "
                               "must be known at compile time";
    }

    ranges.push_back(std::move(*values));
  }

  mlir::Block &body = op.getBody().front();
  llvm::SmallVector<mlir::Value> elements;

  // The last iterator varies along the outermost dimension.
  std::function<mlir::LogicalResult(size_t)> unroll =
      [&](size_t depth) -> mlir::LogicalResult {
    if (depth == 0) {
      auto values = convertYieldedValues(op.getBody());

      if (mlir::failed(values)) {
        return mlir::failure();
      }

      return getElements((*values)[0], resultType.getElementType(), loc,
                         elements);
    }

    mlir::BlockArgument iterator = body.getArgument(depth - 1);

    for (mlir::Attribute value : ranges[depth - 1]) {
      iteratorValues[iterator] = value;

      if (mlir::failed(unroll(depth - 1))) {
        return mlir::failure();
      }
    }

    iteratorValues.erase(iterator);
    return mlir::success();
  };

  if (mlir::failed(unroll(ranges.size()))) {
    return mlir::failure();
  }

  if (!resultType.hasStaticShape()) {
    llvm::SmallVector<int64_t> shape;

    for (const auto &range : llvm::reverse(ranges)) {
      shape.push_back(static_cast<int64_t>(range.size()));
    }

    auto yieldedType = mlir::dyn_cast<mlir::RankedTensorType>(
        mlir::cast<mo::YieldOp>(body.getTerminator()).getValues()[0].getType());

    if (yieldedType) {
      if (!yieldedType.hasStaticShape()) {
        return op.emitError() << "arrays with dimensions not known at "
                                 "compile time are not supported";
      }

      llvm::append_range(shape, yieldedType.getShape());
    }

    resultType = resultType.clone(shape);
  }

  mapping.map(
      op.getResult(),
      builder.create<bmo::TensorFromElementsOp>(loc, resultType, elements)
          .getResult());

  return mlir::success();
}

mlir::LogicalResult Converter::convertReduction(mo::ReductionOp op) {
  mlir::Location loc = op.getLoc();
  mlir::Type resultType = convertType(op.getResult().getType(), loc);

  if (!resultType) {
    return mlir::failure();
  }

  if (mlir::isa<mlir::RankedTensorType>(resultType)) {
    return op.emitError() << "reductions of arrays are not supported";
  }

  llvm::StringRef action;

  switch (op.getKind()) {
  case mo::ReductionKind::sum:
    action = "add";
    break;
  case mo::ReductionKind::product:
    action = "mul";
    break;
  case mo::ReductionKind::min:
    action = "min";
    break;
  case mo::ReductionKind::max:
    action = "max";
    break;
  }

  llvm::SmallVector<mlir::Value> iterables;

  for (mlir::Value range : op.getRanges()) {
    mlir::Value iterable = convertIterable(range, loc);

    if (!iterable) {
      return mlir::failure();
    }

    iterables.push_back(iterable);
  }

  auto bmoReductionOp =
      builder.create<bmo::ReductionOp>(loc, resultType, action, iterables);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(
      bmoReductionOp.createExpressionBlock(builder));

  for (auto [argument, induction] :
       llvm::zip_equal(op.getBody().front().getArguments(),
                       bmoReductionOp.getInductions())) {
    mapping.map(argument, induction);
  }

  auto values = convertYieldedValues(op.getBody());

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  builder.create<bmo::YieldOp>(loc, castTo((*values)[0], resultType, loc));
  mapping.map(op.getResult(), bmoReductionOp.getResult());
  return mlir::success();
}

mlir::LogicalResult Converter::convertCall(mo::CallOp op) {
  mlir::Location loc = op.getLoc();
  llvm::SmallVector<mlir::Value> args;
  llvm::SmallVector<mlir::Type> resultTypes;

  for (mlir::Value arg : op.getArgs()) {
    mlir::Value converted = lookup(arg);

    if (!converted) {
      return mlir::failure();
    }

    args.push_back(converted);
  }

  for (mlir::Type type : op.getResultTypes()) {
    mlir::Type converted = convertType(type, loc);

    if (!converted) {
      return mlir::failure();
    }

    resultTypes.push_back(converted);
  }

  auto orderedArgs =
      orderCallArguments(op, op.getCallee(), args, op.getArgNames());

  if (mlir::failed(orderedArgs)) {
    return mlir::failure();
  }

  auto callOp = builder.create<bmo::CallOp>(loc, op.getCallee(), resultTypes,
                                            *orderedArgs);

  for (auto [oldResult, newResult] :
       llvm::zip_equal(op.getResults(), callOp.getResults())) {
    mapping.map(oldResult, newResult);
  }

  return mlir::success();
}

mlir::FailureOr<llvm::SmallVector<mlir::Value>>
Converter::orderCallArguments(mlir::Operation *op, mlir::SymbolRefAttr callee,
                              mlir::ValueRange args,
                              std::optional<mlir::ArrayAttr> names) {
  llvm::SmallVector<mlir::Value> result(args.begin(), args.end());

  if (!names || names->empty()) {
    return result;
  }

  mo::FunctionOp functionOp = functions.lookup(callee.getRootReference());

  if (!functionOp) {
    return op->emitError() << "unknown function " << callee;
  }

  llvm::SmallVector<llvm::StringRef> inputs;
  llvm::SmallVector<mo::VariableOp> inputOps;

  for (auto variableOp : functionOp.getBody()->getOps<mo::VariableOp>()) {
    if (variableOp.getCausality() == mo::Causality::input) {
      inputs.push_back(variableOp.getSymName());
      inputOps.push_back(variableOp);
    }
  }

  size_t numPositional = args.size() - names->size();
  llvm::SmallVector<mlir::Value> ordered(inputs.size(), nullptr);

  for (size_t i = 0; i < numPositional; ++i) {
    ordered[i] = args[i];
  }

  for (auto [name, arg] :
       llvm::zip_equal(names->getAsValueRange<mlir::StringAttr>(),
                       args.drop_front(numPositional))) {
    auto it = llvm::find(inputs, name);

    if (it == inputs.end()) {
      return op->emitError()
             << "unknown input " << name << " of function " << callee;
    }

    ordered[std::distance(inputs.begin(), it)] = arg;
  }

  // The missing trailing arguments get their default values.
  while (!ordered.empty() && !ordered.back()) {
    ordered.pop_back();
  }

  // The missing arguments followed by other ones get their default values,
  // if they are known at compile time.
  for (auto [arg, inputOp] : llvm::zip(ordered, inputOps)) {
    if (arg) {
      continue;
    }

    mlir::Attribute value;

    if (mlir::Value binding = inputOp.getBinding()) {
      value = mo::Evaluator().evaluate(binding);
    }

    mlir::TypedAttr attr =
        value ? convertAttribute(value, op->getLoc()) : nullptr;

    if (!attr) {
      return op->emitError()
             << "the default value of the input " << inputOp.getSymName()
             << " of function " << callee
             << " must be known at compile time, as it is followed by named "
                "arguments";
    }

    arg = builder.create<bmo::ConstantOp>(op->getLoc(), attr);
  }

  return ordered;
}

mlir::FailureOr<llvm::SmallVector<mlir::Type>>
Converter::getFunctionResultTypes(mlir::SymbolRefAttr callee,
                                  mlir::Location loc) {
  mo::FunctionOp functionOp = functions.lookup(callee.getRootReference());

  if (!functionOp) {
    return mlir::emitError(loc) << "unknown function " << callee;
  }

  llvm::SmallVector<mlir::Type> result;

  for (auto variableOp : functionOp.getBody()->getOps<mo::VariableOp>()) {
    if (variableOp.getCausality() == mo::Causality::output) {
      mlir::Type type = convertType(variableOp.getType(), loc);

      if (!type) {
        return mlir::failure();
      }

      result.push_back(type);
    }
  }

  return result;
}

//===---------------------------------------------------------------------===//
// Evaluation
//===---------------------------------------------------------------------===//

mlir::Attribute Converter::evaluate(mlir::Value value) {
  mo::Evaluator evaluator(
      [this](mlir::Value value) { return evaluationHook(value); });

  return evaluator.evaluate(value);
}

std::optional<llvm::SmallVector<mlir::Attribute>>
Converter::evaluateVector(mlir::Value value) {
  auto arrayAttr = mlir::dyn_cast_or_null<mo::ArrayValueAttr>(evaluate(value));

  if (!arrayAttr || arrayAttr.getTensorType().getRank() != 1) {
    return std::nullopt;
  }

  return llvm::to_vector(arrayAttr.getElements());
}

mlir::Attribute Converter::evaluateYield(mlir::Region &region) {
  if (region.empty()) {
    return nullptr;
  }

  mlir::Operation *terminator = region.front().getTerminator();

  if (!terminator || terminator->getNumOperands() != 1) {
    return nullptr;
  }

  return evaluate(terminator->getOperand(0));
}

std::optional<mlir::Attribute> Converter::evaluationHook(mlir::Value value) {
  if (auto it = iteratorValues.find(value); it != iteratorValues.end()) {
    return it->second;
  }

  auto refOp = value.getDefiningOp<mo::RefOp>();

  if (!refOp || !refOp.hasSimpleName()) {
    return std::nullopt;
  }

  llvm::StringRef name = refOp.getSimpleName();

  if (auto it = parameterValues.find(name); it != parameterValues.end()) {
    return it->second;
  }

  // The values of parameters and constants are given by their bindings.
  mo::VariableOp variableOp = variables.lookup(name);

  if (!variableOp || !variableOp.getVariability() ||
      *variableOp.getVariability() == mo::Variability::discrete) {
    return mlir::Attribute();
  }

  mlir::Value binding = variableOp.getBinding();

  if (!binding ||
      variableOp.getModificationRegion().front().getNumArguments() != 0 ||
      !evaluatingParameters.insert(name).second) {
    return mlir::Attribute();
  }

  // The value of the binding may need the implicit conversion to the type of
  // the variable.
  mlir::Attribute result = evaluate(binding);

  if (result) {
    result = mo::convertConstant(result, refOp.getResult().getType());
  }

  evaluatingParameters.erase(name);
  parameterValues[name] = result;
  return result;
}

std::optional<Progression>
Converter::getProgression(llvm::ArrayRef<mlir::Attribute> values) {
  llvm::SmallVector<int64_t> integers;

  for (mlir::Attribute value : values) {
    if (auto intAttr = mlir::dyn_cast<mo::IntegerAttr>(value)) {
      integers.push_back(intAttr.getValue());
    } else if (auto enumAttr = mlir::dyn_cast<mo::EnumLiteralAttr>(value)) {
      integers.push_back(enumAttr.getOrdinal());
    } else {
      return std::nullopt;
    }
  }

  if (integers.empty()) {
    return std::nullopt;
  }

  if (integers.size() == 1) {
    return Progression{integers[0], integers[0], 1};
  }

  int64_t step = integers[1] - integers[0];

  if (step == 0) {
    return std::nullopt;
  }

  for (size_t i = 2; i < integers.size(); ++i) {
    if (integers[i] - integers[i - 1] != step) {
      return std::nullopt;
    }
  }

  return Progression{integers.front(), integers.back(), step};
}

//===---------------------------------------------------------------------===//
// Pass
//===---------------------------------------------------------------------===//

namespace {
class ModelicaToBaseModelicaConversionPass
    : public mlir::impl::ModelicaToBaseModelicaConversionPassBase<
          ModelicaToBaseModelicaConversionPass> {
public:
  using ModelicaToBaseModelicaConversionPassBase<
      ModelicaToBaseModelicaConversionPass>::
      ModelicaToBaseModelicaConversionPassBase;

  void runOnOperation() override {
    if (mlir::failed(mlir::convertModelicaToBaseModelica(getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
mlir::LogicalResult convertModelicaToBaseModelica(mlir::ModuleOp moduleOp) {
  return Converter(moduleOp).run();
}

std::unique_ptr<mlir::Pass> createModelicaToBaseModelicaConversionPass() {
  return std::make_unique<ModelicaToBaseModelicaConversionPass>();
}
} // namespace mlir
