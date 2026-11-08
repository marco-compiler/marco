#include "marco/Dialect/Modelica/IR/ConstantFolding.h"
#include "marco/Dialect/Modelica/Transforms/Evaluation.h"
#include "marco/Dialect/Modelica/Transforms/Instantiation/Instantiator.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::instantiation;

namespace {
/// Get the type resulting from subscripting an array.
mlir::Type getSubscriptType(mlir::Type arrayType, llvm::ArrayRef<bool> colons,
                            mlir::ValueRange indices) {
  llvm::ArrayRef<int64_t> shape = mlir::modelica::getShape(arrayType);
  llvm::SmallVector<int64_t> resultShape;
  auto index = indices.begin();

  for (size_t dim = 0; dim < shape.size(); ++dim) {
    if (dim >= colons.size() || colons[dim]) {
      resultShape.push_back(shape[dim]);
      continue;
    }

    mlir::Type indexType = (*index++).getType();

    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(indexType)) {
      resultShape.push_back(tensorType.getDimSize(0));
    }
  }

  return getValueType(resultShape, mlir::getElementTypeOrSelf(arrayType));
}

/// Merge the type of a value with the one inferred from its operands,
/// keeping the most precise information.
mlir::Type refineType(mlir::Type current, mlir::Type inferred) {
  if (!areCompatibleTypes(current, inferred)) {
    return inferred;
  }

  auto currentTensor = mlir::dyn_cast<mlir::RankedTensorType>(current);
  auto inferredTensor = mlir::dyn_cast<mlir::RankedTensorType>(inferred);

  if (!currentTensor || !inferredTensor) {
    return inferred;
  }

  llvm::SmallVector<int64_t> shape;

  for (auto [lhs, rhs] :
       llvm::zip_equal(currentTensor.getShape(), inferredTensor.getShape())) {
    shape.push_back(mlir::ShapedType::isDynamic(rhs) ? lhs : rhs);
  }

  return mlir::RankedTensorType::get(shape, inferredTensor.getElementType());
}

/// Create the operation corresponding to a function implemented by the
/// tool (external "builtin"), if known.
mlir::Value createBuiltinOperation(mlir::OpBuilder &builder,
                                   mlir::Location loc, llvm::StringRef name,
                                   mlir::ValueRange args) {
  using UnaryBuilder =
      mlir::Value (*)(mlir::OpBuilder &, mlir::Location, mlir::Value);

  static const llvm::StringMap<UnaryBuilder> unaryOps = {
#define UNARY(name, op)                                                        \
  {name, [](mlir::OpBuilder &b, mlir::Location l, mlir::Value x) {            \
     return b.create<op>(l, x).getResult();                                    \
   }}
      UNARY("sin", SinOp),     UNARY("cos", CosOp),
      UNARY("tan", TanOp),     UNARY("asin", AsinOp),
      UNARY("acos", AcosOp),   UNARY("atan", AtanOp),
      UNARY("sinh", SinhOp),   UNARY("cosh", CoshOp),
      UNARY("tanh", TanhOp),   UNARY("exp", ExpOp),
      UNARY("log", LogOp),     UNARY("log10", Log10Op),
      UNARY("sqrt", SqrtOp),   UNARY("abs", AbsOp),
      UNARY("sign", SignOp),   UNARY("ceil", CeilOp),
      UNARY("floor", FloorOp),
#undef UNARY
  };

  if (args.size() == 1) {
    if (auto it = unaryOps.find(name); it != unaryOps.end()) {
      return it->second(builder, loc, args[0]);
    }
  }

  if (args.size() == 2 && name == "atan2") {
    return builder.create<Atan2Op>(loc, args[0], args[1]);
  }

  return nullptr;
}
} // namespace

namespace mlir::modelica::instantiation {
//===---------------------------------------------------------------------===//
// Scratch area
//===---------------------------------------------------------------------===//

Instantiator::ScratchArea::ScratchArea(Instantiator *instantiator,
                                       ModelOp container)
    : instantiator(instantiator), container(container),
      builder(container.getContext()) {
  builder.setInsertionPointToStart(container.getBody());
}

Instantiator::ScratchArea::~ScratchArea() {
  container.walk([&](mlir::Operation *op) {
    for (mlir::Value result : op->getResults()) {
      instantiator->referencedComponents.erase(result);
    }
  });

  container.erase();
}

std::unique_ptr<Instantiator::ScratchArea> Instantiator::createScratchArea() {
  mlir::OpBuilder scratchBuilder(module.getContext());
  scratchBuilder.setInsertionPointToEnd(scratchModule->getBody());

  auto container =
      scratchBuilder.create<ModelOp>(module.getLoc(), "scratch");

  return std::make_unique<ScratchArea>(this, container);
}

mlir::FailureOr<mlir::Value>
Instantiator::getUnknownIndex(mlir::OpBuilder &builder, mlir::Location loc) {
  mlir::Operation *parentOp = builder.getInsertionBlock()->getParentOp();

  if (!scratchModule->getOperation()->isAncestor(parentOp)) {
    return mlir::emitError(loc)
           << "the reference depends on the index of an unknown element of "
              "an array of instances";
  }

  return unknownIndex;
}

//===---------------------------------------------------------------------===//
// Emission
//===---------------------------------------------------------------------===//

mlir::FailureOr<mlir::Value>
Instantiator::emitValue(mlir::Value value, Scope *scope,
                        const IndexValues &indices, mlir::OpBuilder &builder,
                        mlir::IRMapping &mapping) {
  if (mlir::Value mapped = mapping.lookupOrNull(value)) {
    return mapped;
  }

  mlir::Operation *definingOp = value.getDefiningOp();

  if (!definingOp) {
    return mlir::emitError(value.getLoc())
           << "the value depends on an unknown block argument";
  }

  // Collect the operations of the block the value depends on.
  mlir::Block *block = definingOp->getBlock();
  llvm::SetVector<mlir::Operation *> slice;
  llvm::SmallVector<mlir::Operation *> worklist = {definingOp};

  while (!worklist.empty()) {
    mlir::Operation *op = worklist.pop_back_val();

    if (!slice.insert(op)) {
      continue;
    }

    llvm::SetVector<mlir::Value> usedValues;
    usedValues.insert(op->getOperands().begin(), op->getOperands().end());

    for (mlir::Region &region : op->getRegions()) {
      mlir::getUsedValuesDefinedAbove(region, usedValues);
    }

    for (mlir::Value used : usedValues) {
      if (mapping.contains(used)) {
        continue;
      }

      mlir::Operation *usedDefiningOp = used.getDefiningOp();

      if (!usedDefiningOp || usedDefiningOp->getBlock() != block) {
        return mlir::emitError(value.getLoc())
               << "the value depends on an unknown value";
      }

      worklist.push_back(usedDefiningOp);
    }
  }

  llvm::SmallVector<mlir::Operation *> ops(slice.begin(), slice.end());

  llvm::sort(ops, [](mlir::Operation *lhs, mlir::Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });

  for (mlir::Operation *op : ops) {
    if (llvm::all_of(op->getResults(),
                     [&](mlir::Value result) { return mapping.contains(result); })) {
      continue;
    }

    if (mlir::failed(emitOp(op, scope, indices, builder, mapping))) {
      return mlir::failure();
    }
  }

  return mapping.lookup(value);
}

mlir::FailureOr<mlir::Operation *>
Instantiator::emitOp(mlir::Operation *op, Scope *scope,
                     const IndexValues &indices, mlir::OpBuilder &builder,
                     mlir::IRMapping &mapping) {
  mlir::Operation *clone = builder.clone(*op, mapping);
  llvm::SmallVector<mlir::Value> replacement;

  if (mlir::failed(fixup(clone, scope, indices, replacement))) {
    return mlir::failure();
  }

  if (!replacement.empty()) {
    mapping.map(op->getResults(), replacement);
    return replacement.front().getDefiningOp();
  }

  return clone;
}

mlir::FailureOr<mlir::Value>
Instantiator::emitModifierValue(const ModifierValue &value,
                                const IndexValues &indices,
                                mlir::OpBuilder &builder) {
  mlir::IRMapping mapping;

  auto result = emitValue(value.op.getValue(), value.scope, indices, builder,
                          mapping);

  if (mlir::failed(result) || value.splitPositions.empty()) {
    return result;
  }

  // Values given to arrays of instances without 'each' are split among the
  // elements. Values with less dimensions than needed are applied to all
  // the elements.
  mlir::Value array = *result;

  if (mlir::modelica::getShape(array.getType()).size() <
      value.splitPositions.size()) {
    return array;
  }

  llvm::SmallVector<mlir::Value> subscripts;

  for (unsigned position : value.splitPositions) {
    mlir::Value index =
        position < indices.size() ? indices[position] : mlir::Value();

    if (!index) {
      auto unknown = getUnknownIndex(builder, array.getLoc());

      if (mlir::failed(unknown)) {
        return mlir::failure();
      }

      index = *unknown;
    }

    subscripts.push_back(index);
  }

  llvm::SmallVector<bool> colons(subscripts.size(), false);
  mlir::Type type = getSubscriptType(array.getType(), colons, subscripts);
  return builder.create<SubscriptOp>(array.getLoc(), type, array, subscripts)
      .getResult();
}

mlir::FailureOr<mlir::Value>
Instantiator::emitReference(Component *component, const IndexValues &indices,
                            mlir::OpBuilder &builder, mlir::Location loc) {
  llvm::SmallVector<Component *> path;

  for (Component *current = component; current && !current->isRoot();
       current = current->parent) {
    path.push_back(current);
  }

  std::reverse(path.begin(), path.end());

  if (path.empty() || path.front()->parent == nullptr) {
    return mlir::emitError(loc) << "invalid reference to '" << component->name
                                << "'";
  }

  mlir::Value value;
  IndexValues currentIndices;

  for (Component *current : path) {
    auto type = getType(current);

    if (mlir::failed(type)) {
      return mlir::failure();
    }

    if (!value) {
      value = builder.create<RefOp>(loc, *type, current->name);
    } else {
      value = builder.create<MemberOp>(loc, *type, value, current->name);
    }

    referencedComponents[value] = ReferencedComponent{current, currentIndices};

    if (current == component || current->isPrimitive() ||
        current->numDeclaredDims == 0) {
      continue;
    }

    llvm::SmallVector<mlir::Value> subscripts;

    for (unsigned i = 0; i < current->numDeclaredDims; ++i) {
      unsigned position = current->indexOffset + i;
      mlir::Value index =
          position < indices.size() ? indices[position] : mlir::Value();

      if (!index) {
        auto unknown = getUnknownIndex(builder, loc);

        if (mlir::failed(unknown)) {
          return mlir::failure();
        }

        index = *unknown;
      }

      subscripts.push_back(index);
      currentIndices.push_back(index);
    }

    value = builder.create<SubscriptOp>(
        loc, mlir::getElementTypeOrSelf(*type), value, subscripts);

    referencedComponents[value] = ReferencedComponent{current, currentIndices};
  }

  return value;
}

const ReferencedComponent *
Instantiator::getReferencedComponent(mlir::Value value) const {
  auto it = referencedComponents.find(value);
  return it == referencedComponents.end() ? nullptr : &it->second;
}

mlir::LogicalResult
Instantiator::fixup(mlir::Operation *root, Scope *scope,
                    const IndexValues &indices,
                    llvm::SmallVectorImpl<mlir::Value> &replacement) {
  // Operations are visited after the ones they depend on.
  llvm::SmallVector<mlir::Operation *> ops;
  root->walk([&](mlir::Operation *op) { ops.push_back(op); });

  for (mlir::Operation *op : ops) {
    mlir::FailureOr<mlir::Value> newValue = mlir::Value();

    if (auto refOp = mlir::dyn_cast<RefOp>(op)) {
      newValue = refOp.hasSimpleName()
                     ? fixupReference(refOp, scope, indices)
                     : fixupClassReference(refOp, scope);
    } else if (auto memberOp = mlir::dyn_cast<MemberOp>(op)) {
      newValue = fixupMember(memberOp);
    } else if (auto subscriptOp = mlir::dyn_cast<SubscriptOp>(op)) {
      if (mlir::failed(fixupSubscript(subscriptOp))) {
        return mlir::failure();
      }
    } else if (mlir::isa<CallOp, CallEquationOp, CallStatementOp>(op)) {
      newValue = fixupCall(op, scope);
    } else if (auto getInstanceNameOp = mlir::dyn_cast<GetInstanceNameOp>(op)) {
      std::string name = modelName;

      if (scope->instance && !scope->instance->isRoot()) {
        name += "." + scope->instance->getPath();
      }

      mlir::OpBuilder opBuilder(op);
      newValue = opBuilder
                     .create<ConstantOp>(op->getLoc(),
                                         StringValueAttr::get(
                                             op->getContext(), name))
                     .getResult();
    } else if (auto inferTypeOp =
                   mlir::dyn_cast<mlir::InferTypeOpInterface>(op)) {
      llvm::SmallVector<mlir::Type> types;

      if (mlir::succeeded(inferTypeOp.inferReturnTypes(
              op->getContext(), op->getLoc(), op->getOperands(),
              op->getRawDictionaryAttrs(), op->getPropertiesStorage(),
              op->getRegions(), types)) &&
          types.size() == op->getNumResults()) {
        for (auto [result, type] : llvm::zip_equal(op->getResults(), types)) {
          result.setType(refineType(result.getType(), type));
        }
      }
    }

    if (mlir::failed(newValue)) {
      return mlir::failure();
    }

    if (!*newValue) {
      continue;
    }

    op->getResult(0).replaceAllUsesWith(*newValue);
    op->erase();

    if (op == root) {
      replacement.push_back(*newValue);
    }
  }

  return mlir::success();
}

mlir::FailureOr<mlir::Value>
Instantiator::fixupReference(RefOp op, Scope *scope,
                             const IndexValues &indices) {
  Element *element = lookupSimple(scope, op.getSimpleName());

  if (!element) {
    return mlir::emitError(op.getLoc())
           << "unknown name '" << op.getSimpleName() << "'";
  }

  if (!element->isComponent()) {
    return mlir::emitError(op.getLoc())
           << "the class '" << op.getSimpleName() << "' can't be used as a value";
  }

  Component *component = getComponent(element);

  if (!component) {
    return mlir::failure();
  }

  mlir::OpBuilder opBuilder(op);

  if (!component->parent) {
    // Constant of a package, or of an enclosing class.
    return emitPackageConstant(component, opBuilder, op.getLoc());
  }

  // Components removed by selective model extension.
  if (Component *parent = component->parent;
      parent->classScope &&
      !lookupMember(parent->classScope, component->name)) {
    return mlir::emitError(op.getLoc())
           << "reference to the deselected component '"
           << component->getPath() << "'";
  }

  if (component->isOuter()) {
    component = resolveOuter(component);

    if (!component) {
      return mlir::failure();
    }
  }

  auto enabled = isEnabled(component);

  if (mlir::failed(enabled)) {
    return mlir::failure();
  }

  if (!*enabled) {
    return mlir::emitError(op.getLoc())
           << "reference to the disabled conditional component '"
           << component->getPath() << "'";
  }

  return emitReference(component, indices, opBuilder, op.getLoc());
}

mlir::FailureOr<mlir::Value>
Instantiator::fixupClassReference(RefOp op, Scope *scope) {
  Element *element = lookupName(scope, op.getName());

  if (!element) {
    return mlir::emitError(op.getLoc())
           << "unknown name '" << op.getName() << "'";
  }

  if (!element->isComponent()) {
    return mlir::emitError(op.getLoc())
           << "the class '" << op.getName() << "' can't be used as a value";
  }

  Component *component = getComponent(element);

  if (!component) {
    return mlir::failure();
  }

  mlir::OpBuilder opBuilder(op);

  if (component->parent) {
    // Elements of classes that are instantiated, such as the local classes
    // of a model, refer to the components of the instance.
    return emitReference(component, {}, opBuilder, op.getLoc());
  }

  return emitPackageConstant(component, opBuilder, op.getLoc());
}

mlir::FailureOr<mlir::Value> Instantiator::fixupMember(MemberOp op) {
  const ReferencedComponent *object = getReferencedComponent(op.getObject());

  if (!object) {
    // Member of a record value, such as the result of a function call.
    return mlir::Value();
  }

  Component *objectComponent = object->component;

  if (objectComponent->isPrimitive()) {
    return mlir::emitError(op.getLoc())
           << "'" << objectComponent->getPath() << "' has no members";
  }

  Element *element = lookupMember(objectComponent->classScope, op.getMember());

  if (!element || !element->isComponent()) {
    return mlir::emitError(op.getLoc())
           << "unknown component '" << op.getMember() << "' of '"
           << objectComponent->getPath() << "'";
  }

  Component *member = getComponent(element);

  if (!member) {
    return mlir::failure();
  }

  IndexValues indices = object->indices;
  indices.resize(objectComponent->getNumIndices(), nullptr);

  if (member->isOuter()) {
    Component *inner = resolveOuter(member);

    if (!inner) {
      return mlir::failure();
    }

    mlir::OpBuilder opBuilder(op);
    return emitReference(inner, indices, opBuilder, op.getLoc());
  }

  auto enabled = isEnabled(member);

  if (mlir::failed(enabled)) {
    return mlir::failure();
  }

  if (!*enabled) {
    return mlir::emitError(op.getLoc())
           << "reference to the disabled conditional component '"
           << member->getPath() << "'";
  }

  auto memberType = getType(member);

  if (mlir::failed(memberType)) {
    return mlir::failure();
  }

  // Slice operation.
  llvm::SmallVector<int64_t> shape(mlir::modelica::getShape(op.getObject().getType()));
  llvm::ArrayRef<int64_t> memberShape = mlir::modelica::getShape(*memberType);
  shape.append(memberShape.begin(), memberShape.end());

  op.getResult().setType(
      getValueType(shape, mlir::getElementTypeOrSelf(*memberType)));

  referencedComponents[op.getResult()] = ReferencedComponent{member, indices};
  return mlir::Value();
}

mlir::LogicalResult Instantiator::fixupSubscript(SubscriptOp op) {
  op.getResult().setType(getSubscriptType(
      op.getArray().getType(), op.getColons(), op.getIndices()));

  const ReferencedComponent *array = getReferencedComponent(op.getArray());

  if (!array || array->component->isPrimitive()) {
    return mlir::success();
  }

  // Indexing of an array of instances.
  Component *component = array->component;
  IndexValues indices = array->indices;
  indices.resize(component->indexOffset, nullptr);

  auto index = op.getIndices().begin();

  for (unsigned dim = 0; dim < component->numDeclaredDims; ++dim) {
    if (dim < op.getColons().size() && !op.getColons()[dim]) {
      mlir::Value value = *index++;
      indices.push_back(mlir::isa<mlir::RankedTensorType>(value.getType())
                            ? mlir::Value()
                            : value);
    } else {
      indices.push_back(nullptr);
    }
  }

  referencedComponents[op.getResult()] =
      ReferencedComponent{component, indices};

  return mlir::success();
}

mlir::FailureOr<mlir::Value> Instantiator::fixupCall(mlir::Operation *op,
                                                     Scope *scope) {
  auto callee = op->getAttrOfType<mlir::SymbolRefAttr>("callee");
  Element *element = lookupName(scope, callee);

  if (!element) {
    return mlir::emitError(op->getLoc())
           << "unknown function '" << callee << "'";
  }

  if (!element->isClass()) {
    return mlir::emitError(op->getLoc())
           << "functional arguments are not supported yet ('" << callee
           << "')";
  }

  Scope *classScope = getClassScope(element);

  if (mlir::failed(expand(classScope))) {
    return mlir::failure();
  }

  ClassKind kind = classScope->getKind();

  if (kind == ClassKind::record || kind == ClassKind::operator_record) {
    return mlir::emitError(op->getLoc())
           << "record constructors are not supported yet ('"
           << classScope->path << "')";
  }

  if (kind != ClassKind::function && kind != ClassKind::operator_function) {
    return mlir::emitError(op->getLoc())
           << "'" << classScope->path << "' is not a function";
  }

  // Functions implemented by the tool.
  if (auto callOp = mlir::dyn_cast<CallOp>(op);
      callOp && callOp.getNumPositionalArgs() == callOp.getArgs().size() &&
      callOp.getNumResults() == 1) {
    for (auto [sectionOp, sectionScope] : classScope->sections) {
      auto externalOp = mlir::dyn_cast<ExternalOp>(sectionOp);

      if (!externalOp || externalOp.getLanguage() != "builtin") {
        continue;
      }

      llvm::StringRef name = externalOp.getFunction()
                                 ? *externalOp.getFunction()
                                 : element->getName();

      mlir::OpBuilder opBuilder(op);

      if (mlir::Value result = createBuiltinOperation(
              opBuilder, op->getLoc(), name, callOp.getArgs())) {
        return result;
      }
    }
  }

  auto function = getFunction(element, op->getLoc());

  if (mlir::failed(function)) {
    return mlir::failure();
  }

  op->setAttr("callee", *function);
  return mlir::Value();
}

mlir::FailureOr<mlir::Value>
Instantiator::emitPackageConstant(Component *component,
                                  mlir::OpBuilder &builder,
                                  mlir::Location loc) {
  auto type = getType(component);

  if (mlir::failed(type)) {
    return mlir::failure();
  }

  if (mlir::Attribute value = evaluateComponent(component, {})) {
    mlir::Attribute converted = convertConstant(value, *type);
    auto typedValue = mlir::cast<mlir::TypedAttr>(converted ? converted : value);
    return builder.create<ConstantOp>(loc, typedValue).getResult();
  }

  // The value can't be computed, and thus its expression is used.
  const Modifier &modifier = component->isPrimitive()
                                 ? component->classScope->primitiveBase->modifier
                                 : component->modifier;

  const ModifierValue *value = modifier.getValue();

  if (!value) {
    return mlir::emitError(loc) << "the constant '" << component->name
                                << "' has no value";
  }

  if (!activeEvaluations.insert(component).second) {
    return mlir::emitError(loc) << "the value of the constant '"
                                << component->name << "' depends on itself";
  }

  auto guard =
      llvm::make_scope_exit([&]() { activeEvaluations.erase(component); });

  return emitModifierValue(*value, {}, builder);
}

//===---------------------------------------------------------------------===//
// Evaluation
//===---------------------------------------------------------------------===//

mlir::Attribute Instantiator::evaluate(mlir::Value value, Scope *scope,
                                       unsigned numIndices,
                                       const mlir::IRMapping &mapping) {
  llvm::SmallVector<std::optional<int64_t>> indices(numIndices, std::nullopt);
  return evaluate(value, scope, indices, mapping);
}

mlir::Attribute
Instantiator::evaluate(mlir::Value value, Scope *scope,
                       llvm::ArrayRef<std::optional<int64_t>> indexValues,
                       const mlir::IRMapping &mapping) {
  auto scratch = createScratchArea();
  IndexValues indices;

  for (std::optional<int64_t> index : indexValues) {
    if (!index) {
      indices.push_back(nullptr);
      continue;
    }

    indices.push_back(scratch->builder
                          .create<ConstantOp>(
                              value.getLoc(),
                              IntegerAttr::get(module.getContext(), *index))
                          .getResult());
  }

  mlir::IRMapping localMapping = mapping;

  auto emitted =
      emitValue(value, scope, indices, scratch->builder, localMapping);

  if (mlir::failed(emitted)) {
    return {};
  }

  return evaluateEmitted(*emitted);
}

mlir::FailureOr<mlir::Attribute>
Instantiator::evaluateStructural(mlir::Value value, Scope *scope,
                                 Component *component, mlir::Location loc) {
  unsigned numIndices = component->indexOffset;

  if (mlir::Attribute result = evaluate(value, scope, numIndices)) {
    return result;
  }

  if (numIndices == 0 || !component->parent) {
    return mlir::Attribute();
  }

  // The value depends on the element of the enclosing arrays of instances,
  // and thus all the elements are evaluated.
  auto sizes = getIndexSizes(component->parent);

  if (mlir::failed(sizes) || sizes->size() != numIndices) {
    return mlir::Attribute();
  }

  int64_t numElements = 1;

  for (int64_t size : *sizes) {
    numElements *= size;
  }

  if (numElements == 0 || numElements > 100000) {
    return mlir::Attribute();
  }

  mlir::Attribute result;
  llvm::SmallVector<std::optional<int64_t>> indices(numIndices, 0);

  for (int64_t element = 0; element < numElements; ++element) {
    int64_t remaining = element;

    for (size_t i = numIndices; i > 0; --i) {
      indices[i - 1] = remaining % (*sizes)[i - 1];
      remaining /= (*sizes)[i - 1];
    }

    mlir::Attribute elementResult = evaluate(value, scope, indices);

    if (!elementResult) {
      return mlir::Attribute();
    }

    if (result && result != elementResult) {
      return mlir::emitError(loc)
             << "the elements of the arrays of instances enclosing '"
             << component->getPath()
             << "' have different structures, which is not supported yet";
    }

    result = elementResult;
  }

  return result;
}

mlir::Attribute Instantiator::evaluateEmitted(mlir::Value value) {
  Evaluator *evaluatorPtr = nullptr;

  Evaluator evaluator(
      [&](mlir::Value v) -> std::optional<mlir::Attribute> {
        if (v == unknownIndex) {
          return mlir::Attribute();
        }

        const ReferencedComponent *referenced = getReferencedComponent(v);

        if (!referenced) {
          return std::nullopt;
        }

        Component *component = referenced->component;

        if (!component->isPrimitive()) {
          return mlir::Attribute();
        }

        llvm::SmallVector<std::optional<int64_t>> indices;

        for (unsigned i = 0; i < component->indexOffset; ++i) {
          mlir::Value index =
              i < referenced->indices.size() ? referenced->indices[i] : nullptr;

          if (!index) {
            indices.push_back(std::nullopt);
            continue;
          }

          indices.push_back(getIndexValue(evaluatorPtr->evaluate(index)));
        }

        return evaluateComponent(component, indices);
      });

  evaluatorPtr = &evaluator;
  return evaluator.evaluate(value);
}

mlir::Attribute Instantiator::evaluateComponent(
    Component *component, llvm::ArrayRef<std::optional<int64_t>> indices) {
  std::vector<int64_t> key;

  for (unsigned i = 0; i < component->indexOffset; ++i) {
    key.push_back(i < indices.size() && indices[i] ? *indices[i] : -1);
  }

  if (auto it = evaluations.find({component, key}); it != evaluations.end()) {
    return it->second;
  }

  if (!component->isPrimitive() || !component->variability ||
      *component->variability == Variability::discrete) {
    return {};
  }

  if (!activeEvaluations.insert(component).second) {
    return {};
  }

  auto guard =
      llvm::make_scope_exit([&]() { activeEvaluations.erase(component); });

  const Modifier &modifier = component->classScope->primitiveBase->modifier;
  const ModifierValue *value = modifier.getValue();

  if (!value) {
    // Parameters without value take the start value.
    if (const Modifier *start = modifier.getChild("start");
        start && *component->variability == Variability::parameter) {
      value = start->getValue();
    }
  }

  if (!value) {
    evaluations[{component, key}] = nullptr;
    return {};
  }

  auto scratch = createScratchArea();
  IndexValues indexValues;

  for (int64_t index : key) {
    if (index < 0) {
      indexValues.push_back(nullptr);
    } else {
      indexValues.push_back(
          scratch->builder
              .create<ConstantOp>(component->getDeclaration().getLoc(),
                                  IntegerAttr::get(module.getContext(), index))
              .getResult());
    }
  }

  auto emitted = emitModifierValue(*value, indexValues, scratch->builder);
  mlir::Attribute result;

  if (mlir::succeeded(emitted)) {
    result = evaluateEmitted(*emitted);
  }

  if (result) {
    if (auto type = getType(component); mlir::succeeded(type)) {
      if (mlir::Attribute converted = convertConstant(result, *type)) {
        result = converted;
      }
    }
  }

  evaluations[{component, key}] = result;
  return result;
}
} // namespace mlir::modelica::instantiation
