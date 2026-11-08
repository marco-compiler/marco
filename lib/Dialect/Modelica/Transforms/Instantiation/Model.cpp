#include "marco/Dialect/Modelica/IR/ConstantFolding.h"
#include "marco/Dialect/Modelica/Transforms/Evaluation.h"
#include "marco/Dialect/Modelica/Transforms/Instantiation/Instantiator.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringExtras.h"

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::instantiation;

namespace {
bool containsConnect(mlir::Operation *op) {
  return op
      ->walk([](ConnectOp) { return mlir::WalkResult::interrupt(); })
      .wasInterrupted();
}

/// Get whether the operation contains if-equations whose branches contain
/// connect-equations.
bool containsConditionalConnect(mlir::Operation *op) {
  return op
      ->walk([](IfEquationOp ifEquationOp) {
        return containsConnect(ifEquationOp) ? mlir::WalkResult::interrupt()
                                             : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

/// Get the value yielded by the only block of a region.
mlir::Value getYieldedValue(mlir::Region &region, size_t index = 0) {
  return mlir::cast<YieldOp>(region.front().getTerminator()).getValues()[index];
}

/// Create the operations computing the range 'from:to' of Integer values.
mlir::Value createIntegerRange(mlir::OpBuilder &builder, mlir::Location loc,
                               int64_t from, int64_t to) {
  mlir::MLIRContext *context = builder.getContext();
  auto from_ = builder.create<ConstantOp>(loc, IntegerAttr::get(context, from));
  auto to_ = builder.create<ConstantOp>(loc, IntegerAttr::get(context, to));

  auto type = mlir::RankedTensorType::get({std::max<int64_t>(to - from + 1, 0)},
                                          IntegerType::get(context));

  return builder.create<RangeOp>(loc, type, from_, to_, mlir::Value());
}

/// Create a for-equation or a for-statement iterating over the zero-based
/// indices '0:size-1', and position the builder at the beginning of its body.
template <typename LoopOp>
std::pair<LoopOp, mlir::Value> createLoop(mlir::OpBuilder &builder,
                                          mlir::Location loc, int64_t size) {
  auto loopOp = builder.create<LoopOp>(loc);

  builder.createBlock(&loopOp.getRangeRegion());
  builder.create<YieldOp>(loc, createIntegerRange(builder, loc, 0, size - 1));

  mlir::Block *body = builder.createBlock(
      &loopOp.getBodyRegion(), {}, IntegerType::get(builder.getContext()),
      loc);

  builder.setInsertionPointToStart(body);
  return {loopOp, body->getArgument(0)};
}

/// Move the constants defined directly within the equation and algorithm
/// sections, such as the values of unrolled iteration variables, into the
/// operations using them.
void sinkConstants(mlir::Operation *container) {
  llvm::SmallVector<ConstantOp> constantOps;

  container->walk([&](ConstantOp constantOp) {
    mlir::Operation *parentOp = constantOp->getParentOp();

    if (mlir::isa<EquationSectionOp, AlgorithmSectionOp, ForEquationOp,
                  ForOp>(parentOp) &&
        constantOp->getParentRegion() == &parentOp->getRegions().back()) {
      constantOps.push_back(constantOp);
    }
  });

  for (ConstantOp constantOp : constantOps) {
    for (mlir::OpOperand &use :
         llvm::make_early_inc_range(constantOp->getUses())) {
      mlir::Operation *user = use.getOwner();
      mlir::OpBuilder builder(user);
      use.set(builder.clone(*constantOp)->getResult(0));
    }

    constantOp.erase();
  }
}

/// Get whether a class is or extends 'ExternalObject'.
bool isExternalObject(Scope *scope) {
  if (scope->predefined == PredefinedClass::ExternalObject) {
    return true;
  }

  return llvm::any_of(scope->bases, isExternalObject);
}

bool isConnector(Component *component) {
  ClassKind kind = component->classScope->getKind();
  return kind == ClassKind::connector || kind == ClassKind::expandable_connector;
}
} // namespace

namespace mlir::modelica::instantiation {
mlir::LogicalResult Instantiator::instantiateModel(llvm::StringRef name) {
  mlir::MLIRContext *context = module.getContext();
  llvm::SmallVector<llvm::StringRef> parts;
  llvm::SplitString(name, parts, ".");

  if (parts.empty()) {
    return mlir::emitError(module.getLoc()) << "no model to be instantiated";
  }

  llvm::SmallVector<mlir::FlatSymbolRefAttr> nested;

  for (llvm::StringRef part : llvm::drop_begin(parts)) {
    nested.push_back(mlir::FlatSymbolRefAttr::get(context, part));
  }

  Element *element = lookupName(
      rootScope, mlir::SymbolRefAttr::get(context, parts.front(), nested));

  if (!element || !element->isClass() || !element->op) {
    return mlir::emitError(module.getLoc()) << "unknown class '" << name << "'";
  }

  modelName = name.str();

  auto &root = components.emplace_back(std::make_unique<Component>());
  root->name = modelName;
  modelRoot = root.get();
  root->classScope = instantiateClass(element, {}, root.get());

  if (mlir::failed(expand(root->classScope))) {
    return mlir::failure();
  }

  if (root->classScope->isPrimitive()) {
    return mlir::emitError(element->op->getLoc())
           << "'" << name << "' is a simple type";
  }

  builder.setInsertionPointToEnd(module.getBody());
  modelOp = builder.create<ModelOp>(element->op->getLoc(), modelName);

  if (auto classOp = mlir::dyn_cast<ClassOp>(element->op)) {
    modelOp.setDescriptionAttr(classOp.getDescriptionAttr());
    modelOp.setAnnotationAttr(classOp.getAnnotationAttr());
  }

  mlir::OpBuilder bodyBuilder(context);
  bodyBuilder.setInsertionPointToStart(modelOp.getBody());

  if (mlir::failed(emitMembers(modelRoot, bodyBuilder)) ||
      mlir::failed(emitSections(modelRoot, modelOp))) {
    return mlir::failure();
  }

  // Inner components added for outer components without matching inner
  // ones. Their emission may require further implicit inner components.
  for (size_t i = 0; i < modelRoot->implicitInners.size(); ++i) {
    Component *component = getComponent(modelRoot->implicitInners[i]);

    if (!component) {
      return mlir::failure();
    }

    bodyBuilder.setInsertionPointToStart(modelOp.getBody());

    if (mlir::failed(emitMember(component, bodyBuilder))) {
      return mlir::failure();
    }

    if (!component->isPrimitive() &&
        mlir::failed(emitSections(component, modelOp))) {
      return mlir::failure();
    }
  }

  sinkConstants(modelOp);
  return mlir::success();
}

void Instantiator::eraseClasses() {
  for (mlir::Operation *op : classOps) {
    op->erase();
  }

  classOps.clear();
}

//===---------------------------------------------------------------------===//
// Members
//===---------------------------------------------------------------------===//

mlir::LogicalResult Instantiator::emitMembers(Component *instance,
                                              mlir::OpBuilder &builder) {
  for (Element *element : instance->classScope->elements) {
    if (!element->isComponent()) {
      continue;
    }

    Component *component = getComponent(element);

    if (!component || mlir::failed(emitMember(component, builder))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult Instantiator::emitMember(Component *component,
                                             mlir::OpBuilder &builder) {
  ComponentOp componentOp = component->getDeclaration();
  auto enabled = isEnabled(component);

  if (mlir::failed(enabled)) {
    return mlir::failure();
  }

  if (!*enabled) {
    return mlir::success();
  }

  if (component->isOuter()) {
    return mlir::success(resolveOuter(component) != nullptr);
  }

  if (isExternalObject(component->classScope)) {
    return mlir::emitError(componentOp.getLoc())
           << "external objects are not supported yet ('"
           << component->getPath() << "')";
  }

  if (!component->isPrimitive() &&
      component->classScope->predefined != PredefinedClass::None) {
    return mlir::emitError(componentOp.getLoc())
           << "components of class '" << component->classScope->path
           << "' are not supported yet ('" << component->getPath() << "')";
  }

  auto type = getType(component);

  if (mlir::failed(type)) {
    return mlir::failure();
  }

  if (component->isPrimitive()) {
    auto variableOp =
        builder.create<VariableOp>(componentOp.getLoc(), component->name, *type);

    if (auto connectorKind = componentOp.getConnectorKindAttr()) {
      variableOp.setConnectorKindAttr(connectorKind);
    }

    if (component->variability) {
      variableOp.setVariability(*component->variability);
    }

    if (component->causality) {
      variableOp.setCausality(*component->causality);
    }

    variableOp.setIsProtected(component->isProtected);
    variableOp.setDescriptionAttr(componentOp.getDescriptionAttr());

    return emitVariableModification(component, variableOp);
  }

  auto instanceOp = builder.create<InstanceOp>(
      componentOp.getLoc(), component->name, *type,
      component->classScope->getKind());

  instanceOp.setIsProtected(component->isProtected);
  instanceOp.setDescriptionAttr(componentOp.getDescriptionAttr());

  mlir::OpBuilder bodyBuilder(builder.getContext());
  bodyBuilder.setInsertionPointToStart(instanceOp.getBody());
  return emitMembers(component, bodyBuilder);
}

mlir::LogicalResult
Instantiator::emitVariableModification(Component *component,
                                       VariableOp variableOp) {
  const Modifier &modifier = component->classScope->primitiveBase->modifier;

  bool hasValues =
      modifier.getValue() ||
      llvm::any_of(llvm::make_second_range(modifier.children),
                   [](const Modifier &child) { return child.getValue(); });

  if (!hasValues) {
    return mlir::success();
  }

  mlir::MLIRContext *context = variableOp.getContext();
  mlir::Location loc = variableOp.getLoc();
  mlir::Block *block = &variableOp.getModificationRegion().emplaceBlock();

  for (unsigned i = 0; i < component->indexOffset; ++i) {
    block->addArgument(IntegerType::get(context), loc);
  }

  IndexValues indices(block->getArguments().begin(),
                      block->getArguments().end());

  mlir::OpBuilder blockBuilder(context);
  blockBuilder.setInsertionPointToStart(block);

  // Attributes of the predefined types.
  for (const auto &[name, child] : modifier.children) {
    const ModifierValue *value = child.getValue();

    if (!value) {
      continue;
    }

    auto modifyOp = blockBuilder.create<ModifyOp>(
        loc, mlir::FlatSymbolRefAttr::get(context, name), child.isEach, false,
        mlir::StringAttr());

    mlir::OpBuilder nestedBuilder(context);
    nestedBuilder.setInsertionPointToStart(
        &modifyOp.getModificationRegion().emplaceBlock());

    auto emitted = emitModifierValue(*value, indices, nestedBuilder);

    if (mlir::failed(emitted)) {
      return mlir::failure();
    }

    nestedBuilder.create<BindOp>(loc, *emitted, false);
  }

  // Declaration equation.
  if (const ModifierValue *value = modifier.getValue()) {
    auto emitted = emitModifierValue(*value, indices, blockBuilder);

    if (mlir::failed(emitted)) {
      return mlir::failure();
    }

    blockBuilder.create<BindOp>(loc, *emitted, false);
  }

  return mlir::success();
}

//===---------------------------------------------------------------------===//
// Equations and algorithms
//===---------------------------------------------------------------------===//

mlir::FailureOr<llvm::SmallVector<int64_t>>
Instantiator::getIndexSizes(Component *component) {
  llvm::SmallVector<Component *> path;

  for (Component *current = component; current && !current->isRoot();
       current = current->parent) {
    path.push_back(current);
  }

  llvm::SmallVector<int64_t> sizes;

  for (Component *current : llvm::reverse(path)) {
    if (current->isPrimitive() || current->numDeclaredDims == 0) {
      continue;
    }

    auto dims = getDims(current);

    if (mlir::failed(dims)) {
      return mlir::failure();
    }

    sizes.append(dims->begin(), dims->end());
  }

  return sizes;
}

mlir::LogicalResult Instantiator::emitSections(Component *instance,
                                               mlir::Operation *container) {
  auto modelOp = mlir::cast<ModelOp>(container);
  auto sizes = getIndexSizes(instance);

  if (mlir::failed(sizes)) {
    return mlir::failure();
  }

  auto getEquationSection = [&](bool initial) {
    for (auto sectionOp : modelOp.getBody()->getOps<EquationSectionOp>()) {
      if (sectionOp.getIsInitial() == initial) {
        return sectionOp;
      }
    }

    mlir::OpBuilder sectionBuilder(modelOp.getContext());
    sectionBuilder.setInsertionPointToEnd(modelOp.getBody());

    auto sectionOp =
        sectionBuilder.create<EquationSectionOp>(modelOp.getLoc(), initial);

    sectionBuilder.createBlock(&sectionOp.getBodyRegion());
    return sectionOp;
  };

  // The equations of arrays of instances are repeated for each element, by
  // means of for-equations shared by all the equation sections of the
  // instance.
  struct LoopNest {
    mlir::Operation *outermostLoop = nullptr;
    mlir::Block *body = nullptr;
    IndexValues indices;
  };

  LoopNest loopNests[2];

  auto getLoopNest = [&](bool initial, mlir::Location loc) -> LoopNest & {
    LoopNest &nest = loopNests[initial ? 1 : 0];

    if (nest.body) {
      return nest;
    }

    mlir::OpBuilder loopBuilder(modelOp.getContext());
    loopBuilder.setInsertionPointToEnd(getEquationSection(initial).getBody());

    for (int64_t size : *sizes) {
      auto [loopOp, index] = createLoop<ForEquationOp>(loopBuilder, loc, size);

      if (!nest.outermostLoop) {
        nest.outermostLoop = loopOp;
      }

      nest.indices.push_back(index);
    }

    nest.body = loopBuilder.getInsertionBlock();
    return nest;
  };

  for (auto [sectionOp, scope] : instance->classScope->sections) {
    mlir::Location loc = sectionOp->getLoc();
    mlir::Region &body = sectionOp->getRegion(0);

    if (body.empty() || body.front().empty()) {
      continue;
    }

    mlir::OpBuilder sectionBuilder(modelOp.getContext());
    IndexValues indices;

    if (auto equationSectionOp = mlir::dyn_cast<EquationSectionOp>(sectionOp)) {
      LoopNest &nest = getLoopNest(equationSectionOp.getIsInitial(), loc);
      sectionBuilder.setInsertionPointToEnd(nest.body);
      indices = nest.indices;
      mlir::IRMapping mapping;

      for (mlir::Operation &op : body.front()) {
        if (mlir::failed(
                emitEquation(&op, scope, indices, sectionBuilder, mapping))) {
          return mlir::failure();
        }
      }
    } else if (auto algorithmSectionOp =
                   mlir::dyn_cast<AlgorithmSectionOp>(sectionOp)) {
      sectionBuilder.setInsertionPointToEnd(modelOp.getBody());

      auto targetOp = sectionBuilder.create<AlgorithmSectionOp>(
          loc, algorithmSectionOp.getIsInitial());

      sectionBuilder.createBlock(&targetOp.getBodyRegion());

      // When-statements can't be nested within for-statements, and thus
      // the algorithms containing them are repeated for each element of the
      // arrays of instances.
      bool unroll = algorithmSectionOp
                        ->walk([](WhenOp) { return mlir::WalkResult::interrupt(); })
                        .wasInterrupted();

      int64_t numElements = 1;

      if (unroll) {
        for (int64_t size : *sizes) {
          numElements *= size;
        }
      } else {
        for (int64_t size : *sizes) {
          auto [loopOp, index] = createLoop<ForOp>(sectionBuilder, loc, size);
          indices.push_back(index);
        }
      }

      for (int64_t element = 0; element < numElements; ++element) {
        if (unroll) {
          indices.assign(sizes->size(), nullptr);
          int64_t remaining = element;

          for (size_t i = sizes->size(); i > 0; --i) {
            indices[i - 1] = sectionBuilder.create<ConstantOp>(
                loc, IntegerAttr::get(modelOp.getContext(),
                                      remaining % (*sizes)[i - 1]));

            remaining /= (*sizes)[i - 1];
          }
        }

        mlir::IRMapping mapping;

        for (mlir::Operation &op : body.front()) {
          if (mlir::failed(
                  emitOp(&op, scope, indices, sectionBuilder, mapping))) {
            return mlir::failure();
          }
        }
      }
    } else {
      return mlir::emitError(loc) << "external clauses are allowed only "
                                     "within functions";
    }
  }

  for (LoopNest &nest : loopNests) {
    if (nest.outermostLoop && nest.body->empty()) {
      nest.outermostLoop->erase();
    }
  }

  for (Element *element : instance->classScope->elements) {
    if (!element->isComponent()) {
      continue;
    }

    Component *component = getComponent(element);

    if (!component) {
      return mlir::failure();
    }

    if (component->isPrimitive() || component->isOuter()) {
      continue;
    }

    auto enabled = isEnabled(component);

    if (mlir::failed(enabled)) {
      return mlir::failure();
    }

    if (*enabled && mlir::failed(emitSections(component, container))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult Instantiator::emitEquation(mlir::Operation *op,
                                               Scope *scope,
                                               const IndexValues &indices,
                                               mlir::OpBuilder &builder,
                                               mlir::IRMapping &mapping) {
  if (auto connectOp = mlir::dyn_cast<ConnectOp>(op)) {
    return emitConnect(connectOp, scope, indices, builder, mapping);
  }

  if (auto forEquationOp = mlir::dyn_cast<ForEquationOp>(op)) {
    return emitForEquation(forEquationOp, scope, indices, builder, mapping);
  }

  if (auto ifEquationOp = mlir::dyn_cast<IfEquationOp>(op);
      ifEquationOp && containsConnect(ifEquationOp)) {
    return emitIfEquation(ifEquationOp, scope, indices, builder, mapping);
  }

  return emitOp(op, scope, indices, builder, mapping);
}

mlir::LogicalResult Instantiator::emitConnect(ConnectOp op, Scope *scope,
                                              const IndexValues &indices,
                                              mlir::OpBuilder &builder,
                                              mlir::IRMapping &mapping) {
  auto yieldOp = mlir::cast<YieldOp>(op.getBodyRegion().front().getTerminator());
  bool outside[2] = {false, false};

  // Connections removed by selective model extension.
  std::string lhsText = getReferenceText(yieldOp.getValues()[0]);
  std::string rhsText = getReferenceText(yieldOp.getValues()[1]);

  for (Scope *current = scope; current; current = current->derived) {
    if (current->deselectedConnections.count({lhsText, rhsText}) ||
        current->deselectedConnections.count({rhsText, lhsText})) {
      return mlir::success();
    }
  }

  // Determine whether the connectors are inside or outside,
  // and whether the connection involves disabled
  // conditional components or deselected ones,
  // in which case it is removed.
  for (size_t side = 0; side < 2; ++side) {
    llvm::SmallVector<llvm::StringRef> names;
    mlir::Value current = yieldOp.getValues()[side];

    while (mlir::Operation *definingOp = current.getDefiningOp()) {
      if (auto memberOp = mlir::dyn_cast<MemberOp>(definingOp)) {
        names.push_back(memberOp.getMember());
        current = memberOp.getObject();
      } else if (auto subscriptOp = mlir::dyn_cast<SubscriptOp>(definingOp)) {
        current = subscriptOp.getArray();
      } else if (auto refOp = mlir::dyn_cast<RefOp>(definingOp);
                 refOp && refOp.hasSimpleName()) {
        names.push_back(refOp.getSimpleName());
        break;
      } else {
        return op.emitError("invalid connector reference");
      }
    }

    std::reverse(names.begin(), names.end());

    if (names.empty()) {
      return op.emitError("invalid connector reference");
    }

    Element *element = lookupSimple(scope, names.front());

    if (!element || !element->isComponent()) {
      if (scope->deselected.contains(names.front())) {
        return mlir::success();
      }

      return op.emitError("unknown connector '") << names.front() << "'";
    }

    Component *component = getComponent(element);

    if (!component) {
      return mlir::failure();
    }

    // Components removed by selective model extension in derived classes
    // are not part of the instance.
    if (Component *parent = component->parent;
        parent && parent->classScope &&
        !lookupMember(parent->classScope, component->name)) {
      return mlir::success();
    }

    outside[side] = isConnector(component);

    for (size_t i = 0; i < names.size(); ++i) {
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
        return mlir::success();
      }

      if (i + 1 == names.size()) {
        break;
      }

      Element *member = lookupMember(component->classScope, names[i + 1]);

      if (!member || !member->isComponent()) {
        if (component->classScope->deselected.contains(names[i + 1])) {
          return mlir::success();
        }

        return op.emitError("unknown connector '") << names[i + 1] << "'";
      }

      component = getComponent(member);

      if (!component) {
        return mlir::failure();
      }
    }
  }

  auto emitted = emitOp(op, scope, indices, builder, mapping);

  if (mlir::failed(emitted)) {
    return mlir::failure();
  }

  auto connectOp = mlir::cast<ConnectOp>(*emitted);
  connectOp.setLhsOutside(outside[0]);
  connectOp.setRhsOutside(outside[1]);

  // Replace the parameter expressions used by the subscripts with their
  // values, so that the connections can be expanded without knowing the
  // values of the parameters.
  auto connectYieldOp =
      mlir::cast<YieldOp>(connectOp.getBodyRegion().front().getTerminator());

  llvm::DenseSet<mlir::Operation *> connectorChains;

  for (mlir::Value value : connectYieldOp.getValues()) {
    while (mlir::Operation *definingOp = value.getDefiningOp()) {
      connectorChains.insert(definingOp);

      if (auto memberOp = mlir::dyn_cast<MemberOp>(definingOp)) {
        value = memberOp.getObject();
      } else if (auto subscriptOp = mlir::dyn_cast<SubscriptOp>(definingOp)) {
        value = subscriptOp.getArray();
      } else {
        break;
      }
    }
  }

  for (mlir::Operation &op :
       llvm::make_early_inc_range(connectOp.getBodyRegion().front())) {
    if (connectorChains.contains(&op) || op.getNumResults() != 1 ||
        mlir::isa<ConstantOp, YieldOp>(op) ||
        mlir::isa<ObjectType>(mlir::getElementTypeOrSelf(op.getResult(0)))) {
      continue;
    }

    if (auto value = mlir::dyn_cast_or_null<mlir::TypedAttr>(
            evaluateEmitted(op.getResult(0)))) {
      mlir::OpBuilder constantBuilder(&op);
      op.getResult(0).replaceAllUsesWith(
          constantBuilder.create<ConstantOp>(op.getLoc(), value).getResult());
    }
  }

  for (mlir::Operation &op : llvm::make_early_inc_range(
           llvm::reverse(connectOp.getBodyRegion().front()))) {
    if (op.use_empty() && mlir::isMemoryEffectFree(&op) &&
        !mlir::isa<YieldOp>(op)) {
      op.erase();
    }
  }

  return mlir::success();
}

mlir::LogicalResult Instantiator::emitForEquation(ForEquationOp op,
                                                  Scope *scope,
                                                  const IndexValues &indices,
                                                  mlir::OpBuilder &builder,
                                                  mlir::IRMapping &mapping) {
  if (op.hasImplicitRange()) {
    return op.emitError("implicit iteration ranges are not supported yet");
  }

  bool hasConnects = containsConnect(op);
  mlir::Value range = getYieldedValue(op.getRangeRegion());
  mlir::Attribute rangeValue = evaluate(range, scope, indices.size(), mapping);

  // The ranges of the for-equations containing connect-equations must be
  // parameter expressions.
  if (hasConnects && !rangeValue) {
    return op.emitError("the range of a for-equation containing "
                        "connect-equations must be evaluable");
  }

  llvm::SmallVector<mlir::Attribute> values = getConstantElements(rangeValue);
  mlir::Location loc = op.getLoc();

  if (hasConnects && containsConditionalConnect(op)) {
    // The conditions of the if-equations may depend on the iteration
    // variable, and thus the loop is unrolled.
    for (mlir::Attribute value : values) {
      mlir::IRMapping iterationMapping = mapping;

      iterationMapping.map(
          op.getInductionVariable(),
          builder.create<ConstantOp>(loc, mlir::cast<mlir::TypedAttr>(value))
              .getResult());

      for (mlir::Operation &nestedOp : op.getBodyRegion().front()) {
        if (mlir::failed(emitEquation(&nestedOp, scope, indices, builder,
                                      iterationMapping))) {
          return mlir::failure();
        }
      }
    }

    return mlir::success();
  }

  auto forEquationOp = builder.create<ForEquationOp>(loc);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&forEquationOp.getRangeRegion());

  if (rangeValue) {
    // Ranges of consecutive Integer values are kept compact.
    std::optional<int64_t> first =
        values.empty() ? std::nullopt : getIntegerValue(values.front());

    bool isConsecutive =
        first && mlir::isa<IntegerAttr>(values.front()) &&
        llvm::all_of(llvm::enumerate(values), [&](const auto &it) {
          auto value = getIntegerValue(it.value());
          return value && *value == *first + static_cast<int64_t>(it.index());
        });

    mlir::Value rangeResult;

    if (isConsecutive) {
      rangeResult = createIntegerRange(builder, loc, *first,
                                       *first + values.size() - 1);
    } else {
      rangeResult =
          builder
              .create<ConstantOp>(loc, mlir::cast<mlir::TypedAttr>(rangeValue))
              .getResult();
    }

    builder.create<YieldOp>(loc, rangeResult);
  } else {
    mlir::IRMapping rangeMapping = mapping;
    auto rangeResult = emitValue(range, scope, indices, builder, rangeMapping);

    if (mlir::failed(rangeResult)) {
      return mlir::failure();
    }

    builder.create<YieldOp>(loc, *rangeResult);
  }

  mlir::BlockArgument inductionVariable = op.getInductionVariable();

  mlir::Block *body =
      builder.createBlock(&forEquationOp.getBodyRegion(), {},
                          inductionVariable.getType(), loc);

  mapping.map(inductionVariable, body->getArgument(0));
  builder.setInsertionPointToStart(body);

  for (mlir::Operation &nestedOp : op.getBodyRegion().front()) {
    if (mlir::failed(
            emitEquation(&nestedOp, scope, indices, builder, mapping))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult Instantiator::emitIfEquation(IfEquationOp op, Scope *scope,
                                                 const IndexValues &indices,
                                                 mlir::OpBuilder &builder,
                                                 mlir::IRMapping &mapping) {
  // The conditions of the if-equations containing connect-equations must be
  // parameter expressions.
  std::optional<bool> condition =
      getBooleanValue(evaluate(getYieldedValue(op.getConditionRegion()), scope,
                               indices.size(), mapping));

  if (!condition) {
    return op.emitError("the condition of an if-equation containing "
                        "connect-equations must be evaluable");
  }

  mlir::Region &region = *condition ? op.getThenRegion() : op.getElseRegion();

  if (region.empty()) {
    return mlir::success();
  }

  for (mlir::Operation &nestedOp : region.front()) {
    if (mlir::failed(emitEquation(&nestedOp, scope, indices, builder, mapping))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

//===---------------------------------------------------------------------===//
// Functions
//===---------------------------------------------------------------------===//

mlir::FailureOr<mlir::FlatSymbolRefAttr>
Instantiator::getFunction(Element *element, mlir::Location loc) {
  mlir::MLIRContext *context = module.getContext();
  Scope *classScope = getClassScope(element);

  if (mlir::failed(expand(classScope))) {
    return mlir::failure();
  }

  // Functions declared within unmodified packages are shared by all the
  // elements referring to them.
  bool isUnmodified = true;

  for (Scope *scope = classScope; scope; scope = scope->parent) {
    if (!scope->modifier.empty() || scope->replaced || scope->instance) {
      isUnmodified = false;
      break;
    }
  }

  Scope *key = classScope;

  if (isUnmodified) {
    auto it = canonicalFunctionScopes.try_emplace(element->op, classScope);
    key = it.first->second;
  }

  if (auto it = functions.find(key); it != functions.end()) {
    return it->second;
  }

  std::string name = classScope->path;

  for (size_t counter = 1; functionNames.contains(name); ++counter) {
    name = classScope->path + "_" + std::to_string(counter);
  }

  functionNames.insert(name);
  auto symbol = mlir::FlatSymbolRefAttr::get(context, name);
  functions[key] = symbol;

  auto classOp = mlir::cast<ClassOp>(element->op);
  mlir::OpBuilder functionBuilder(context);
  functionBuilder.setInsertionPointToEnd(module.getBody());

  auto functionOp = functionBuilder.create<FunctionOp>(classOp.getLoc(), name);
  functionOp.setIsImpure(classOp.getIsImpure());
  functionOp.setDescriptionAttr(classOp.getDescriptionAttr());
  functionOp.setAnnotationAttr(classOp.getAnnotationAttr());

  // The function is instantiated as the root of its own instance tree, so
  // that the references to its components become local references.
  auto &root = components.emplace_back(std::make_unique<Component>());
  root->name = name;
  root->classScope = instantiateClass(element, {}, root.get());

  if (mlir::failed(expand(root->classScope))) {
    return mlir::failure();
  }

  mlir::OpBuilder bodyBuilder(context);
  bodyBuilder.setInsertionPointToStart(functionOp.getBody());

  for (Element *member : root->classScope->elements) {
    if (!member->isComponent()) {
      continue;
    }

    Component *component = getComponent(member);

    if (!component) {
      return mlir::failure();
    }

    ComponentOp componentOp = component->getDeclaration();

    if (!component->isPrimitive()) {
      return mlir::emitError(componentOp.getLoc())
             << "structured components of functions are not supported yet ('"
             << name << "." << component->name << "')";
    }

    auto type = getType(component);

    if (mlir::failed(type)) {
      return mlir::failure();
    }

    auto variableOp = bodyBuilder.create<VariableOp>(
        componentOp.getLoc(), component->name, *type);

    if (component->variability) {
      variableOp.setVariability(*component->variability);
    }

    if (component->causality) {
      variableOp.setCausality(*component->causality);
    }

    variableOp.setIsProtected(component->isProtected);
    variableOp.setDescriptionAttr(componentOp.getDescriptionAttr());

    // Dimensions depending on the inputs.
    llvm::ArrayRef<int64_t> shape = modelica::getShape(*type);

    if (mlir::ShapedType::isDynamicShape(shape) &&
        component->classScope->shortClassExtends.empty()) {
      llvm::SmallVector<mlir::Value> sizes;
      mlir::Block *dimsBlock = &variableOp.getDimensionsRegion().emplaceBlock();
      mlir::OpBuilder dimsBuilder(context);
      dimsBuilder.setInsertionPointToStart(dimsBlock);
      mlir::IRMapping mapping;
      bool complete = true;
      size_t valuePos = 0;

      for (auto [dimension, size] :
           llvm::zip(componentOp.getDimensions(), shape)) {
        auto dimensionAttr = mlir::cast<DimensionAttr>(dimension);
        bool isExpression =
            dimensionAttr.getKind() == DimensionKind::expression;

        mlir::Value value =
            isExpression ? getYieldedValue(componentOp.getDimensionsRegion(),
                                           valuePos++)
                         : mlir::Value();

        if (!mlir::ShapedType::isDynamic(size)) {
          continue;
        }

        if (!isExpression) {
          complete = false;
          break;
        }

        auto emitted = emitValue(value, member->declScope, {}, dimsBuilder,
                                 mapping);

        if (mlir::failed(emitted)) {
          return mlir::failure();
        }

        sizes.push_back(*emitted);
      }

      if (complete) {
        dimsBuilder.create<YieldOp>(componentOp.getLoc(), sizes);
      } else {
        variableOp.getDimensionsRegion().getBlocks().clear();
      }
    }

    if (mlir::failed(emitVariableModification(component, variableOp))) {
      return mlir::failure();
    }
  }

  for (auto [sectionOp, scope] : root->classScope->sections) {
    mlir::IRMapping mapping;

    if (auto algorithmSectionOp = mlir::dyn_cast<AlgorithmSectionOp>(sectionOp)) {
      if (algorithmSectionOp.getBodyRegion().empty()) {
        continue;
      }

      auto targetOp = bodyBuilder.create<AlgorithmSectionOp>(
          algorithmSectionOp.getLoc(), algorithmSectionOp.getIsInitial());

      mlir::OpBuilder sectionBuilder(context);
      sectionBuilder.setInsertionPointToStart(
          sectionBuilder.createBlock(&targetOp.getBodyRegion()));

      for (mlir::Operation &op : algorithmSectionOp.getBodyRegion().front()) {
        if (mlir::failed(emitOp(&op, scope, {}, sectionBuilder, mapping))) {
          return mlir::failure();
        }
      }

      continue;
    }

    if (mlir::isa<ExternalOp>(sectionOp)) {
      if (mlir::failed(emitOp(sectionOp, scope, {}, bodyBuilder, mapping))) {
        return mlir::failure();
      }

      continue;
    }

    return mlir::emitError(sectionOp->getLoc())
           << "functions can't have equations";
  }

  // Functions instantiated within different contexts may still be equal, in
  // which case the first one is used.
  for (FunctionOp existingOp : module.getOps<FunctionOp>()) {
    if (existingOp == functionOp ||
        !existingOp.getSymName().starts_with(classScope->path) ||
        existingOp.getIsImpure() != functionOp.getIsImpure() ||
        existingOp.getAnnotationAttr() != functionOp.getAnnotationAttr()) {
      continue;
    }

    if (mlir::OperationEquivalence::isRegionEquivalentTo(
            &existingOp.getBodyRegion(), &functionOp.getBodyRegion(),
            mlir::OperationEquivalence::IgnoreLocations)) {
      functionOp.erase();
      functionNames.erase(name);
      auto existingSymbol = mlir::FlatSymbolRefAttr::get(existingOp);
      functions[key] = existingSymbol;
      return existingSymbol;
    }
  }

  return symbol;
}
} // namespace mlir::modelica::instantiation
