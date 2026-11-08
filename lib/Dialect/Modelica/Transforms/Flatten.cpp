#include "marco/Dialect/Modelica/Transforms/Flatten.h"
#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::modelica {
#define GEN_PASS_DEF_FLATTENPASS
#include "marco/Dialect/Modelica/Transforms/Passes.h.inc"
} // namespace mlir::modelica

using namespace ::mlir::modelica;

namespace {
/// Primitive variable of the instance tree.
struct Variable {
  mutable VariableOp op;
  llvm::SmallVector<mlir::Operation *> path;
  std::string name;

  /// Dimensions of the enclosing instances.
  llvm::SmallVector<int64_t> prefixShape;
};

class ModelFlattener {
public:
  explicit ModelFlattener(ModelOp modelOp) : modelOp(modelOp) {}

  mlir::LogicalResult run();

private:
  void collectVariables(mlir::Operation *op,
                        llvm::SmallVectorImpl<mlir::Operation *> &path);

  mlir::LogicalResult createFlatVariable(const Variable &variable,
                                         mlir::OpBuilder &builder);

  /// Copy a value of the modification region of a variable, extending it to
  /// the elements of the enclosing arrays of instances.
  mlir::FailureOr<mlir::Value> copyValue(const Variable &variable,
                                         mlir::Value value, bool isEach,
                                         mlir::OpBuilder &builder);

  mlir::LogicalResult flattenReferences();

private:
  ModelOp modelOp;
  llvm::SmallVector<Variable> variables;
};

void ModelFlattener::collectVariables(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::Operation *> &path) {
  for (mlir::Operation &nested : op->getRegion(0).front()) {
    if (auto instanceOp = mlir::dyn_cast<InstanceOp>(nested)) {
      path.push_back(instanceOp);
      collectVariables(instanceOp, path);
      path.pop_back();
      continue;
    }

    if (auto variableOp = mlir::dyn_cast<VariableOp>(nested)) {
      Variable &variable = variables.emplace_back();
      variable.op = variableOp;
      variable.path.assign(path.begin(), path.end());

      for (mlir::Operation *pathOp : path) {
        auto instanceOp = mlir::cast<InstanceOp>(pathOp);
        variable.name += instanceOp.getSymName().str() + ".";
        llvm::append_range(variable.prefixShape, instanceOp.getShape());
      }

      variable.name += variableOp.getSymName().str();
    }
  }
}

/// Get whether a value computed within a region depends on the arguments of
/// the region.
bool dependsOnArguments(mlir::Value value, mlir::Block *block) {
  llvm::SmallVector<mlir::Value> worklist = {value};
  llvm::DenseSet<mlir::Value> visited;

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();

    if (!visited.insert(current).second) {
      continue;
    }

    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      if (argument.getOwner() == block) {
        return true;
      }

      continue;
    }

    mlir::Operation *op = current.getDefiningOp();
    llvm::append_range(worklist, op->getOperands());

    llvm::SetVector<mlir::Value> usedValues;

    for (mlir::Region &region : op->getRegions()) {
      mlir::getUsedValuesDefinedAbove(region, usedValues);
    }

    llvm::append_range(worklist, usedValues);
  }

  return false;
}

/// Clone the operations computing a value, together with the ones they
/// depend on.
mlir::Value cloneValue(mlir::Value value, mlir::OpBuilder &builder,
                       mlir::IRMapping &mapping) {
  if (mlir::Value mapped = mapping.lookupOrNull(value)) {
    return mapped;
  }

  mlir::Operation *op = value.getDefiningOp();
  llvm::SetVector<mlir::Value> usedValues;
  usedValues.insert(op->getOperands().begin(), op->getOperands().end());

  for (mlir::Region &region : op->getRegions()) {
    mlir::getUsedValuesDefinedAbove(region, usedValues);
  }

  for (mlir::Value used : usedValues) {
    cloneValue(used, builder, mapping);
  }

  builder.clone(*op, mapping);
  return mapping.lookup(value);
}

mlir::FailureOr<mlir::Value>
ModelFlattener::copyValue(const Variable &variable, mlir::Value value,
                          bool isEach, mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Location loc = value.getLoc();
  mlir::Block *block = &variable.op.getModificationRegion().front();
  llvm::ArrayRef<int64_t> shape = getShape(variable.op.getType());

  // Values given with 'each' apply to all the elements of the variable.
  auto expandEach = [&](mlir::Value element) -> mlir::Value {
    if (!isEach || shape.empty()) {
      return element;
    }

    llvm::SmallVector<mlir::Value> sizes;

    for (int64_t size : shape) {
      sizes.push_back(
          builder.create<ConstantOp>(loc, IntegerAttr::get(context, size)));
    }

    return builder.create<FillOp>(loc, element, sizes);
  };

  if (!dependsOnArguments(value, block)) {
    mlir::IRMapping mapping;
    mlir::Value result = cloneValue(value, builder, mapping);

    if (isEach || variable.prefixShape.empty()) {
      return result;
    }

    // The same value applies to all the elements of the enclosing arrays.
    llvm::SmallVector<mlir::Value> sizes;

    for (int64_t size : variable.prefixShape) {
      sizes.push_back(
          builder.create<ConstantOp>(loc, IntegerAttr::get(context, size)));
    }

    return builder.create<FillOp>(loc, result, sizes).getResult();
  }

  // Array constructor iterating over the indices of the enclosing arrays.
  // The last iterator varies along the outermost dimension,
  // and thus the iterators are listed in reverse
  // order.
  llvm::SmallVector<mlir::Value> ranges;

  for (int64_t size : llvm::reverse(variable.prefixShape)) {
    auto from = builder.create<ConstantOp>(loc, IntegerAttr::get(context, 0));
    auto to =
        builder.create<ConstantOp>(loc, IntegerAttr::get(context, size - 1));

    ranges.push_back(builder.create<RangeOp>(
        loc, mlir::RankedTensorType::get({size}, IntegerType::get(context)),
        from, to, mlir::Value()));
  }

  llvm::SmallVector<int64_t> resultShape(variable.prefixShape);
  llvm::SmallVector<int64_t> elementShape;

  if (isEach) {
    elementShape.assign(shape.begin(), shape.end());
  } else {
    elementShape.append(getShape(value.getType()).begin(),
                        getShape(value.getType()).end());
  }

  resultShape.append(elementShape.begin(), elementShape.end());

  auto arrayForOp = builder.create<ArrayForOp>(
      loc,
      mlir::RankedTensorType::get(resultShape,
                                  mlir::getElementTypeOrSelf(value.getType())),
      ranges);

  mlir::OpBuilder::InsertionGuard guard(builder);

  llvm::SmallVector<mlir::Type> argTypes(ranges.size(),
                                         IntegerType::get(context));
  llvm::SmallVector<mlir::Location> argLocs(ranges.size(), loc);

  mlir::Block *body =
      builder.createBlock(&arrayForOp.getBody(), {}, argTypes, argLocs);

  mlir::IRMapping mapping;

  for (auto [i, argument] : llvm::enumerate(block->getArguments())) {
    mapping.map(argument, body->getArgument(ranges.size() - 1 - i));
  }

  mlir::Value element = expandEach(cloneValue(value, builder, mapping));
  builder.create<YieldOp>(loc, element);
  return arrayForOp.getResult();
}

mlir::LogicalResult
ModelFlattener::createFlatVariable(const Variable &variable,
                                   mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();
  VariableOp variableOp = variable.op;

  llvm::SmallVector<int64_t> shape(variable.prefixShape);
  llvm::append_range(shape, getShape(variableOp.getType()));

  auto flatOp = builder.create<VariableOp>(
      variableOp.getLoc(), variable.name,
      getValueType(shape, mlir::getElementTypeOrSelf(variableOp.getType())));

  flatOp.setVariabilityAttr(variableOp.getVariabilityAttr());
  flatOp.setIsProtected(variableOp.getIsProtected());
  flatOp.setDescriptionAttr(variableOp.getDescriptionAttr());

  // The inputs and outputs of the model are the top-level ones, and the
  // ones of the top-level connectors.
  bool isInterface =
      variable.path.empty() ||
      (variable.path.size() == 1 &&
       (mlir::cast<InstanceOp>(variable.path.front()).getKind() ==
            ClassKind::connector ||
        mlir::cast<InstanceOp>(variable.path.front()).getKind() ==
            ClassKind::expandable_connector));

  if (isInterface) {
    flatOp.setCausalityAttr(variableOp.getCausalityAttr());
  }

  mlir::Region &region = variableOp.getModificationRegion();

  if (region.empty()) {
    return mlir::success();
  }

  if (variable.prefixShape.empty()) {
    flatOp.getModificationRegion().takeBody(region);
    return mlir::success();
  }

  mlir::OpBuilder regionBuilder(context);
  regionBuilder.setInsertionPointToStart(
      &flatOp.getModificationRegion().emplaceBlock());

  for (mlir::Operation &op : region.front()) {
    if (auto bindOp = mlir::dyn_cast<BindOp>(op)) {
      auto value = copyValue(variable, bindOp.getValue(), false, regionBuilder);

      if (mlir::failed(value)) {
        return mlir::failure();
      }

      regionBuilder.create<BindOp>(bindOp.getLoc(), *value, false);
      continue;
    }

    auto modifyOp = mlir::dyn_cast<ModifyOp>(op);

    if (!modifyOp || modifyOp.getModificationRegion().empty()) {
      continue;
    }

    mlir::Value nestedValue;

    for (auto nestedBindOp :
         modifyOp.getModificationRegion().front().getOps<BindOp>()) {
      nestedValue = nestedBindOp.getValue();
    }

    if (!nestedValue) {
      continue;
    }

    // Scalar values not depending on the element apply to each element.
    bool keepEach = (modifyOp.getIsEach() ||
                     getShape(nestedValue.getType()).empty()) &&
                    !dependsOnArguments(nestedValue, &region.front());

    auto flatModifyOp = regionBuilder.create<ModifyOp>(
        modifyOp.getLoc(), modifyOp.getNameAttr(), keepEach, false,
        mlir::StringAttr());

    mlir::OpBuilder nestedBuilder(context);
    nestedBuilder.setInsertionPointToStart(
        &flatModifyOp.getModificationRegion().emplaceBlock());

    mlir::FailureOr<mlir::Value> value = nestedValue;

    if (keepEach) {
      mlir::IRMapping mapping;
      value = cloneValue(nestedValue, nestedBuilder, mapping);
    } else {
      value = copyValue(variable, nestedValue, modifyOp.getIsEach(),
                        nestedBuilder);
    }

    if (mlir::failed(value)) {
      return mlir::failure();
    }

    nestedBuilder.create<BindOp>(modifyOp.getLoc(), *value, false);
  }

  return mlir::success();
}

mlir::LogicalResult ModelFlattener::flattenReferences() {
  // References to primitive variables through instances, that is member
  // operations producing values of simple types.
  llvm::SmallVector<MemberOp> memberOps;

  modelOp.walk([&](MemberOp memberOp) {
    if (!mlir::isa<ObjectType>(mlir::getElementTypeOrSelf(memberOp.getType()))) {
      memberOps.push_back(memberOp);
    }
  });

  for (MemberOp memberOp : memberOps) {
    // Collect the path, from the model to the variable.
    llvm::SmallVector<mlir::Operation *> chain;
    mlir::Value current = memberOp.getResult();

    while (mlir::Operation *definingOp = current.getDefiningOp()) {
      chain.push_back(definingOp);

      if (auto nestedMemberOp = mlir::dyn_cast<MemberOp>(definingOp)) {
        current = nestedMemberOp.getObject();
      } else if (auto subscriptOp = mlir::dyn_cast<SubscriptOp>(definingOp)) {
        current = subscriptOp.getArray();
      } else {
        break;
      }
    }

    std::reverse(chain.begin(), chain.end());
    auto refOp = mlir::dyn_cast<RefOp>(chain.front());

    if (!refOp) {
      // Member of a record value.
      return memberOp.emitError("records are not supported yet");
    }

    if (!refOp.hasSimpleName()) {
      return memberOp.emitError("invalid reference");
    }

    std::string name;
    llvm::SmallVector<mlir::Value> indices;
    llvm::SmallVector<bool> colons;
    mlir::Operation *container = modelOp;
    mlir::Operation *level = nullptr;
    size_t levelRank = 0;
    size_t levelSubscripts = 0;

    auto closeLevel = [&]() {
      for (; levelSubscripts < levelRank; ++levelSubscripts) {
        colons.push_back(true);
      }
    };

    for (mlir::Operation *op : chain) {
      if (auto subscriptOp = mlir::dyn_cast<SubscriptOp>(op)) {
        auto index = subscriptOp.getIndices().begin();

        for (bool isColon : subscriptOp.getColons()) {
          colons.push_back(isColon);

          if (!isColon) {
            indices.push_back(*index++);
          }

          ++levelSubscripts;
        }

        continue;
      }

      closeLevel();

      llvm::StringRef levelName =
          mlir::isa<RefOp>(op) ? mlir::cast<RefOp>(op).getSimpleName()
                               : mlir::cast<MemberOp>(op).getMember();

      level = mlir::SymbolTable::lookupSymbolIn(container, levelName);

      if (!level) {
        return op->emitError("unknown component '") << levelName << "'";
      }

      name += (name.empty() ? "" : ".") + levelName.str();
      levelSubscripts = 0;

      if (auto instanceOp = mlir::dyn_cast<InstanceOp>(level)) {
        levelRank = instanceOp.getShape().size();
        container = instanceOp;
      } else {
        // The subscripts of the variable itself follow the member operation.
        levelRank = 0;
      }
    }

    closeLevel();

    // Trailing colons are implicit.
    while (!colons.empty() && colons.back()) {
      colons.pop_back();
    }

    auto variableOp = mlir::dyn_cast<VariableOp>(level);

    if (!variableOp) {
      return memberOp.emitError("invalid reference");
    }

    mlir::OpBuilder builder(memberOp);
    auto flatVariable = mlir::cast<VariableOp>(
        mlir::SymbolTable::lookupSymbolIn(modelOp, name));

    mlir::Value result =
        builder.create<RefOp>(memberOp.getLoc(), flatVariable.getType(), name);

    if (!colons.empty()) {
      result = builder.create<SubscriptOp>(memberOp.getLoc(), memberOp.getType(),
                                           result, indices, colons);
    }

    memberOp.getResult().replaceAllUsesWith(result);
    memberOp.erase();
  }

  // Erase the references to the instances, which are now unused.
  bool changed = true;

  while (changed) {
    changed = false;

    modelOp.walk([&](mlir::Operation *op) {
      if (!mlir::isa<RefOp, MemberOp, SubscriptOp>(op) || !op->use_empty()) {
        return;
      }

      if (mlir::isa<ObjectType>(mlir::getElementTypeOrSelf(op->getResult(0)))) {
        op->erase();
        changed = true;
      }
    });
  }

  return mlir::success();
}

mlir::LogicalResult ModelFlattener::run() {
  if (modelOp.walk([](ConnectOp) { return mlir::WalkResult::interrupt(); })
          .wasInterrupted()) {
    return modelOp.emitError("the connect-equations must be expanded before "
                             "flattening");
  }

  llvm::SmallVector<mlir::Operation *> path;
  collectVariables(modelOp, path);

  mlir::OpBuilder builder(modelOp.getContext());
  builder.setInsertionPointToStart(modelOp.getBody());

  // The flattened variables are created before removing the instances, as
  // their modifications may refer to the variables of other instances.

  for (const Variable &variable : variables) {
    if (variable.path.empty()) {
      continue;
    }

    if (mlir::failed(createFlatVariable(variable, builder))) {
      return mlir::failure();
    }
  }

  if (mlir::failed(flattenReferences())) {
    return mlir::failure();
  }

  for (auto instanceOp :
       llvm::make_early_inc_range(modelOp.getBody()->getOps<InstanceOp>())) {
    instanceOp.erase();
  }

  // Values of structured types, such as record values, are not flattened.
  mlir::WalkResult result = modelOp.walk([&](mlir::Operation *op) {
    for (mlir::Value value : op->getResults()) {
      if (mlir::isa<ObjectType>(mlir::getElementTypeOrSelf(value.getType()))) {
        op->emitError("records are not supported yet");
        return mlir::WalkResult::interrupt();
      }
    }

    return mlir::WalkResult::advance();
  });

  return mlir::failure(result.wasInterrupted());
}

class FlattenPass final
    : public mlir::modelica::impl::FlattenPassBase<FlattenPass> {
public:
  using FlattenPassBase<FlattenPass>::FlattenPassBase;

  void runOnOperation() override {
    for (ModelOp modelOp : getOperation().getOps<ModelOp>()) {
      ModelFlattener flattener(modelOp);

      if (mlir::failed(flattener.run())) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace
