#include "marco/Dialect/BaseModelica/Transforms/RecordInlining.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_RECORDINLININGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
std::string getComposedComponentName(llvm::StringRef record,
                                     llvm::StringRef component) {
  return record.str() + "." + component.str();
}

std::string getComposedComponentName(VariableOp record, VariableOp component) {
  return getComposedComponentName(record.getSymName(), component.getSymName());
}
} // namespace

namespace {
template <typename Op>
class RecordInliningPattern : public mlir::OpRewritePattern<Op> {
public:
  RecordInliningPattern(mlir::MLIRContext *context, mlir::ModuleOp moduleOp,
                        mlir::SymbolTableCollection &symbolTableCollection)
      : mlir::OpRewritePattern<Op>(context), moduleOp(moduleOp),
        symbolTable(&symbolTableCollection) {}

protected:
  mlir::SymbolTableCollection &getSymbolTableCollection() const {
    return *symbolTable;
  }

  RecordOp getRecordOp(RecordType recordType) const {
    return mlir::cast<RecordOp>(
        recordType.getRecordOp(getSymbolTableCollection(), moduleOp));
  }

  bool isRecordBased(mlir::Value value) const {
    return isRecordBased(value.getType());
  }

  bool isRecordBased(mlir::Type type) const {
    if (auto tensorType = mlir::dyn_cast<mlir::TensorType>(type)) {
      return mlir::isa<RecordType>(tensorType.getElementType());
    }

    return mlir::isa<RecordType>(type);
  }

  void mergeShapes(llvm::SmallVectorImpl<int64_t> &result,
                   llvm::ArrayRef<int64_t> parent,
                   llvm::ArrayRef<int64_t> child) const {
    result.clear();
    result.append(parent.begin(), parent.end());
    result.append(child.begin(), child.end());
  }

  mlir::LogicalResult replaceRecordGetters(
      mlir::PatternRewriter &rewriter,
      llvm::function_ref<mlir::Value(mlir::OpBuilder &builder,
                                     mlir::Location loc, llvm::StringRef)>
          replaceFn,
      llvm::SmallVectorImpl<mlir::Operation *> &subscriptions,
      mlir::Value usedValue, mlir::Operation *user) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(user);

    if (mlir::isa<TensorViewOp, TensorExtractOp>(user)) {
      subscriptions.push_back(user);

      for (mlir::Value userResult : user->getResults()) {
        for (mlir::Operation *nestedUser :
             llvm::make_early_inc_range(userResult.getUsers())) {
          if (mlir::failed(replaceRecordGetters(rewriter, replaceFn,
                                                subscriptions, userResult,
                                                nestedUser))) {
            return mlir::failure();
          }
        }
      }

      if (user->use_empty()) {
        rewriter.eraseOp(user);
      }

      subscriptions.pop_back();
      return mlir::success();
    }

    if (auto componentGetOp = mlir::dyn_cast<ComponentGetOp>(user)) {
      mlir::Value replacement = replaceFn(rewriter, componentGetOp.getLoc(),
                                          componentGetOp.getComponentName());

      if (!replacement) {
        return mlir::failure();
      }

      replacement = applySubscriptions(rewriter, replacement, subscriptions);

      if (auto tensorType =
              mlir::dyn_cast<mlir::TensorType>(replacement.getType());
          tensorType && !tensorType.hasRank()) {
        replacement = rewriter.create<TensorExtractOp>(componentGetOp.getLoc(),
                                                       replacement);
      }

      rewriter.replaceOp(componentGetOp, replacement);
      return mlir::success();
    }

    if (auto callOp = mlir::dyn_cast<CallOp>(user)) {
      auto newCallOp =
          unpackCallArg(rewriter, callOp, usedValue, replaceFn, subscriptions);

      rewriter.replaceOp(callOp, newCallOp->getResults());
      return mlir::success();
    }

    return mlir::failure();
  }

  CallOp
  unpackCallArg(mlir::OpBuilder &builder, CallOp callOp, mlir::Value arg,
                llvm::function_ref<mlir::Value(mlir::OpBuilder &,
                                               mlir::Location, llvm::StringRef)>
                    componentGetter,
                llvm::ArrayRef<mlir::Operation *> subscriptions) const {
    llvm::SmallVector<mlir::Value> newArgs;
    llvm::SmallVector<mlir::Attribute> newArgNames;

    for (auto currentArg : llvm::enumerate(callOp.getArgs())) {
      if (currentArg.value() == arg) {
        auto recordType = mlir::cast<RecordType>(currentArg.value().getType());
        RecordOp recordOp = getRecordOp(recordType);

        for (VariableOp component : recordOp.getVariables()) {
          mlir::Value componentValue = componentGetter(
              builder, currentArg.value().getLoc(), component.getSymName());

          componentValue =
              applySubscriptions(builder, componentValue, subscriptions);

          if (auto tensorType =
                  mlir::dyn_cast<mlir::TensorType>(componentValue.getType());
              tensorType && !tensorType.hasRank()) {
            componentValue = builder.create<TensorExtractOp>(callOp.getLoc(),
                                                             componentValue);
          }

          newArgs.push_back(componentValue);

          if (auto argNames = callOp.getArgNames()) {
            auto argName = mlir::cast<mlir::FlatSymbolRefAttr>(
                               (*argNames)[currentArg.index()])
                               .getValue();

            auto composedName = mlir::FlatSymbolRefAttr::get(
                builder.getContext(),
                getComposedComponentName(argName, component.getSymName()));

            newArgNames.push_back(composedName);
          }
        }
      } else {
        newArgs.push_back(currentArg.value());

        if (auto argNames = callOp.getArgNames()) {
          newArgNames.push_back((*argNames)[currentArg.index()]);
        }
      }
    }

    std::optional<mlir::ArrayAttr> argNamesAttr = std::nullopt;

    if (!newArgNames.empty()) {
      argNamesAttr = builder.getArrayAttr(newArgNames);
    }

    return builder.create<CallOp>(callOp.getLoc(), callOp.getCallee(),
                                  callOp.getResultTypes(), newArgs,
                                  argNamesAttr);
  }

  mlir::Value
  applySubscriptions(mlir::OpBuilder &builder, mlir::Value root,
                     llvm::ArrayRef<mlir::Operation *> subscriptions) const {
    mlir::Value result = root;

    for (mlir::Operation *op : subscriptions) {
      if (auto extractOp = mlir::dyn_cast<TensorExtractOp>(op)) {
        int64_t rank = mlir::cast<mlir::TensorType>(result.getType()).getRank();

        auto numOfSubscripts =
            static_cast<int64_t>(extractOp.getIndices().size());

        if (numOfSubscripts == rank) {
          result = builder.create<TensorExtractOp>(extractOp.getLoc(), result,
                                                   extractOp.getIndices());
        } else {
          result = builder.create<TensorViewOp>(extractOp.getLoc(), result,
                                                extractOp.getIndices());
        }
      } else if (auto viewOp = mlir::dyn_cast<TensorViewOp>(op)) {
        result = builder.create<TensorViewOp>(viewOp.getLoc(), result,
                                              viewOp.getSubscriptions());
      }
    }

    return result;
  }

private:
  mlir::ModuleOp moduleOp;
  mlir::SymbolTableCollection *symbolTable;
};

/// Unpack the assignment of a record value into multiple assignments
/// involving the components of the record variable.
class VariableSetOpUnpackPattern : public RecordInliningPattern<VariableSetOp> {
public:
  using RecordInliningPattern<VariableSetOp>::RecordInliningPattern;

  mlir::LogicalResult
  matchAndRewrite(VariableSetOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Type valueType = op.getValue().getType();
    mlir::Type valueBaseType = valueType;

    if (auto tensorType = mlir::dyn_cast<mlir::TensorType>(valueType)) {
      valueBaseType = tensorType.getElementType();
    }

    auto recordType = mlir::dyn_cast<RecordType>(valueBaseType);

    if (!recordType) {
      return mlir::failure();
    }

    auto recordOp = getRecordOp(recordType);

    for (VariableOp recordComponentOp : recordOp.getVariables()) {
      mlir::Value componentValue = rewriter.create<ComponentGetOp>(
          op.getLoc(), recordComponentOp.getVariableType().unwrap(),
          op.getValue(), recordComponentOp.getSymName());

      llvm::SmallVector<mlir::Attribute, 1> newPath;

      newPath.push_back(mlir::FlatSymbolRefAttr::get(op.getVariableAttr()));

      newPath.push_back(
          mlir::FlatSymbolRefAttr::get(recordComponentOp.getSymNameAttr()));

      llvm::SmallVector<mlir::Value> subscripts;
      llvm::SmallVector<int64_t> subscriptsAmounts;

      if (auto tensorType = mlir::dyn_cast<mlir::TensorType>(valueType)) {
        mlir::Value unboundedRange =
            rewriter.create<UnboundedRangeOp>(op.getLoc());

        subscripts.append(tensorType.getRank(), unboundedRange);
        subscriptsAmounts.push_back(tensorType.getRank());
      } else {
        subscriptsAmounts.push_back(0);
      }

      rewriter.create<VariableComponentSetOp>(
          op.getLoc(), rewriter.getArrayAttr(newPath), subscripts,
          rewriter.getI64ArrayAttr(subscriptsAmounts), componentValue);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/// Unpack the assignment of a record component into multiple assignments
/// involving the components of the record variable.
/// Together with the above pattern, this enables the handling of assignments
/// of nested records.
class VariableComponentSetOpUnpackPattern
    : public RecordInliningPattern<VariableComponentSetOp> {
public:
  using RecordInliningPattern<VariableComponentSetOp>::RecordInliningPattern;

  mlir::LogicalResult
  matchAndRewrite(VariableComponentSetOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Type valueType = op.getValue().getType();
    mlir::Type valueBaseType = valueType;

    if (auto tensorType = mlir::dyn_cast<mlir::TensorType>(valueType)) {
      valueBaseType = tensorType.getElementType();
    }

    auto recordType = mlir::dyn_cast<RecordType>(valueBaseType);

    if (!recordType) {
      return mlir::failure();
    }

    auto recordOp = getRecordOp(recordType);
    size_t pathLength = op.getPath().size();

    for (VariableOp recordComponentOp : recordOp.getVariables()) {
      mlir::Value componentValue = rewriter.create<ComponentGetOp>(
          op.getLoc(), recordComponentOp.getVariableType().unwrap(),
          op.getValue(), recordComponentOp.getSymName());

      llvm::SmallVector<mlir::Attribute, 10> newPath;

      for (size_t component = 1; component < pathLength; ++component) {
        newPath.push_back(op.getPath()[component]);
      }

      newPath.push_back(
          mlir::FlatSymbolRefAttr::get(recordComponentOp.getSymNameAttr()));

      llvm::SmallVector<mlir::Value, 10> subscripts;
      llvm::SmallVector<int64_t, 10> subscriptsAmounts;

      for (mlir::Value subscript : op.getSubscriptions()) {
        subscripts.push_back(subscript);
      }

      for (mlir::IntegerAttr subscriptsAmount :
           op.getSubscriptionsAmounts().getAsRange<mlir::IntegerAttr>()) {
        subscriptsAmounts.push_back(subscriptsAmount.getInt());
      }

      if (auto tensorType = mlir::dyn_cast<mlir::TensorType>(valueType)) {
        mlir::Value unboundedRange =
            rewriter.create<UnboundedRangeOp>(op.getLoc());

        subscripts.append(tensorType.getRank(), unboundedRange);
        subscriptsAmounts.push_back(tensorType.getRank());
      } else {
        subscriptsAmounts.push_back(0);
      }

      rewriter.create<VariableComponentSetOp>(
          op.getLoc(), rewriter.getArrayAttr(newPath), subscripts,
          rewriter.getI64ArrayAttr(subscriptsAmounts), componentValue);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class EquationSideOpUnpackPattern
    : public RecordInliningPattern<EquationSideOp> {
public:
  using RecordInliningPattern<EquationSideOp>::RecordInliningPattern;

  mlir::LogicalResult
  matchAndRewrite(EquationSideOp op,
                  mlir::PatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value, 3> newValues;
    bool recordFound = false;

    for (mlir::Value value : op.getValues()) {
      if (auto recordType = mlir::dyn_cast<RecordType>(value.getType())) {
        auto recordOp = getRecordOp(recordType);

        for (VariableOp component : recordOp.getVariables()) {
          auto componentGetOp = rewriter.create<ComponentGetOp>(
              value.getLoc(), component.getVariableType().unwrap(), value,
              component.getSymName());

          newValues.push_back(componentGetOp.getResult());
        }

        recordFound = true;
      } else {
        newValues.push_back(value);
      }
    }

    if (!recordFound) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<EquationSideOp>(op, newValues);
    return mlir::success();
  }
};

struct SymbolUsers {
  llvm::DenseSet<VariableGetOp> variableGetOps;
  llvm::DenseSet<VariableComponentSetOp> variableComponentSetOps;
  llvm::DenseSet<StartOp> startOps;
  llvm::DenseSet<DefaultOp> defaultOps;
  llvm::DenseSet<BindingEquationOp> bindingEquationOps;
};

using VariableUsers = llvm::DenseMap<mlir::StringAttr, SymbolUsers>;
using ClassVariableUsers = llvm::DenseMap<ClassInterface, VariableUsers>;

/// Unpack record variables into their components.
class VariableOpUnpackPattern : public RecordInliningPattern<VariableOp> {
  ClassVariableUsers &symbolUsers;

public:
  VariableOpUnpackPattern(
      mlir::MLIRContext *context, mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection &symbolTableCollection,
      llvm::DenseMap<ClassInterface,
                     llvm::DenseMap<mlir::StringAttr, SymbolUsers>>
          &symbolUsers)
      : RecordInliningPattern(context, moduleOp, symbolTableCollection),
        symbolUsers(symbolUsers) {}

  mlir::LogicalResult
  matchAndRewrite(VariableOp op,
                  mlir::PatternRewriter &rewriter) const override {
    VariableType variableType = op.getVariableType();
    mlir::Type elementType = variableType.getElementType();

    if (!mlir::isa<RecordType>(elementType)) {
      // Not a record or an array of records.
      return mlir::failure();
    }

    // Create a variable for each component and map it for faster lookups.
    auto recordType = mlir::cast<RecordType>(elementType);
    auto recordOp = getRecordOp(recordType);

    llvm::StringMap<VariableOp> componentsMap;

    for (VariableOp component : recordOp.getVariables()) {
      VariableType componentVariableType = component.getVariableType();
      llvm::SmallVector<int64_t, 3> dimensions;

      // Use the shape of the original record variable.
      for (int64_t dimension : variableType.getShape()) {
        dimensions.push_back(dimension);
      }

      // Append the dimensions of the component.
      for (int64_t dimension : componentVariableType.getShape()) {
        dimensions.push_back(dimension);
      }

      // Start from the original variable type in order to keep the
      // modifiers.
      componentVariableType =
          variableType.withShape(dimensions)
              .withType(component.getVariableType().getElementType());

      // Create the variable for the component.
      auto unpackedComponent = rewriter.create<VariableOp>(
          op.getLoc(), getComposedComponentName(op, component),
          componentVariableType);

      componentsMap[component.getSymName()] = unpackedComponent;
    }

    // Replace the uses of the original record.
    auto classOp = op->getParentOfType<ClassInterface>();

    for (StartOp startOp : symbolUsers[classOp][op.getSymNameAttr()].startOps) {
      if (mlir::failed(
              unpackStartOp(rewriter, classOp, startOp, componentsMap))) {
        return mlir::failure();
      }
    }

    for (DefaultOp defaultOp :
         symbolUsers[classOp][op.getSymNameAttr()].defaultOps) {
      if (mlir::failed(
              unpackDefaultOp(rewriter, classOp, defaultOp, componentsMap))) {
        return mlir::failure();
      }
    }

    for (BindingEquationOp bindingEquationOp :
         symbolUsers[classOp][op.getSymNameAttr()].bindingEquationOps) {
      if (mlir::failed(unpackBindingEquationOp(
              rewriter, classOp, bindingEquationOp, componentsMap))) {
        return mlir::failure();
      }
    }

    for (VariableGetOp getOp :
         symbolUsers[classOp][op.getSymNameAttr()].variableGetOps) {
      if (mlir::failed(
              replaceVariableGetOp(rewriter, classOp, getOp, componentsMap))) {
        return mlir::failure();
      }
    }

    for (VariableComponentSetOp setOp :
         symbolUsers[classOp][op.getSymNameAttr()].variableComponentSetOps) {
      if (mlir::failed(replaceVariableComponentSetOp(rewriter, classOp, op,
                                                     setOp, componentsMap))) {
        return mlir::failure();
      }
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  mlir::LogicalResult
  unpackStartOp(mlir::PatternRewriter &rewriter, ClassInterface classOp,
                StartOp op,
                const llvm::StringMap<VariableOp> &componentsMap) const {
    if (auto oldNestedRefs = op.getVariable().getNestedReferences();
        !oldNestedRefs.empty()) {
      // The StartOp already goes through the components of the record.
      std::string newRoot =
          getComposedComponentName(op.getVariable().getRootReference(),
                                   oldNestedRefs.front().getValue());

      op.setVariableAttr(mlir::SymbolRefAttr::get(
          rewriter.getStringAttr(newRoot), oldNestedRefs.drop_front()));

      return mlir::success();
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    auto modelOp = op->getParentOfType<ModelOp>();

    auto variableOp = getSymbolTableCollection().lookupSymbolIn<VariableOp>(
        modelOp, op.getVariable().getRootReference());

    auto recordType =
        mlir::cast<RecordType>(variableOp.getVariableType().getElementType());

    auto recordOp = getRecordOp(recordType);

    for (VariableOp component : recordOp.getVariables()) {
      if (!componentsMap.contains(component.getSymName())) {
        return mlir::failure();
      }

      VariableOp componentVariableOp =
          componentsMap.lookup(component.getSymName());

      // Start from the original operation.
      auto clonedOp = mlir::cast<StartOp>(rewriter.clone(*op.getOperation()));

      clonedOp.setVariableAttr(
          mlir::SymbolRefAttr::get(componentVariableOp.getSymNameAttr()));

      symbolUsers[classOp][componentVariableOp.getSymNameAttr()]
          .startOps.insert(clonedOp);

      // Extract the component from the record.
      auto yieldOp = mlir::cast<YieldOp>(clonedOp.getBody()->getTerminator());

      rewriter.setInsertionPointAfter(yieldOp);

      mlir::Value componentValue = rewriter.create<ComponentGetOp>(
          yieldOp.getLoc(),
          component.getVariableType()
              .withShape(componentVariableOp.getVariableType().getShape())
              .unwrap(),
          yieldOp.getValues()[0], component.getSymName());

      rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, componentValue);
      rewriter.setInsertionPointAfter(clonedOp);
    }

    symbolUsers[classOp][op.getVariableAttr().getRootReference()]
        .startOps.erase(op);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::LogicalResult
  unpackDefaultOp(mlir::PatternRewriter &rewriter, ClassInterface classOp,
                  DefaultOp op,
                  const llvm::StringMap<VariableOp> &componentsMap) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    VariableOp variableOp = op.getVariableOp(getSymbolTableCollection());

    auto recordType =
        mlir::cast<RecordType>(variableOp.getVariableType().getElementType());

    auto recordOp = getRecordOp(recordType);

    for (VariableOp component : recordOp.getVariables()) {
      if (!componentsMap.contains(component.getSymName())) {
        return mlir::failure();
      }

      VariableOp componentVariableOp =
          componentsMap.lookup(component.getSymName());

      // Start from the original operation.
      auto clonedOp = mlir::cast<DefaultOp>(rewriter.clone(*op.getOperation()));

      clonedOp.setVariable(componentVariableOp.getSymName());

      symbolUsers[classOp][componentVariableOp.getSymNameAttr()]
          .defaultOps.insert(clonedOp);

      // Extract the component from the record.
      auto yieldOp = mlir::cast<YieldOp>(clonedOp.getBody()->getTerminator());

      rewriter.setInsertionPointAfter(yieldOp);

      mlir::Value componentValue = rewriter.create<ComponentGetOp>(
          yieldOp.getLoc(),
          component.getVariableType()
              .withShape(componentVariableOp.getVariableType().getShape())
              .unwrap(),
          yieldOp.getValues()[0], component.getSymName());

      rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, componentValue);
      rewriter.setInsertionPointAfter(clonedOp);
    }

    symbolUsers[classOp][op.getVariableAttr()].defaultOps.erase(op);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::LogicalResult unpackBindingEquationOp(
      mlir::PatternRewriter &rewriter, ClassInterface classOp,
      BindingEquationOp op,
      const llvm::StringMap<VariableOp> &componentsMap) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    VariableOp variableOp = op.getVariableOp(getSymbolTableCollection());

    auto recordType =
        mlir::cast<RecordType>(variableOp.getVariableType().getElementType());

    auto recordOp = getRecordOp(recordType);

    for (VariableOp component : recordOp.getVariables()) {
      if (!componentsMap.contains(component.getSymName())) {
        return mlir::failure();
      }

      VariableOp componentVariableOp =
          componentsMap.lookup(component.getSymName());

      // Start from the original operation.
      auto clonedOp =
          mlir::cast<BindingEquationOp>(rewriter.clone(*op.getOperation()));

      clonedOp.setVariable(componentVariableOp.getSymName());

      symbolUsers[classOp][componentVariableOp.getSymNameAttr()]
          .bindingEquationOps.insert(clonedOp);

      // Extract the component from the record.
      auto yieldOp =
          mlir::cast<YieldOp>(clonedOp.getBodyRegion().back().getTerminator());

      rewriter.setInsertionPointAfter(yieldOp);

      mlir::Value componentValue = rewriter.create<ComponentGetOp>(
          yieldOp.getLoc(),
          component.getVariableType()
              .withShape(componentVariableOp.getVariableType().getShape())
              .unwrap(),
          yieldOp.getValues()[0], component.getSymName());

      rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, componentValue);
      rewriter.setInsertionPointAfter(clonedOp);
    }

    symbolUsers[classOp][op.getVariableAttr()].bindingEquationOps.erase(op);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::LogicalResult
  replaceVariableGetOp(mlir::PatternRewriter &rewriter, ClassInterface classOp,
                       VariableGetOp op,
                       const llvm::StringMap<VariableOp> &componentsMap) const {
    llvm::SmallVector<mlir::Operation *> subscriptions;

    auto componentGetter = [&](mlir::OpBuilder &builder, mlir::Location loc,
                               llvm::StringRef componentName) -> mlir::Value {
      auto componentIt = componentsMap.find(componentName);

      if (componentIt == componentsMap.end()) {
        return nullptr;
      }

      return builder.create<VariableGetOp>(loc, componentIt->getValue());
    };

    for (mlir::Operation *user : llvm::make_early_inc_range(op->getUsers())) {
      if (mlir::failed(replaceRecordGetters(rewriter, componentGetter,
                                            subscriptions, op.getResult(),
                                            user))) {
        return mlir::failure();
      }
    }

    symbolUsers[classOp][op.getVariableAttr()].variableGetOps.erase(op);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::LogicalResult replaceVariableComponentSetOp(
      mlir::PatternRewriter &rewriter, ClassInterface classOp,
      VariableOp variableOp, VariableComponentSetOp setOp,
      const llvm::StringMap<VariableOp> &componentsMap) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(setOp);

    int64_t rootVariableRank = variableOp.getVariableType().getRank();
    size_t pathLength = setOp.getPath().size();

    if (pathLength > 2) {
      std::string composedName = getComposedComponentName(
          mlir::cast<mlir::FlatSymbolRefAttr>(setOp.getPath()[0]).getValue(),
          mlir::cast<mlir::FlatSymbolRefAttr>(setOp.getPath()[1]).getValue());

      llvm::SmallVector<mlir::Attribute> destination;

      destination.push_back(
          mlir::FlatSymbolRefAttr::get(rewriter.getContext(), composedName));

      for (size_t i = 2; i < pathLength; ++i) {
        destination.push_back(setOp.getPath()[i]);
      }

      llvm::SmallVector<mlir::Value> subscripts;
      llvm::SmallVector<int64_t> subscriptsAmounts;

      subscriptsAmounts.push_back(rootVariableRank);

      getFullRankSubscripts(rewriter, setOp.getLoc(), rootVariableRank,
                            setOp.getComponentSubscripts(0), subscripts);

      for (size_t component = 1; component < pathLength; ++component) {
        auto componentSubscripts = setOp.getComponentSubscripts(component);

        subscripts.append(componentSubscripts.begin(),
                          componentSubscripts.end());
      }

      for (mlir::IntegerAttr subscriptsAmount :
           setOp.getSubscriptionsAmounts().getAsRange<mlir::IntegerAttr>()) {
        subscriptsAmounts.push_back(subscriptsAmount.getInt());
      }

      auto componentSetOp = rewriter.create<VariableComponentSetOp>(
          setOp.getLoc(), rewriter.getArrayAttr(destination), subscripts,
          rewriter.getI64ArrayAttr(subscriptsAmounts), setOp.getValue());

      symbolUsers[classOp][mlir::cast<mlir::FlatSymbolRefAttr>(
                               componentSetOp.getPath()[0])
                               .getAttr()]
          .variableComponentSetOps.insert(componentSetOp);
    } else {
      auto componentName =
          mlir::cast<mlir::FlatSymbolRefAttr>(setOp.getPath()[1]);

      if (!componentsMap.contains(componentName.getValue())) {
        return mlir::failure();
      }

      VariableOp componentVariableOp =
          componentsMap.lookup(componentName.getValue());

      auto subscriptions = setOp.getSubscriptions();

      if (mlir::isa<mlir::TensorType>(setOp.getValue().getType())) {
        if (subscriptions.empty()) {
          rewriter.create<VariableSetOp>(setOp.getLoc(), componentVariableOp,
                                         setOp.getValue());
        } else {
          mlir::Value previousValue = rewriter.create<VariableGetOp>(
              setOp.getLoc(), componentVariableOp);

          llvm::SmallVector<mlir::Value> subscripts;

          getFullRankSubscripts(rewriter, setOp.getLoc(), rootVariableRank,
                                setOp.getComponentSubscripts(0), subscripts);

          for (size_t component = 1; component < pathLength; ++component) {
            auto componentSubscripts = setOp.getComponentSubscripts(component);

            subscripts.append(componentSubscripts.begin(),
                              componentSubscripts.end());
          }

          mlir::Value newValue = rewriter.create<TensorInsertSliceOp>(
              setOp.getLoc(), setOp.getValue(), previousValue, subscripts);

          rewriter.create<VariableSetOp>(setOp.getLoc(), componentVariableOp,
                                         newValue);
        }
      } else {
        if (subscriptions.empty()) {
          rewriter.create<VariableSetOp>(setOp.getLoc(), componentVariableOp,
                                         setOp.getValue());
        } else {
          mlir::Value previousValue = rewriter.create<VariableGetOp>(
              setOp.getLoc(), componentVariableOp);

          mlir::Value newValue = rewriter.create<TensorInsertOp>(
              setOp.getLoc(), setOp.getValue(), previousValue, subscriptions);

          rewriter.create<VariableSetOp>(setOp.getLoc(), componentVariableOp,
                                         newValue);
        }
      }
    }

    symbolUsers[classOp][mlir::cast<mlir::FlatSymbolRefAttr>(setOp.getPath()[0])
                             .getAttr()]
        .variableComponentSetOps.erase(setOp);

    rewriter.eraseOp(setOp);
    return mlir::success();
  }

  void getFullRankSubscripts(mlir::OpBuilder &builder, mlir::Location loc,
                             int64_t rank, mlir::ValueRange givenSubscripts,
                             llvm::SmallVectorImpl<mlir::Value> &result) const {
    size_t numOfGivenSubscripts = givenSubscripts.size();
    result.append(givenSubscripts.begin(), givenSubscripts.end());

    int64_t numOfAdditionalSubscripts =
        rank - static_cast<int64_t>(numOfGivenSubscripts);

    for (int64_t i = 0; i < numOfAdditionalSubscripts; ++i) {
      result.push_back(builder.create<UnboundedRangeOp>(loc));
    }
  }
};

class CallResultUnpackPattern : public RecordInliningPattern<CallOp> {
public:
  using RecordInliningPattern<CallOp>::RecordInliningPattern;

  mlir::LogicalResult
  matchAndRewrite(CallOp op, mlir::PatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> newResultTypes;
    llvm::DenseMap<size_t, llvm::StringMap<size_t>> components;

    for (auto result : llvm::enumerate(op.getResults())) {
      mlir::Type resultType = result.value().getType();

      llvm::SmallVector<mlir::StringAttr, 3> unpackedNames;
      llvm::SmallVector<mlir::Type, 3> unpackedTypes;

      if (auto componentsCount =
              unpackResultType(resultType, unpackedNames, unpackedTypes);
          componentsCount > 0) {
        for (size_t i = 0; i < componentsCount; ++i) {
          components[result.index()][unpackedNames[i]] = newResultTypes.size();
          newResultTypes.push_back(unpackedTypes[i]);
        }
      } else {
        newResultTypes.push_back(resultType);
      }
    }

    if (components.empty()) {
      return mlir::failure();
    }

    auto newCallOp =
        rewriter.create<CallOp>(op.getLoc(), op.getCallee(), newResultTypes,
                                op.getArgs(), op.getArgNames());

    size_t newResultsCounter = 0;

    for (auto oldResult : llvm::enumerate(op.getResults())) {
      if (isRecordBased(oldResult.value())) {
        llvm::SmallVector<mlir::Operation *> subscriptions;

        auto componentGetter =
            [&](mlir::OpBuilder &builder, mlir::Location loc,
                llvm::StringRef componentName) -> mlir::Value {
          return newCallOp.getResult(
              components[oldResult.index()][componentName]);
        };

        for (mlir::Operation *user :
             llvm::make_early_inc_range(oldResult.value().getUsers())) {
          if (mlir::failed(replaceRecordGetters(rewriter, componentGetter,
                                                subscriptions,
                                                oldResult.value(), user))) {
            return mlir::failure();
          }
        }
      } else {
        oldResult.value().replaceAllUsesWith(
            newCallOp.getResult(newResultsCounter++));
      }
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  size_t
  unpackResultType(mlir::Type resultType,
                   llvm::SmallVectorImpl<mlir::StringAttr> &unpackedNames,
                   llvm::SmallVectorImpl<mlir::Type> &unpackedTypes) const {
    size_t result = 0;
    mlir::Type baseType = resultType;

    if (auto tensorType = mlir::dyn_cast<mlir::TensorType>(resultType)) {
      baseType = tensorType.getElementType();
    }

    auto recordType = mlir::dyn_cast<RecordType>(baseType);

    if (!recordType) {
      return result;
    }

    llvm::SmallVector<int64_t, 3> baseDimensions;

    if (auto tensorType = mlir::dyn_cast<mlir::TensorType>(resultType)) {
      auto shape = tensorType.getShape();
      baseDimensions.append(shape.begin(), shape.end());
    }

    auto recordOp = getRecordOp(recordType);
    llvm::SmallVector<int64_t, 3> dimensions;

    for (VariableOp component : recordOp.getVariables()) {
      unpackedNames.push_back(component.getSymNameAttr());

      dimensions.clear();
      dimensions.append(baseDimensions);

      auto variableType = component.getVariableType();
      auto shape = variableType.getShape();
      dimensions.append(shape.begin(), shape.end());

      if (dimensions.empty()) {
        unpackedTypes.push_back(variableType.unwrap());
      } else {
        unpackedTypes.push_back(
            mlir::RankedTensorType::get(dimensions, variableType.unwrap()));
      }

      ++result;
    }

    return result;
  }
};

class RecordCreateOpUnpackPattern
    : public RecordInliningPattern<RecordCreateOp> {
public:
  using RecordInliningPattern<RecordCreateOp>::RecordInliningPattern;

  mlir::LogicalResult
  matchAndRewrite(RecordCreateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto recordType = mlir::cast<RecordType>(op.getResult().getType());
    auto recordOp = getRecordOp(recordType);

    llvm::SmallVector<ComponentGetOp> componentGetOps;

    for (mlir::Operation *user : llvm::make_early_inc_range(op->getUsers())) {
      if (auto getOp = mlir::dyn_cast<ComponentGetOp>(user)) {
        componentGetOps.push_back(getOp);
      }
    }

    if (componentGetOps.empty()) {
      return mlir::failure();
    }

    llvm::StringMap<mlir::Value> componentsMap;

    for (auto component : llvm::enumerate(recordOp.getVariables())) {
      componentsMap[component.value().getSymName()] =
          op.getValues()[component.index()];
    }

    for (ComponentGetOp getOp : componentGetOps) {
      rewriter.replaceOp(getOp, componentsMap[getOp.getComponentName()]);
    }

    return mlir::success();
  }
};

class TensorFromElementsUnpackPattern
    : public RecordInliningPattern<TensorFromElementsOp> {
public:
  using RecordInliningPattern<TensorFromElementsOp>::RecordInliningPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorFromElementsOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Type resultType = op.getResult().getType();
    mlir::Type resultBaseType = resultType;

    if (auto tensorType = mlir::dyn_cast<mlir::TensorType>(resultType)) {
      resultBaseType = tensorType.getElementType();
    }

    auto recordType = mlir::dyn_cast<RecordType>(resultBaseType);

    if (!recordType) {
      return mlir::failure();
    }

    auto recordOp = getRecordOp(recordType);

    llvm::SmallVector<ComponentGetOp> componentGetOps;

    for (mlir::Operation *user : llvm::make_early_inc_range(op->getUsers())) {
      if (auto getOp = mlir::dyn_cast<ComponentGetOp>(user)) {
        componentGetOps.push_back(getOp);
      }
    }

    if (componentGetOps.empty()) {
      return mlir::failure();
    }

    llvm::StringMap<mlir::Value> componentsMap;

    for (VariableOp component : recordOp.getVariables()) {
      llvm::SmallVector<mlir::Value, 3> componentValues;

      for (mlir::Value element : op.getValues()) {
        llvm::SmallVector<int64_t, 3> shape;
        llvm::ArrayRef<int64_t> elementShape;

        if (auto elementTensorType =
                mlir::dyn_cast<mlir::TensorType>(element.getType())) {
          elementShape = elementTensorType.getShape();
        }

        mergeShapes(shape, elementShape,
                    component.getVariableType().getShape());

        auto componentGetOp = rewriter.create<ComponentGetOp>(
            op.getLoc(), component.getVariableType().withShape(shape).unwrap(),
            element, component.getSymName());

        componentValues.push_back(componentGetOp);
      }

      llvm::SmallVector<int64_t, 3> shape;

      mergeShapes(shape, op.getTensor().getType().getShape(),
                  component.getVariableType().getShape());

      auto sliceOp = rewriter.create<TensorFromElementsOp>(
          op.getLoc(),
          op.getTensor().getType().clone(shape).clone(
              component.getVariableType().getElementType()),
          componentValues);

      componentsMap[component.getSymName()] = sliceOp;
    }

    llvm::SmallVector<mlir::Operation *> subscriptions;

    auto componentGetter = [&](mlir::OpBuilder &builder, mlir::Location loc,
                               llvm::StringRef componentName) -> mlir::Value {
      return componentsMap[componentName];
    };

    for (mlir::Operation *user :
         llvm::make_early_inc_range(op.getResult().getUsers())) {
      if (mlir::failed(replaceRecordGetters(rewriter, componentGetter,
                                            subscriptions, op.getResult(),
                                            user))) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }
};

class TensorBroadcastUnpackPattern
    : public RecordInliningPattern<TensorBroadcastOp> {
public:
  using RecordInliningPattern<TensorBroadcastOp>::RecordInliningPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorBroadcastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Type resultType = op.getResult().getType();
    mlir::Type resultBaseType = resultType;

    if (auto tensorType = mlir::dyn_cast<mlir::TensorType>(resultType)) {
      resultBaseType = tensorType.getElementType();
    }

    auto recordType = mlir::dyn_cast<RecordType>(resultBaseType);

    if (!recordType) {
      return mlir::failure();
    }

    auto recordOp = getRecordOp(recordType);

    llvm::SmallVector<ComponentGetOp> componentGetOps;

    for (mlir::Operation *user : llvm::make_early_inc_range(op->getUsers())) {
      if (auto getOp = mlir::dyn_cast<ComponentGetOp>(user)) {
        componentGetOps.push_back(getOp);
      }
    }

    if (componentGetOps.empty()) {
      return mlir::failure();
    }

    llvm::StringMap<mlir::Value> componentsMap;

    for (VariableOp component : recordOp.getVariables()) {
      llvm::SmallVector<mlir::Value, 3> componentValues;
      mlir::Value element = op.getValue();
      llvm::SmallVector<int64_t, 3> getResultShape;
      llvm::ArrayRef<int64_t> elementShape;

      if (auto elementTensorType =
              mlir::dyn_cast<mlir::TensorType>(element.getType())) {
        elementShape = elementTensorType.getShape();
      }

      mergeShapes(getResultShape, elementShape,
                  component.getVariableType().getShape());

      auto componentGetOp = rewriter.create<ComponentGetOp>(
          op.getLoc(),
          component.getVariableType().withShape(getResultShape).unwrap(),
          element, component.getSymName());

      componentValues.push_back(componentGetOp);

      llvm::SmallVector<int64_t, 3> shape;

      mergeShapes(shape, op.getTensor().getType().getShape(),
                  component.getVariableType().getShape());

      auto sliceOp = rewriter.create<TensorBroadcastOp>(
          op.getLoc(),
          op.getTensor().getType().clone(shape).clone(
              component.getVariableType().getElementType()),
          componentValues);

      componentsMap[component.getSymName()] = sliceOp;
    }

    llvm::SmallVector<mlir::Operation *> subscriptions;

    auto componentGetter = [&](mlir::OpBuilder &builder, mlir::Location loc,
                               llvm::StringRef componentName) -> mlir::Value {
      return componentsMap[componentName];
    };

    for (mlir::Operation *user :
         llvm::make_early_inc_range(op.getResult().getUsers())) {
      if (mlir::failed(replaceRecordGetters(rewriter, componentGetter,
                                            subscriptions, op.getResult(),
                                            user))) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }
};

class RecordCreateOpFoldPattern
    : public mlir::OpRewritePattern<RecordCreateOp> {
public:
  using mlir::OpRewritePattern<RecordCreateOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(RecordCreateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

namespace {
class RecordInliningPass
    : public mlir::bmodelica::impl::RecordInliningPassBase<RecordInliningPass> {
public:
  using RecordInliningPassBase<RecordInliningPass>::RecordInliningPassBase;

  void runOnOperation() override;

  mlir::LogicalResult
  explicitateAccesses(mlir::SymbolTableCollection &symbolTableCollection,
                      mlir::ModuleOp moduleOp);

  mlir::LogicalResult
  unpackRecordVariables(mlir::SymbolTableCollection &symbolTableCollection,
                        mlir::ModuleOp moduleOp);

  mlir::LogicalResult
  foldRecordCreateOps(mlir::SymbolTableCollection &symbolTableCollection,
                      mlir::ModuleOp moduleOp);
};
} // namespace

void RecordInliningPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTableCollection;

  if (mlir::failed(explicitateAccesses(symbolTableCollection, moduleOp))) {
    return signalPassFailure();
  }

  if (mlir::failed(unpackRecordVariables(symbolTableCollection, moduleOp))) {
    return signalPassFailure();
  }

  if (mlir::failed(foldRecordCreateOps(symbolTableCollection, moduleOp))) {
    return signalPassFailure();
  }
}

mlir::LogicalResult RecordInliningPass::explicitateAccesses(
    mlir::SymbolTableCollection &symbolTableCollection,
    mlir::ModuleOp moduleOp) {
  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<VariableSetOpUnpackPattern, VariableComponentSetOpUnpackPattern,
               EquationSideOpUnpackPattern>(&getContext(), moduleOp,
                                            symbolTableCollection);

  mlir::GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  config.setMaxIterations(mlir::GreedyRewriteConfig::kNoLimit);

  return mlir::applyPatternsGreedily(moduleOp, std::move(patterns), config);
}

mlir::LogicalResult RecordInliningPass::unpackRecordVariables(
    mlir::SymbolTableCollection &symbolTableCollection,
    mlir::ModuleOp moduleOp) {
  ClassVariableUsers symbolUsers;

  auto collectUsersFn = [&](VariableUsers &users, mlir::Operation *op) {
    op->walk([&](mlir::Operation *nestedOp) {
      if (auto startOp = mlir::dyn_cast<StartOp>(nestedOp)) {
        users[startOp.getVariableAttr().getRootReference()].startOps.insert(
            startOp);
      }

      if (auto defaultOp = mlir::dyn_cast<DefaultOp>(nestedOp)) {
        users[defaultOp.getVariableAttr()].defaultOps.insert(defaultOp);
      }

      if (auto bindingEquationOp =
              mlir::dyn_cast<BindingEquationOp>(nestedOp)) {
        users[bindingEquationOp.getVariableAttr()].bindingEquationOps.insert(
            bindingEquationOp);
      }

      if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(nestedOp)) {
        users[variableGetOp.getVariableAttr()].variableGetOps.insert(
            variableGetOp);
      }

      if (auto variableComponentSetOp =
              mlir::dyn_cast<VariableComponentSetOp>(nestedOp)) {
        auto root = mlir::cast<mlir::FlatSymbolRefAttr>(
            variableComponentSetOp.getPath()[0]);

        users[root.getAttr()].variableComponentSetOps.insert(
            variableComponentSetOp);
      }
    });
  };

  llvm::SmallVector<mlir::Operation *> stack;

  for (auto &nestedOp : moduleOp.getOps()) {
    if (auto classOp = mlir::dyn_cast<ClassInterface>(nestedOp)) {
      stack.push_back(classOp);
    }
  }

  while (!stack.empty()) {
    mlir::Operation *currentOp = stack.pop_back_val();

    for (mlir::Region &nestedRegion : currentOp->getRegions()) {
      for (auto &nestedOp : nestedRegion.getOps()) {
        stack.push_back(&nestedOp);
      }
    }

    if (auto classOp = mlir::dyn_cast<ClassInterface>(currentOp)) {
      for (mlir::Region &nestedRegion : classOp->getRegions()) {

        for (auto &nestedOp : nestedRegion.getOps()) {
          if (mlir::isa<ClassInterface>(nestedOp)) {
            continue;
          }

          collectUsersFn(symbolUsers[classOp], &nestedOp);
        }
      }
    }
  }

  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<VariableOpUnpackPattern>(&getContext(), moduleOp,
                                        symbolTableCollection, symbolUsers);

  patterns.add<CallResultUnpackPattern>(&getContext(), moduleOp,
                                        symbolTableCollection);

  mlir::GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  config.setMaxIterations(mlir::GreedyRewriteConfig::kNoLimit);

  return mlir::applyPatternsGreedily(moduleOp, std::move(patterns), config);
}

mlir::LogicalResult RecordInliningPass::foldRecordCreateOps(
    mlir::SymbolTableCollection &symbolTableCollection,
    mlir::ModuleOp moduleOp) {
  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<RecordCreateOpUnpackPattern, TensorFromElementsUnpackPattern,
               TensorBroadcastUnpackPattern>(&getContext(), moduleOp,
                                             symbolTableCollection);

  patterns.add<RecordCreateOpFoldPattern>(&getContext());

  mlir::GreedyRewriteConfig config;
  config.setMaxIterations(mlir::GreedyRewriteConfig::kNoLimit);

  return mlir::applyPatternsGreedily(moduleOp, std::move(patterns), config);
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createRecordInliningPass() {
  return std::make_unique<RecordInliningPass>();
}
} // namespace mlir::bmodelica
