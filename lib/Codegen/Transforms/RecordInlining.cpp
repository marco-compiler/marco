#include "marco/Codegen/Transforms/RecordInlining.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_RECORDINLININGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

static std::string getComposedComponentName(
    llvm::StringRef record, llvm::StringRef component)
{
  return record.str() + "." + component.str();
}

static std::string getComposedComponentName(
    VariableOp record, VariableOp component)
{
  return getComposedComponentName(record.getSymName(), component.getSymName());
}

namespace
{
  template <typename Op>
  class RecordInliningPattern : public mlir::OpRewritePattern<Op>
  {
    public:
      RecordInliningPattern(
          mlir::MLIRContext* context,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTable)
          : mlir::OpRewritePattern<Op>(context),
            moduleOp(moduleOp),
            symbolTable(&symbolTable)
      {
      }

    protected:
      mlir::SymbolTableCollection& getSymbolTableCollection() const
      {
        return *symbolTable;
      }

      RecordOp getRecordOp(RecordType recordType) const
      {
        return mlir::cast<RecordOp>(
            recordType.getRecordOp(getSymbolTableCollection(), moduleOp));
      }

      bool isRecordBased(mlir::Value value) const
      {
        return isRecordBased(value.getType());
      }

      bool isRecordBased(mlir::Type type) const
      {
        if (auto arrayType = type.dyn_cast<ArrayType>()) {
          return arrayType.getElementType().isa<RecordType>();
        }

        return type.isa<RecordType>();
      }

      void mergeShapes(
          llvm::SmallVectorImpl<int64_t>& result,
          llvm::ArrayRef<int64_t> parent,
          llvm::ArrayRef<int64_t> child) const
      {
        result.clear();
        result.append(parent.begin(), parent.end());
        result.append(child.begin(), child.end());
      }

      mlir::LogicalResult replaceRecordUsage(
          mlir::PatternRewriter& rewriter,
          llvm::function_ref<mlir::Value(
              mlir::OpBuilder& builder,
              mlir::Location loc,
              llvm::StringRef)> componentGetter,
          llvm::function_ref<mlir::LogicalResult(
              mlir::OpBuilder&,
              mlir::Location loc,
              llvm::StringRef,
              mlir::Value)> componentSetter,
          llvm::SmallVectorImpl<mlir::Operation*>& subscriptions,
          mlir::Value usedValue,
          mlir::Operation* user) const
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(user);

        if (mlir::isa<SubscriptionOp, LoadOp>(user)) {
          subscriptions.push_back(user);

          for (mlir::Value userResult : user->getResults()) {
            for (mlir::Operation* nestedUser :
                 llvm::make_early_inc_range(userResult.getUsers())) {
              if (mlir::failed(replaceRecordUsage(
                      rewriter, componentGetter, componentSetter,
                      subscriptions, userResult, nestedUser))) {
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
          mlir::Value replacement = componentGetter(
              rewriter, componentGetOp.getLoc(),
              componentGetOp.getComponentName());

          if (!replacement) {
            return mlir::failure();
          }

          replacement = applySubscriptions(
              rewriter, replacement, subscriptions);

          if (auto arrayType = replacement.getType().dyn_cast<ArrayType>();
              arrayType && arrayType.isScalar()) {
            replacement = rewriter.create<LoadOp>(
                componentGetOp.getLoc(), replacement, std::nullopt);
          }

          rewriter.replaceOp(componentGetOp, replacement);
          return mlir::success();
        }

        if (auto componentSetOp = mlir::dyn_cast<ComponentSetOp>(user)) {
          if (subscriptions.empty()) {
            if (mlir::failed(componentSetter(
                    rewriter,
                    componentSetOp.getLoc(),
                    componentSetOp.getComponentName(),
                    componentSetOp.getValue()))) {
              return mlir::failure();
            }

            rewriter.eraseOp(componentSetOp);
          } else {
            mlir::Value destination = componentGetter(
                rewriter, componentSetOp.getLoc(),
                componentSetOp.getComponentName());

            destination = applySubscriptions(
                rewriter, destination, subscriptions);

            mlir::Value value = componentSetOp.getValue();

            if (auto recordType = value.getType().dyn_cast<RecordType>()) {
              destination = rewriter.create<LoadOp>(
                  componentSetOp.getLoc(), destination, std::nullopt);

              auto recordOp = getRecordOp(recordType);

              for (VariableOp component : recordOp.getVariables()) {
                mlir::Value componentValue = rewriter.create<ComponentGetOp>(
                    componentSetOp.getLoc(),
                    component.getVariableType().unwrap(),
                    destination, component.getSymName());

                rewriter.create<ComponentSetOp>(
                    componentSetOp.getLoc(),
                    destination, component.getSymName(), componentValue);
              }
            } else {
              rewriter.create<AssignmentOp>(
                  componentSetOp.getLoc(),
                  destination, componentSetOp.getValue());
            }

            rewriter.eraseOp(componentSetOp);
          }

          return mlir::success();
        }

        if (auto callOp = mlir::dyn_cast<CallOp>(user)) {
          auto newCallOp = unpackCallArg(
              rewriter, callOp, usedValue, componentGetter, subscriptions);

          rewriter.replaceOp(callOp, newCallOp->getResults());
          return mlir::success();
        }

        return mlir::failure();
      }

      CallOp unpackCallArg(
          mlir::OpBuilder& builder,
          CallOp callOp,
          mlir::Value arg,
          llvm::function_ref<mlir::Value(
              mlir::OpBuilder&,
              mlir::Location,
              llvm::StringRef)> componentGetter,
          llvm::ArrayRef<mlir::Operation*> subscriptions) const
      {
        llvm::SmallVector<mlir::Value> newArgs;
        llvm::SmallVector<mlir::Attribute> newArgNames;

        for (auto currentArg : llvm::enumerate(callOp.getArgs())) {
          if (currentArg.value() == arg) {
            auto recordType = mlir::cast<RecordType>(currentArg.value().getType());
            RecordOp recordOp = getRecordOp(recordType);

            for (VariableOp component : recordOp.getVariables()) {
              mlir::Value componentValue = componentGetter(
                  builder, currentArg.value().getLoc(),
                  component.getSymName());

              componentValue = applySubscriptions(
                  builder, componentValue, subscriptions);

              if (auto arrayType =
                      componentValue.getType().dyn_cast<ArrayType>();
                  arrayType && arrayType.isScalar()) {
                componentValue = builder.create<LoadOp>(
                    callOp.getLoc(), componentValue, std::nullopt);
              }

              newArgs.push_back(componentValue);

              if (auto argNames = callOp.getArgNames()) {
                auto argName = (*argNames)[currentArg.index()]
                                   .cast<mlir::FlatSymbolRefAttr>().getValue();

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

        return builder.create<CallOp>(
            callOp.getLoc(),
            callOp.getCallee(),
            callOp.getResultTypes(),
            newArgs,
            argNamesAttr);
      }

      mlir::Value applySubscriptions(
          mlir::OpBuilder& builder,
          mlir::Value root,
          llvm::ArrayRef<mlir::Operation*> subscriptions) const
      {
        mlir::Value result = root;

        for (mlir::Operation* op : subscriptions) {
          if (auto loadOp = mlir::dyn_cast<LoadOp>(op)) {
            result = builder.create<SubscriptionOp>(
                loadOp.getLoc(), result, loadOp.getIndices());
          } else if (auto subscriptionOp =
                         mlir::dyn_cast<SubscriptionOp>(op)) {
            result = builder.create<SubscriptionOp>(
                subscriptionOp.getLoc(), result, subscriptionOp.getIndices());
          }
        }

        return result;
      }

    private:
      mlir::ModuleOp moduleOp;
      mlir::SymbolTableCollection* symbolTable;
  };

  /// Unpack the assignment of a record value into multiple assignments
  /// involving the components of the record variable.
  class VariableSetOpUnpackPattern
      : public RecordInliningPattern<VariableSetOp>
  {
    public:
      using RecordInliningPattern<VariableSetOp>::RecordInliningPattern;

      mlir::LogicalResult matchAndRewrite(
          VariableSetOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Type valueType = op.getValue().getType();
        mlir::Type valueBaseType = valueType;

        if (auto arrayType = valueType.dyn_cast<ArrayType>()) {
          valueBaseType = arrayType.getElementType();
        }

        auto recordType = valueBaseType.dyn_cast<RecordType>();

        if (!recordType) {
          return mlir::failure();
        }

        auto recordOp = getRecordOp(recordType);
        auto cls = op->getParentOfType<ClassInterface>();

        auto variableOp =
            getSymbolTableCollection().lookupSymbolIn<VariableOp>(
                cls, op.getVariableAttr());

        mlir::Value recordValue =
            rewriter.create<VariableGetOp>(op.getLoc(), variableOp);

        for (VariableOp recordVariableOp : recordOp.getVariables()) {
          mlir::Value componentValue = rewriter.create<ComponentGetOp>(
              op.getLoc(),
              recordVariableOp.getVariableType().unwrap(),
              op.getValue(),
              recordVariableOp.getSymName());

          rewriter.create<ComponentSetOp>(
              op.getLoc(),
              recordValue,
              recordVariableOp.getSymName(),
              componentValue);
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  /// Unpack the assignment of a record component into multiple assignments
  /// involving the components of the record variable.
  /// Together with the above pattern, this enables the handling of assignments
  /// of nested records.
  class ComponentSetOpUnpackPattern
      : public RecordInliningPattern<ComponentSetOp>
  {
    public:
      using RecordInliningPattern<ComponentSetOp>::RecordInliningPattern;

      mlir::LogicalResult matchAndRewrite(
          ComponentSetOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Type valueType = op.getValue().getType();
        mlir::Type valueBaseType = valueType;

        if (auto arrayType = valueType.dyn_cast<ArrayType>()) {
          valueBaseType = arrayType.getElementType();
        }

        auto recordType = valueBaseType.dyn_cast<RecordType>();

        if (!recordType) {
          return mlir::failure();
        }

        auto recordOp = getRecordOp(recordType);

        for (VariableOp recordVariableOp : recordOp.getVariables()) {
          mlir::Value componentValue = rewriter.create<ComponentGetOp>(
              op.getLoc(),
              recordVariableOp.getVariableType().unwrap(),
              op.getValue(),
              recordVariableOp.getSymName());

          rewriter.create<ComponentSetOp>(
              op.getLoc(),
              op.getVariable(),
              recordVariableOp.getSymName(),
              componentValue);
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class EquationSideOpUnpackPattern
      : public RecordInliningPattern<EquationSideOp>
  {
    public:
      using RecordInliningPattern<EquationSideOp>::RecordInliningPattern;

      mlir::LogicalResult matchAndRewrite(
          EquationSideOp op, mlir::PatternRewriter& rewriter) const override
      {
        llvm::SmallVector<mlir::Value, 3> newValues;
        bool recordFound = false;

        for (mlir::Value value : op.getValues()) {
          if (auto recordType = value.getType().dyn_cast<RecordType>()) {
            auto recordOp = getRecordOp(recordType);

            for (VariableOp component : recordOp.getVariables()) {
              auto componentGetOp = rewriter.create<ComponentGetOp>(
                  value.getLoc(),
                  component.getVariableType().unwrap(),
                  value,
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

  /// Unpack record variables into their components.
  class VariableOpUnpackPattern : public RecordInliningPattern<VariableOp>
  {
    public:
      using RecordInliningPattern<VariableOp>::RecordInliningPattern;

      mlir::LogicalResult matchAndRewrite(
          VariableOp op, mlir::PatternRewriter& rewriter) const override
      {
        VariableType variableType = op.getVariableType();
        mlir::Type elementType = variableType.getElementType();

        if (!elementType.isa<RecordType>()) {
          // Not a record or an array of records.
          return mlir::failure();
        }

        // Create a variable for each component and map it for faster lookups.
        auto recordType = elementType.cast<RecordType>();
        auto recordOp = getRecordOp(recordType);

        llvm::StringMap<VariableOp> componentsMap;

        for (VariableOp component : recordOp.getVariables()) {
          llvm::SmallVector<int64_t, 3> dimensions;

          // Use the shape of the original record variable.
          for (int64_t dimension : op.getVariableType().getShape()) {
            dimensions.push_back(dimension);
          }

          // Append the dimensions of the component.
          for (int64_t dimension : component.getVariableType().getShape()) {
            dimensions.push_back(dimension);
          }

          // Start from the original variable type in order to keep the
          // modifiers.
          auto componentVariableType =
              variableType
                  .withShape(dimensions)
                  .withType(component.getVariableType().getElementType());

          // Create the variable for the component.
          auto unpackedComponent = rewriter.create<VariableOp>(
              op.getLoc(),
              getComposedComponentName(op, component),
              componentVariableType);

          componentsMap[component.getSymName()] = unpackedComponent;
        }

        // Replace the uses of the original record.
        auto cls = op->getParentOfType<ClassInterface>();

        llvm::SmallVector<StartOp> startOps;
        llvm::SmallVector<DefaultOp> defaultOps;
        llvm::SmallVector<BindingEquationOp> bindingEquationOps;

        for (auto& bodyOp : cls->getRegion(0).getOps()) {
          if (auto startOp = mlir::dyn_cast<StartOp>(bodyOp)) {
            if (startOp.getVariable() == op.getSymName() &&
                startOp.getVariableOp(
                           getSymbolTableCollection()).getVariableType()
                    .getElementType().isa<RecordType>()) {
              startOps.push_back(startOp);
            }
          }

          if (auto defaultOp = mlir::dyn_cast<DefaultOp>(bodyOp)) {
            if (defaultOp.getVariable() == op.getSymName() &&
                defaultOp.getVariableOp(
                             getSymbolTableCollection()).getVariableType()
                    .getElementType().isa<RecordType>()) {
              defaultOps.push_back(defaultOp);
            }
          }

          if (auto bindingEquationOp =
                         mlir::dyn_cast<BindingEquationOp>(bodyOp)) {
            if (bindingEquationOp.getVariable() == op.getSymName() &&
                bindingEquationOp.getVariableOp(
                                     getSymbolTableCollection()).getVariableType()
                    .getElementType().isa<RecordType>()) {
              bindingEquationOps.push_back(bindingEquationOp);
            }
          }
        }

        for (StartOp startOp : startOps) {
          unpackStartOp(rewriter, startOp);
        }

        for (DefaultOp defaultOp : defaultOps) {
          unpackDefaultOp(rewriter, defaultOp);
        }

        for (BindingEquationOp bindingEquationOp : bindingEquationOps) {
          unpackBindingEquationOp(rewriter, bindingEquationOp);
        }

        llvm::SmallVector<mlir::Operation*> variableGetOps;
        llvm::SmallVector<mlir::Operation*> variableSetOps;

        cls->getRegion(0).walk([&](VariableGetOp getOp) {
          if (getOp.getVariable() == op.getSymName()) {
            variableGetOps.push_back(getOp);
          }
        });

        for (mlir::Operation* variableGetOp : variableGetOps) {
          if (mlir::failed(replaceVariableGetOp(
                  rewriter,
                  mlir::cast<VariableGetOp>(variableGetOp),
                  componentsMap))) {
            return mlir::failure();
          }
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }

    private:
      void unpackStartOp(
          mlir::PatternRewriter& rewriter,
          StartOp op) const
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(op);

        VariableOp variableOp = op.getVariableOp(getSymbolTableCollection());

        auto recordType = variableOp.getVariableType()
                              .getElementType().cast<RecordType>();

        auto recordOp = getRecordOp(recordType);

        for (VariableOp component : recordOp.getVariables()) {
          llvm::SmallVector<int64_t, 3> shape;

          mergeShapes(
              shape,
              variableOp.getVariableType().getShape(),
              component.getVariableType().getShape());

          auto clonedOp = mlir::cast<StartOp>(
              rewriter.clone(*op.getOperation()));

          clonedOp.setVariable(
              getComposedComponentName(variableOp, component));

          auto yieldOp = mlir::cast<YieldOp>(
              clonedOp.getBody()->getTerminator());

          rewriter.setInsertionPointAfter(yieldOp);

          mlir::Value componentValue = rewriter.create<ComponentGetOp>(
              yieldOp.getLoc(),
              component.getVariableType().withShape(shape).unwrap(),
              yieldOp.getValues()[0],
              component.getSymName());

          rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, componentValue);
          rewriter.setInsertionPointAfter(clonedOp);
        }

        rewriter.eraseOp(op);
      }

      void unpackDefaultOp(
          mlir::PatternRewriter& rewriter,
          DefaultOp op) const
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(op);

        VariableOp variableOp = op.getVariableOp(getSymbolTableCollection());

        auto recordType = variableOp.getVariableType()
                              .getElementType().cast<RecordType>();

        auto recordOp = getRecordOp(recordType);

        for (VariableOp component : recordOp.getVariables()) {
          llvm::SmallVector<int64_t, 3> shape;

          mergeShapes(
              shape,
              variableOp.getVariableType().getShape(),
              component.getVariableType().getShape());

          auto clonedOp = mlir::cast<DefaultOp>(
              rewriter.clone(*op.getOperation()));

          clonedOp.setVariable(
              getComposedComponentName(variableOp, component));

          auto yieldOp = mlir::cast<YieldOp>(
              clonedOp.getBody()->getTerminator());

          rewriter.setInsertionPointAfter(yieldOp);

          mlir::Value componentValue = rewriter.create<ComponentGetOp>(
              yieldOp.getLoc(),
              component.getVariableType().withShape(shape).unwrap(),
              yieldOp.getValues()[0],
              component.getSymName());

          rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, componentValue);
          rewriter.setInsertionPointAfter(clonedOp);
        }

        rewriter.eraseOp(op);
      }

      void unpackBindingEquationOp(
          mlir::PatternRewriter& rewriter,
          BindingEquationOp op) const
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(op);

        VariableOp variableOp = op.getVariableOp(getSymbolTableCollection());

        auto recordType = variableOp.getVariableType()
                              .getElementType().cast<RecordType>();

        auto recordOp = getRecordOp(recordType);

        for (VariableOp component : recordOp.getVariables()) {
          llvm::SmallVector<int64_t, 3> shape;

          mergeShapes(
              shape,
              variableOp.getVariableType().getShape(),
              component.getVariableType().getShape());

          auto clonedOp = mlir::cast<BindingEquationOp>(
              rewriter.clone(*op.getOperation()));

          clonedOp.setVariable(
              getComposedComponentName(variableOp, component));

          auto yieldOp = mlir::cast<YieldOp>(
              clonedOp.getBodyRegion().back().getTerminator());

          rewriter.setInsertionPointAfter(yieldOp);

          mlir::Value componentValue = rewriter.create<ComponentGetOp>(
              yieldOp.getLoc(),
              component.getVariableType().withShape(shape).unwrap(),
              yieldOp.getValues()[0],
              component.getSymName());

          rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, componentValue);
          rewriter.setInsertionPointAfter(clonedOp);
        }

        rewriter.eraseOp(op);
      }

      mlir::LogicalResult replaceVariableGetOp(
          mlir::PatternRewriter& rewriter,
          VariableGetOp variableGetOp,
          const llvm::StringMap<VariableOp>& componentsMap) const
      {
        llvm::SmallVector<mlir::Operation*> subscriptions;

        auto componentGetter =
            [&](mlir::OpBuilder& builder,
                mlir::Location loc,
                llvm::StringRef componentName) -> mlir::Value {
          auto componentIt = componentsMap.find(componentName);

          if (componentIt == componentsMap.end()) {
            return nullptr;
          }

          return builder.create<VariableGetOp>(loc, componentIt->getValue());
        };

        auto componentSetter =
            [&](mlir::OpBuilder& builder,
                mlir::Location loc,
                llvm::StringRef componentName,
                mlir::Value value) -> mlir::LogicalResult {
              auto componentIt = componentsMap.find(componentName);

              if (componentIt == componentsMap.end()) {
                return mlir::failure();
              }

              builder.create<VariableSetOp>(
                  loc, componentIt->getValue(), value);

              return mlir::success();
            };

        for (mlir::Operation* user :
             llvm::make_early_inc_range(variableGetOp->getUsers())) {
          if (mlir::failed(replaceRecordUsage(
                  rewriter, componentGetter, componentSetter,
                  subscriptions, variableGetOp.getResult(), user))) {
            return mlir::failure();
          }
        }

        rewriter.eraseOp(variableGetOp);
        return mlir::success();
      }
  };

  class CallResultUnpackPattern : public RecordInliningPattern<CallOp>
  {
    public:
      using RecordInliningPattern<CallOp>::RecordInliningPattern;

      mlir::LogicalResult matchAndRewrite(
          CallOp op, mlir::PatternRewriter& rewriter) const override
      {
        llvm::SmallVector<mlir::Type> newResultTypes;
        llvm::DenseMap<size_t, llvm::StringMap<size_t>> components;

        for (auto result : llvm::enumerate(op.getResults())) {
          mlir::Type resultType = result.value().getType();

          llvm::SmallVector<mlir::StringAttr, 3> unpackedNames;
          llvm::SmallVector<mlir::Type, 3> unpackedTypes;

          if (auto componentsCount = unpackResultType(
                  resultType, unpackedNames, unpackedTypes);
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

        auto newCallOp = rewriter.create<CallOp>(
            op.getLoc(),
            op.getCallee(), newResultTypes, op.getArgs(), op.getArgNames());

        size_t newResultsCounter = 0;

        for (auto oldResult : llvm::enumerate(op.getResults())) {
          if (isRecordBased(oldResult.value())) {
            llvm::SmallVector<mlir::Operation*> subscriptions;

            auto componentGetter =
                [&](mlir::OpBuilder& builder,
                    mlir::Location loc,
                    llvm::StringRef componentName) -> mlir::Value {
                  return newCallOp.getResult(
                      components[oldResult.index()][componentName]);
                };

            auto componentSetter =
                [&](mlir::OpBuilder& builder,
                    mlir::Location loc,
                    llvm::StringRef componentName,
                    mlir::Value value) -> mlir::LogicalResult {
                  builder.create<AssignmentOp>(
                      loc,
                      newCallOp.getResult(
                          components[oldResult.index()][componentName]),
                      value);

                  return mlir::success();
                };

            for (mlir::Operation* user :
                 llvm::make_early_inc_range(oldResult.value().getUsers())) {
              if (mlir::failed(replaceRecordUsage(
                      rewriter, componentGetter, componentSetter,
                      subscriptions, oldResult.value(), user))) {
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
      size_t unpackResultType(
          mlir::Type resultType,
          llvm::SmallVectorImpl<mlir::StringAttr>& unpackedNames,
          llvm::SmallVectorImpl<mlir::Type>& unpackedTypes) const
      {
        size_t result = 0;
        mlir::Type baseType = resultType;

        if (auto arrayType = resultType.dyn_cast<ArrayType>()) {
          baseType = arrayType.getElementType();
        }

        auto recordType = baseType.dyn_cast<RecordType>();

        if (!recordType) {
          return result;
        }

        llvm::SmallVector<int64_t, 3> baseDimensions;

        if (auto arrayType = resultType.dyn_cast<ArrayType>()) {
          auto shape = arrayType.getShape();
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
                ArrayType::get(dimensions, variableType.unwrap()));
          }

          ++result;
        }

        return result;
      }
  };

  class RecordCreateOpUnpackPattern
      : public RecordInliningPattern<RecordCreateOp>
  {
    public:
      using RecordInliningPattern<RecordCreateOp>::RecordInliningPattern;

      mlir::LogicalResult matchAndRewrite(
          RecordCreateOp op, mlir::PatternRewriter& rewriter) const override
      {
        auto recordType = op.getResult().getType().cast<RecordType>();
        auto recordOp = getRecordOp(recordType);

        llvm::SmallVector<ComponentGetOp> componentGetOps;
        llvm::SmallVector<ComponentSetOp> componentSetOps;

        for (mlir::Operation* user :
             llvm::make_early_inc_range(op->getUsers())) {
          if (auto getOp = mlir::dyn_cast<ComponentGetOp>(user)) {
            componentGetOps.push_back(getOp);
          } else if (auto setOp = mlir::dyn_cast<ComponentSetOp>(user)) {
            componentSetOps.push_back(setOp);
          }
        }

        if (!componentSetOps.empty()) {
          return mlir::failure();
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

  class ArrayFromElementsUnpackPattern
      : public RecordInliningPattern<ArrayFromElementsOp>
  {
    public:
      using RecordInliningPattern<ArrayFromElementsOp>::RecordInliningPattern;

      mlir::LogicalResult matchAndRewrite(
          ArrayFromElementsOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        mlir::Type resultType = op.getResult().getType();
        mlir::Type resultBaseType = resultType;

        if (auto arrayType = resultType.dyn_cast<ArrayType>()) {
          resultBaseType = arrayType.getElementType();
        }

        auto recordType = resultBaseType.dyn_cast<RecordType>();

        if (!recordType) {
          return mlir::failure();
        }

        auto recordOp = getRecordOp(recordType);

        llvm::SmallVector<ComponentGetOp> componentGetOps;
        llvm::SmallVector<ComponentSetOp> componentSetOps;

        for (mlir::Operation* user :
             llvm::make_early_inc_range(op->getUsers())) {
          if (auto getOp = mlir::dyn_cast<ComponentGetOp>(user)) {
            componentGetOps.push_back(getOp);
          } else if (auto setOp = mlir::dyn_cast<ComponentSetOp>(user)) {
            componentSetOps.push_back(setOp);
          }
        }

        if (componentSetOps.empty() && componentGetOps.empty()) {
          return mlir::failure();
        }

        llvm::StringMap<mlir::Value> componentsMap;

        for (VariableOp component : recordOp.getVariables()) {
          llvm::SmallVector<mlir::Value, 3> componentValues;

          for (mlir::Value element : op.getValues()) {
            llvm::SmallVector<int64_t, 3> shape;
            llvm::ArrayRef<int64_t> elementShape = std::nullopt;

            if (auto elementArrayType = element.getType().dyn_cast<ArrayType>()) {
              elementShape = elementArrayType.getShape();
            }

            mergeShapes(
                shape, elementShape, component.getVariableType().getShape());

            auto componentGetOp = rewriter.create<ComponentGetOp>(
                op.getLoc(),
                component.getVariableType().withShape(shape).unwrap(),
                element,
                component.getSymName());

            componentValues.push_back(componentGetOp);
          }

          llvm::SmallVector<int64_t, 3> shape;

          mergeShapes(
              shape,
              op.getArrayType().getShape(),
              component.getVariableType().getShape());

          auto sliceOp = rewriter.create<ArrayFromElementsOp>(
              op.getLoc(),
              op.getArrayType().withShape(shape).toElementType(
                  component.getVariableType().getElementType()),
              componentValues);

          componentsMap[component.getSymName()] = sliceOp;
        }

        llvm::SmallVector<mlir::Operation*> subscriptions;

        auto componentGetter =
            [&](mlir::OpBuilder& builder,
                mlir::Location loc,
                llvm::StringRef componentName) -> mlir::Value {
          return componentsMap[componentName];
        };

        auto componentSetter =
            [&](mlir::OpBuilder& builder,
                mlir::Location loc,
                llvm::StringRef componentName,
                mlir::Value value) -> mlir::LogicalResult {
              builder.create<AssignmentOp>(
                  loc, componentsMap[componentName], value);

              return mlir::success();
            };

        for (mlir::Operation* user :
             llvm::make_early_inc_range(op.getResult().getUsers())) {
          if (mlir::failed(replaceRecordUsage(
                  rewriter, componentGetter, componentSetter,
                  subscriptions, op.getResult(), user))) {
            return mlir::failure();
          }
        }

        return mlir::success();
      }
  };

  class RecordCreateOpFoldPattern
      : public mlir::OpRewritePattern<RecordCreateOp>
  {
    public:
      using mlir::OpRewritePattern<RecordCreateOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          RecordCreateOp op, mlir::PatternRewriter& rewriter) const override
      {
        if (op->use_empty()) {
          rewriter.eraseOp(op);
          return mlir::success();
        }

        return mlir::failure();
      }
  };
}

namespace
{
  class RecordInliningPass
      : public mlir::modelica::impl::RecordInliningPassBase<
            RecordInliningPass>
  {
    public:
      using RecordInliningPassBase::RecordInliningPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(explicitateAccesses())) {
          return signalPassFailure();
        }

        if (mlir::failed(unpackRecordVariables())) {
          return signalPassFailure();
        }

        if (mlir::failed(foldRecordCreateOps())) {
          return signalPassFailure();
        }
      }

      mlir::LogicalResult explicitateAccesses();

      mlir::LogicalResult unpackRecordVariables();

      mlir::LogicalResult foldRecordCreateOps();
  };
}

mlir::LogicalResult RecordInliningPass::explicitateAccesses()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTable;

  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<
      VariableSetOpUnpackPattern,
      ComponentSetOpUnpackPattern,
      EquationSideOpUnpackPattern>(&getContext(), moduleOp, symbolTable);

  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;

  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns), config);
}

mlir::LogicalResult RecordInliningPass::unpackRecordVariables()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTable;

  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<
      VariableOpUnpackPattern,
      CallResultUnpackPattern>(&getContext(), moduleOp, symbolTable);

  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;

  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns), config);
}

mlir::LogicalResult RecordInliningPass::foldRecordCreateOps()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTable;

  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<
      RecordCreateOpUnpackPattern,
      ArrayFromElementsUnpackPattern>(&getContext(), moduleOp, symbolTable);

  patterns.add<RecordCreateOpFoldPattern>(&getContext());

  mlir::GreedyRewriteConfig config;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;

  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns), config);
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createRecordInliningPass()
  {
    return std::make_unique<RecordInliningPass>();
  }
}
