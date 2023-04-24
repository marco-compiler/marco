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
      mlir::SymbolTableCollection& getSymbolTable() const
      {
        return *symbolTable;
      }

      RecordOp getRecordOp(RecordType recordType) const
      {
        return mlir::cast<RecordOp>(
            recordType.getRecordOp(getSymbolTable(), moduleOp));
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

      void replaceRecordUsage(
          mlir::PatternRewriter& rewriter,
          std::function<mlir::Value(
              mlir::OpBuilder& builder,
              mlir::Location loc,
              llvm::StringRef)> componentGetter,
          std::function<void(
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
            for (mlir::Operation* nestedUser : userResult.getUsers()) {
              replaceRecordUsage(rewriter, componentGetter, componentSetter,
                                 subscriptions, userResult, nestedUser);
            }
          }

          subscriptions.pop_back();
        } else if (auto componentGetOp =
                       mlir::dyn_cast<ComponentGetOp>(user)) {
          mlir::Value replacement = componentGetter(
              rewriter, componentGetOp.getLoc(),
              componentGetOp.getComponentName());

          replacement = applySubscriptions(
              rewriter, replacement, subscriptions);

          if (auto arrayType = replacement.getType().dyn_cast<ArrayType>();
              arrayType && arrayType.isScalar()) {
            replacement = rewriter.create<LoadOp>(
                componentGetOp.getLoc(), replacement, llvm::None);
          }

          rewriter.replaceOp(componentGetOp, replacement);
        } else if (auto componentSetOp =
                       mlir::dyn_cast<ComponentSetOp>(user)) {
          if (subscriptions.empty()) {
            componentSetter(
                rewriter,
                componentSetOp.getLoc(),
                componentSetOp.getComponentName(),
                componentSetOp.getValue());

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
                  componentSetOp.getLoc(), destination, llvm::None);

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
        } else if (auto callOp = mlir::dyn_cast<CallOp>(user)) {
          auto newCallOp = unpackCallArg(
              rewriter, callOp, usedValue, componentGetter, subscriptions);

          rewriter.replaceOp(callOp, newCallOp->getResults());
        }

        cleanSubscriptions(rewriter, subscriptions);
      }

      CallOp unpackCallArg(
          mlir::OpBuilder& builder,
          CallOp callOp,
          mlir::Value arg,
          std::function<mlir::Value(
              mlir::OpBuilder&,
              mlir::Location,
              llvm::StringRef)> componentGetter,
          llvm::ArrayRef<mlir::Operation*> subscriptions) const
      {
        llvm::SmallVector<mlir::Value> newArgs;
        llvm::SmallVector<mlir::Attribute> newArgNames;

        for (auto currentArg : llvm::enumerate(callOp.getArgs())) {
          if (currentArg.value() == arg) {
            auto recordType = currentArg.value().getType().cast<RecordType>();
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
                    callOp.getLoc(), componentValue, llvm::None);
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

        llvm::Optional<mlir::ArrayAttr> argNamesAttr = llvm::None;

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

      void cleanSubscriptions(
          mlir::PatternRewriter& rewriter,
          llvm::ArrayRef<mlir::Operation*> subscriptions) const
      {
        for (mlir::Operation* op : llvm::reverse(subscriptions)) {
          if (op->use_empty()) {
            rewriter.eraseOp(op);
          }
        }
      }

    private:
      mlir::ModuleOp moduleOp;
      mlir::SymbolTableCollection* symbolTable;
  };

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

        auto variableOp = getSymbolTable().lookupSymbolIn<VariableOp>(
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

  class VariableOpUnpackPattern : public RecordInliningPattern<VariableOp>
  {
    public:
      using RecordInliningPattern<VariableOp>::RecordInliningPattern;

      mlir::LogicalResult matchAndRewrite(
          VariableOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Type elementType = op.getVariableType().getElementType();

        if (!elementType.isa<RecordType>()) {
          return mlir::failure();
        }

        auto recordType = elementType.cast<RecordType>();
        auto recordOp = getRecordOp(recordType);

        llvm::StringMap<VariableOp> componentsMap;

        for (VariableOp component : recordOp.getVariables()) {
          llvm::SmallVector<int64_t, 3> dimensions;

          for (int64_t dimension : op.getVariableType().getShape()) {
            dimensions.push_back(dimension);
          }

          for (int64_t dimension : component.getVariableType().getShape()) {
            dimensions.push_back(dimension);
          }

          auto componentVariableType =
              op.getVariableType()
                  .withShape(dimensions)
                  .withType(component.getVariableType().getElementType());

          auto unpackedComponent = rewriter.create<VariableOp>(
              op.getLoc(),
              getComposedComponentName(op, component),
              componentVariableType);

          componentsMap[component.getSymName()] = unpackedComponent;
        }

        auto cls = op->getParentOfType<ClassInterface>();

        llvm::SmallVector<StartOp> startOps;
        llvm::SmallVector<DefaultOp> defaultOps;
        llvm::SmallVector<BindingEquationOp> bindingEquationOps;

        for (auto& bodyOp : cls->getRegion(0).getOps()) {
          if (auto startOp = mlir::dyn_cast<StartOp>(bodyOp)) {
            if (startOp.getVariableOp(getSymbolTable()).getVariableType()
                    .getElementType().isa<RecordType>()) {
              startOps.push_back(startOp);
            }
          } else if (auto defaultOp = mlir::dyn_cast<DefaultOp>(bodyOp)) {
            if (defaultOp.getVariableOp(getSymbolTable()).getVariableType()
                    .getElementType().isa<RecordType>()) {
              defaultOps.push_back(defaultOp);
            }
          } else if (auto bindingEquationOp =
                         mlir::dyn_cast<BindingEquationOp>(bodyOp)) {
            if (bindingEquationOp.getVariableOp(getSymbolTable()).getVariableType()
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
          replaceVariableGetOp(
              rewriter,
              mlir::cast<VariableGetOp>(variableGetOp),
              componentsMap);
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

        VariableOp variableOp = op.getVariableOp(getSymbolTable());

        auto recordType = variableOp.getVariableType()
                              .getElementType().cast<RecordType>();

        auto recordOp = getRecordOp(recordType);

        for (VariableOp component : recordOp.getVariables()) {
          auto clonedOp = mlir::cast<StartOp>(
              rewriter.clone(*op.getOperation()));

          clonedOp.setVariable(
              getComposedComponentName(variableOp, component));

          auto yieldOp = mlir::cast<YieldOp>(
              clonedOp.getBody()->getTerminator());

          rewriter.setInsertionPointAfter(yieldOp);

          mlir::Value componentValue = rewriter.create<ComponentGetOp>(
              yieldOp.getLoc(),
              component.getVariableType().unwrap(),
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

        VariableOp variableOp = op.getVariableOp(getSymbolTable());

        auto recordType = variableOp.getVariableType()
                              .getElementType().cast<RecordType>();

        auto recordOp = getRecordOp(recordType);

        for (VariableOp component : recordOp.getVariables()) {
          auto clonedOp = mlir::cast<DefaultOp>(
              rewriter.clone(*op.getOperation()));

          clonedOp.setVariable(
              getComposedComponentName(variableOp, component));

          auto yieldOp = mlir::cast<YieldOp>(
              clonedOp.getBody()->getTerminator());

          rewriter.setInsertionPointAfter(yieldOp);

          mlir::Value componentValue = rewriter.create<ComponentGetOp>(
              yieldOp.getLoc(),
              component.getVariableType().unwrap(),
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

        VariableOp variableOp = op.getVariableOp(getSymbolTable());

        auto recordType = variableOp.getVariableType()
                              .getElementType().cast<RecordType>();

        auto recordOp = getRecordOp(recordType);

        for (VariableOp component : recordOp.getVariables()) {
          auto clonedOp = mlir::cast<BindingEquationOp>(
              rewriter.clone(*op.getOperation()));

          clonedOp.setVariable(
              getComposedComponentName(variableOp, component));

          auto yieldOp = mlir::cast<YieldOp>(
              clonedOp.getBodyRegion().back().getTerminator());

          rewriter.setInsertionPointAfter(yieldOp);

          mlir::Value componentValue = rewriter.create<ComponentGetOp>(
              yieldOp.getLoc(),
              component.getVariableType().unwrap(),
              yieldOp.getValues()[0],
              component.getSymName());

          rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, componentValue);
          rewriter.setInsertionPointAfter(clonedOp);
        }

        rewriter.eraseOp(op);
      }

      void replaceVariableGetOp(
          mlir::PatternRewriter& rewriter,
          VariableGetOp variableGetOp,
          llvm::StringMap<VariableOp>& componentsMap) const
      {
        llvm::SmallVector<mlir::Operation*> subscriptions;

        auto componentGetter =
            [&](mlir::OpBuilder& builder,
                mlir::Location loc,
                llvm::StringRef componentName) -> mlir::Value {
          return builder.create<VariableGetOp>(
              loc, componentsMap[componentName]);
        };

        auto componentSetter =
            [&](mlir::OpBuilder& builder,
                mlir::Location loc,
                llvm::StringRef componentName,
                mlir::Value value) {
              builder.create<VariableSetOp>(
                  loc, componentsMap[componentName], value);
            };

        for (mlir::Operation* user : variableGetOp->getUsers()) {
          replaceRecordUsage(rewriter, componentGetter, componentSetter,
                             subscriptions, variableGetOp.getResult(), user);
        }

        rewriter.eraseOp(variableGetOp);
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
                    mlir::Value value) {
                  builder.create<AssignmentOp>(
                      loc,
                      newCallOp.getResult(
                          components[oldResult.index()][componentName]),
                      value);
                };

            for (mlir::Operation* user : oldResult.value().getUsers()) {
              replaceRecordUsage(rewriter, componentGetter, componentSetter,
                                 subscriptions, oldResult.value(), user);
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

        for (mlir::Operation* user : op->getUsers()) {
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
  config.maxIterations = mlir::GreedyRewriteConfig::kNoIterationLimit;

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
  config.maxIterations = mlir::GreedyRewriteConfig::kNoIterationLimit;

  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns), config);
}

mlir::LogicalResult RecordInliningPass::foldRecordCreateOps()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTable;

  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<RecordCreateOpUnpackPattern>(
      &getContext(), moduleOp, symbolTable);

  patterns.add<RecordCreateOpFoldPattern>(&getContext());

  mlir::GreedyRewriteConfig config;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoIterationLimit;

  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns), config);
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createRecordInliningPass()
  {
    return std::make_unique<RecordInliningPass>();
  }
}
