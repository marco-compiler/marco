#include "marco/Dialect/BaseModelica/Transforms/EquationExpressionOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

using namespace ::mlir::bmodelica;

namespace {
void printExpression(llvm::raw_ostream &os, mlir::Value value,
                     const llvm::DenseMap<mlir::Value, int64_t> &inductions) {
  mlir::Operation *op = value.getDefiningOp();

  if (!op) {
    if (auto inductionsIt = inductions.find(value);
        inductionsIt != inductions.end()) {
      os << "{ind " << inductionsIt->getSecond() << "}";
    } else {
      os << "(" << value << ")";
    }
    return;
  }

  auto expressionOp = mlir::dyn_cast<EquationExpressionOpInterface>(op);

  if (!expressionOp) {
    os << "(" << value << ")";
    return;
  }

  expressionOp.printExpression(os, inductions);
}

template <typename BinaryOpType>
void printBinaryExpression(
    mlir::Operation *op, llvm::raw_ostream &os,
    const llvm::DenseMap<mlir::Value, int64_t> &inductions,
    const llvm::StringLiteral opSymbol) {
  auto castedOp = mlir::cast<BinaryOpType>(op);

  os << "(";
  ::printExpression(os, castedOp.getLhs(), inductions);
  os << " " << opSymbol << " ";
  ::printExpression(os, castedOp.getRhs(), inductions);
  os << ")";
}

template <typename UnaryOpType>
void printUnaryExpression(
    mlir::Operation *op, llvm::raw_ostream &os,
    const llvm::DenseMap<mlir::Value, int64_t> &inductions,
    const llvm::StringLiteral opName) {

  auto castedOp = mlir::cast<UnaryOpType>(op);

  os << opName << "(";
  ::printExpression(os, castedOp.getOperand(), inductions);
  os << ")";
}

bool areExpressionOperandsEquivalent(
    mlir::ValueRange firstOperands, mlir::ValueRange secondOperands,
    llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
        &inductionAccesses,
    mlir::SymbolTableCollection &symbolTableCollection) {
  if (firstOperands.size() != secondOperands.size()) {
    return false;
  }

  llvm::SmallVector<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
      tmpInductionAccesses;

  for (auto [firstOperand, secondOperand] :
       llvm::zip(firstOperands, secondOperands)) {
    auto firstExp = firstOperand.getDefiningOp<EquationExpressionOpInterface>();
    auto secondExp =
        secondOperand.getDefiningOp<EquationExpressionOpInterface>();

    // For expression operations, continue the recursion
    if (firstExp && secondExp) {
      if (!firstExp.isEquivalent(secondExp, inductionAccesses,
                                 symbolTableCollection)) {
        return false;
      }
      continue;
    }

    // Check if both are induction variables
    auto firstArg = mlir::dyn_cast<mlir::BlockArgument>(firstOperand);
    auto secondArg = mlir::dyn_cast<mlir::BlockArgument>(secondOperand);
    if (!firstArg || !secondArg) {
      return false;
    }
    // The args may come from a nested operation, not the template loop args.
    auto firstIsInduction =
        llvm::isa<EquationTemplateOp>(firstArg.getOwner()->getParentOp());
    auto secondIsInduction =
        llvm::isa<EquationTemplateOp>(secondArg.getOwner()->getParentOp());
    if (firstIsInduction && secondIsInduction) {
      // If both are inductions we have a match
      tmpInductionAccesses.push_back({firstArg, secondArg});
    } else if (firstIsInduction || secondIsInduction) {
      // If one of them was then its a miss
      return false;
    }
  }

  // Add collected inductions to the total (only if all were a match).
  inductionAccesses.append(tmpInductionAccesses);
  return true;
}

bool areEquationExpressionsEquivalent(
    mlir::Operation *firstOp, mlir::Operation *secondOp,
    llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
        &inductionAccesses,
    mlir::SymbolTableCollection &symbolTableCollection) {
  if (firstOp->getResultTypes() != secondOp->getResultTypes()) {
    return false;
  }

  return areExpressionOperandsEquivalent(
      firstOp->getOperands(), secondOp->getOperands(), inductionAccesses,
      symbolTableCollection);
}

template <typename OpType>
bool isEquivalent(
    mlir::Operation *op, mlir::Operation *other,
    llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
        &inductionAccesses,
    mlir::SymbolTableCollection &symbolTableCollection) {
  auto otherCasted = mlir::dyn_cast<OpType>(other);

  if (!otherCasted) {
    return false;
  }

  return areEquationExpressionsEquivalent(
      op, otherCasted.getOperation(), inductionAccesses, symbolTableCollection);
}

#define DEFINE_DEFAULT_IS_EQUIVALENT(T)                                        \
  bool isEquivalent(mlir::Operation *op, mlir::Operation *other,               \
                    llvm::SmallVectorImpl<                                     \
                        std::pair<mlir::BlockArgument, mlir::BlockArgument>>   \
                        &inductionAccesses,                                    \
                    mlir::SymbolTableCollection &symbolTableCollection)        \
      const {                                                                  \
    return ::isEquivalent<T>(op, other, inductionAccesses,                     \
                             symbolTableCollection);                           \
  }
} // namespace

namespace {
struct RangeOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::RangeOpInterface,
                                                          RangeOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    os << "range(";
    llvm::interleaveComma(op->getOperands(), os, [&](mlir::Value operand) {
      ::printExpression(os, operand, inductions);
    });
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(RangeOp)
};

struct ReductionOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::ReductionOpInterface, ReductionOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ReductionOp>(op);

    // Add the inductions to the inductions map.
    llvm::DenseMap<mlir::Value, int64_t> expandedInductions(inductions);
    auto inductionValues = castedOp.getInductions();

    for (mlir::Value inductionValue : inductionValues) {
      auto id = static_cast<int64_t>(expandedInductions.size());
      expandedInductions[inductionValue] = id;
    }

    // Print the operation.
    os << castedOp.getAction();
    os << "(";

    auto terminator = mlir::cast<YieldOp>(castedOp.getBody()->getTerminator());

    llvm::interleaveComma(terminator.getValues(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, expandedInductions);
    });

    os << " for ";
    auto iterables = castedOp.getIterables();

    for (size_t i = 0, e = inductionValues.size(); i < e; ++i) {
      if (i != 0) {
        os << ", ";
      }

      ::printExpression(os, inductionValues[i], expandedInductions);
    }

    os << " in ";

    llvm::interleaveComma(iterables, os, [&](mlir::Value exp) {
      ::printExpression(os, exp, expandedInductions);
    });

    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
          &inductionAccesses,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto casted = mlir::cast<ReductionOp>(op);
    auto otherCasted = mlir::dyn_cast<ReductionOp>(other);

    if (!otherCasted) {
      return false;
    }

    if (casted.getAction() != otherCasted.getAction()) {
      return false;
    }

    if (!areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                          inductionAccesses,
                                          symbolTableCollection)) {
      return false;
    }

    auto yieldOp = mlir::cast<YieldOp>(casted.getBody()->getTerminator());
    auto otherYieldOp =
        mlir::cast<YieldOp>(otherCasted.getBody()->getTerminator());
    return areExpressionOperandsEquivalent(
        yieldOp.getValues(), otherYieldOp.getValues(), inductionAccesses,
        symbolTableCollection);
  }

  uint64_t getNumOfExpressionElements(mlir::Operation *op) const {
    auto castedOp = mlir::cast<ReductionOp>(op);

    auto terminator = mlir::cast<YieldOp>(castedOp.getBody()->getTerminator());

    return terminator.getValues().size();
  }

  mlir::Value getExpressionElement(mlir::Operation *op,
                                   const uint64_t element) const {
    auto castedOp = mlir::cast<ReductionOp>(op);

    auto terminator = mlir::cast<YieldOp>(castedOp.getBody()->getTerminator());

    return terminator.getValues()[element];
  }

  llvm::SmallVector<mlir::Value>
  getAdditionalInductions(mlir::Operation *op) const {
    auto castedOp = mlir::cast<ReductionOp>(op);
    llvm::SmallVector<mlir::Value> result;
    auto inductions = castedOp.getInductions();
    result.append(inductions.begin(), inductions.end());
    return result;
  }

  mlir::LogicalResult
  mapAdditionalInductions(mlir::Operation *op,
                          AdditionalInductions &additionalInductions) const {
    auto castedOp = mlir::cast<ReductionOp>(op);

    IndexSet indices;
    llvm::SmallVector<std::pair<mlir::Value, uint64_t>> inductionsMap;

    for (const auto &[induction, iterable] :
         zip(castedOp.getInductions(), castedOp.getIterables())) {
      auto constantOp = iterable.getDefiningOp<ConstantOp>();

      if (!constantOp) {
        return mlir::failure();
      }

      auto iterableAttr = constantOp.getValue();

      if (auto rangeAttr = mlir::dyn_cast<IntegerRangeAttr>(iterableAttr)) {
        assert(rangeAttr.getStep() == 1);

        auto lowerBound =
            static_cast<Range::data_type>(rangeAttr.getLowerBound());

        auto upperBound =
            static_cast<Range::data_type>(rangeAttr.getUpperBound());

        Range range(lowerBound, upperBound + 1);
        indices = indices.append(IndexSet(MultidimensionalRange(range)));

        auto currentDimension = static_cast<int64_t>(indices.rank() - 1);
        inductionsMap.emplace_back(induction, currentDimension);

        continue;
      }

      if (auto rangeAttr = mlir::dyn_cast<RealRangeAttr>(iterableAttr)) {
        assert(rangeAttr.getStep().convertToDouble() == 1);

        auto lowerBound = static_cast<Range::data_type>(
            rangeAttr.getLowerBound().convertToDouble());

        auto upperBound = static_cast<Range::data_type>(
            rangeAttr.getUpperBound().convertToDouble());

        Range range(lowerBound, upperBound);
        indices = indices.append(IndexSet(MultidimensionalRange(range)));

        auto currentDimension = static_cast<int64_t>(indices.rank() - 1);
        inductionsMap.emplace_back(induction, currentDimension);

        continue;
      }

      return mlir::failure();
    }

    uint64_t iterationSpace =
        additionalInductions.addIterationSpace(std::move(indices));

    for (size_t i = 0, e = inductionsMap.size(); i < e; ++i) {
      additionalInductions.addInductionVariable(
          inductionsMap[i].first, iterationSpace, inductionsMap[i].second);
    }

    return mlir::success();
  }
};

struct CallOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::CallOpInterface,
                                                          CallOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto casted = mlir::cast<CallOp>(op);
    os << casted.getCallee() << "(";

    llvm::interleaveComma(casted.getArgs(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
          &inductionAccesses,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto casted = mlir::cast<CallOp>(op);
    auto otherCasted = mlir::dyn_cast<CallOp>(other);

    if (!otherCasted) {
      return false;
    }

    if (casted.getCallee() != otherCasted.getCallee()) {
      return false;
    }

    if (casted->getResultTypes() != otherCasted->getResultTypes()) {
      return false;
    }

    auto argNames = casted.getArgNames();
    auto otherArgNames = otherCasted.getArgNames();

    llvm::StringMap<size_t> argNamesPos;
    llvm::StringMap<size_t> otherArgNamesPos;

    if (argNames) {
      getArgNamesPos(*argNames, argNamesPos);
    }

    if (otherArgNames) {
      getArgNamesPos(*otherArgNames, otherArgNamesPos);
    }

    if (argNames && otherArgNames) {
      if (!haveSameArgNames(argNamesPos, otherArgNamesPos)) {
        return false;
      }

      for (auto &entry : argNamesPos) {
        mlir::Value arg = casted.getArgs()[entry.getValue()];
        mlir::Value otherArg =
            otherCasted.getArgs()[otherArgNamesPos[entry.getKey()]];

        if (!areExpressionOperandsEquivalent(arg, otherArg, inductionAccesses,
                                             symbolTableCollection)) {
          return false;
        }
      }
    } else if (argNames) {
      if (mlir::failed(getArgNamesPos(otherCasted, symbolTableCollection,
                                      otherArgNamesPos))) {
        return false;
      }

      if (!compareNamedUnnamedArgs(casted.getArgs(), argNamesPos,
                                   otherCasted.getArgs(), otherArgNamesPos,
                                   false, inductionAccesses,
                                   symbolTableCollection)) {
        return false;
      }
    } else if (otherArgNames) {
      if (mlir::failed(
              getArgNamesPos(casted, symbolTableCollection, argNamesPos))) {
        return false;
      }

      if (!compareNamedUnnamedArgs(otherCasted.getArgs(), otherArgNamesPos,
                                   casted.getArgs(), argNamesPos, true,
                                   inductionAccesses, symbolTableCollection)) {
        return false;
      }
    } else {
      if (!areExpressionOperandsEquivalent(
              casted.getArgs(), otherCasted.getArgs(), inductionAccesses,
              symbolTableCollection)) {
        return false;
      }
    }

    return true;
  }

  static void getArgNamesPos(mlir::ArrayAttr argNames,
                             llvm::StringMap<size_t> &pos) {
    for (auto argName : llvm::enumerate(argNames)) {
      auto name =
          mlir::cast<mlir::FlatSymbolRefAttr>(argName.value()).getValue();
      pos[name] = argName.index();
    }
  }

  static mlir::LogicalResult
  getArgNamesPos(CallOp callOp,
                 mlir::SymbolTableCollection &symbolTableCollection,
                 llvm::StringMap<size_t> &pos) {
    auto otherFunctionOp = mlir::dyn_cast<FunctionOp>(callOp.getFunction(
        callOp->getParentOfType<mlir::ModuleOp>(), symbolTableCollection));

    if (!otherFunctionOp) {
      return mlir::failure();
    }

    size_t variablePos = 0;

    for (VariableOp variableOp : otherFunctionOp.getVariables()) {
      if (variableOp.isInput()) {
        pos[variableOp.getSymName()] = variablePos++;
      }
    }

    return mlir::success();
  }

  bool containsArgNames(const llvm::StringMap<size_t> &parent,
                        const llvm::StringMap<size_t> &child) const {
    return all_of(child,
                  [&](auto &entry) { return parent.contains(entry.getKey()); });
  }

  bool haveSameArgNames(const llvm::StringMap<size_t> &first,
                        const llvm::StringMap<size_t> &second) const {
    return containsArgNames(first, second) && containsArgNames(second, first);
  }

  bool compareNamedUnnamedArgs(
      mlir::ValueRange namedArgs, const llvm::StringMap<size_t> &namedArgsPos,
      mlir::ValueRange unnamedArgs,
      const llvm::StringMap<size_t> &unnamedArgsPos,
      const bool invertValueOrder,
      llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
          &inductionAccesses,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    if (namedArgs.size() != unnamedArgs.size()) {
      return false;
    }

    llvm::DenseMap<size_t, std::string> inverseUnnamedArgsPos;

    for (auto &entry : unnamedArgsPos) {
      inverseUnnamedArgsPos[entry.getValue()] = entry.getKey().str();
    }

    for (auto unnamedArg : llvm::enumerate(unnamedArgs)) {
      auto inverseUnnamedArgPosIt =
          inverseUnnamedArgsPos.find(unnamedArg.index());

      if (inverseUnnamedArgPosIt == inverseUnnamedArgsPos.end()) {
        return false;
      }

      auto namedArgsPosIt =
          namedArgsPos.find(inverseUnnamedArgPosIt->getSecond());

      if (namedArgsPosIt == namedArgsPos.end()) {
        return false;
      }

      assert(namedArgsPosIt->getValue() < namedArgs.size());
      mlir::Value namedArg = namedArgs[namedArgsPosIt->getValue()];

      // To preserve the association of inductionAccesses to each original
      // operation we might have to reorder the operand checking order.
      if (!invertValueOrder) {
        if (!areExpressionOperandsEquivalent(namedArg, unnamedArg.value(),
                                             inductionAccesses,
                                             symbolTableCollection)) {
          return false;
        }
      } else {
        if (!areExpressionOperandsEquivalent(unnamedArg.value(), namedArg,
                                             inductionAccesses,
                                             symbolTableCollection)) {
          return false;
        }
      }
    }

    return true;
  }
};

struct EquationSidesOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::EquationSidesOpInterface, EquationSidesOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<EquationSidesOp>(op);

    os << "{";

    llvm::interleaveComma(castedOp.getLhsValues(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << "} = {";

    llvm::interleaveComma(castedOp.getRhsValues(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << "}";
  }
};

struct TensorFromElementsOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::TensorFromElementsOpInterface, TensorFromElementsOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<TensorFromElementsOp>(op);

    os << "{";

    llvm::interleaveComma(castedOp.getValues(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << "}";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(TensorFromElementsOp)
};

struct TensorBroadcastOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::TensorBroadcastOpInterface, TensorBroadcastOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<TensorBroadcastOp>(op);

    os << "{";
    mlir::TensorType tensorType = castedOp.getResult().getType();

    for (int64_t i = 0, e = tensorType.getNumElements(); i < e; ++i) {
      if (i != 0) {
        os << ", ";
      }

      ::printExpression(os, castedOp.getValue(), inductions);
    }

    os << "}";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(TensorBroadcastOp)
};

struct TensorViewOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::TensorViewOpInterface, TensorViewOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<TensorViewOp>(op);

    ::printExpression(os, castedOp.getSource(), inductions);
    os << "[";

    llvm::interleaveComma(
        castedOp.getSubscriptions(), os,
        [&](mlir::Value exp) { ::printExpression(os, exp, inductions); });

    os << "]";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(TensorViewOp)

  mlir::LogicalResult getEquationAccesses(
      mlir::Operation *op, llvm::SmallVectorImpl<VariableAccess> &accesses,
      mlir::SymbolTableCollection &symbolTable,
      llvm::DenseMap<mlir::Value, unsigned int> &explicitInductionsPositionMap,
      AdditionalInductions &additionalInductions,
      llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>
          &dimensionAccesses,
      EquationPath path) const {
    auto castedOp = mlir::cast<TensorViewOp>(op);
    auto indices = castedOp.getSubscriptions();

    for (size_t i = 0, e = indices.size(); i < e; ++i) {
      mlir::Value index = indices[e - 1 - i];

      auto dimensionAccess = getDimensionAccess(explicitInductionsPositionMap,
                                                additionalInductions, index);

      if (!dimensionAccess) {
        return mlir::failure();
      }

      dimensionAccesses.push_back(std::move(dimensionAccess));
    }

    auto sourceOp =
        castedOp.getSource().getDefiningOp<EquationExpressionOpInterface>();

    if (!sourceOp) {
      return mlir::failure();
    }

    return sourceOp.getEquationAccesses(
        accesses, symbolTable, explicitInductionsPositionMap,
        additionalInductions, dimensionAccesses, std::move(path));
  }
};

struct TensorExtractOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::TensorExtractOpInterface, TensorExtractOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<TensorExtractOp>(op);

    ::printExpression(os, castedOp.getTensor(), inductions);
    os << "[";

    llvm::interleaveComma(castedOp.getIndices(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << "]";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(TensorExtractOp)

  mlir::LogicalResult getEquationAccesses(
      mlir::Operation *op, llvm::SmallVectorImpl<VariableAccess> &accesses,
      mlir::SymbolTableCollection &symbolTable,
      llvm::DenseMap<mlir::Value, unsigned int> &explicitInductionsPositionMap,
      AdditionalInductions &additionalInductions,
      llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>
          &dimensionAccesses,
      EquationPath path) const {
    auto castedOp = mlir::cast<TensorExtractOp>(op);
    auto indices = castedOp.getIndices();

    for (size_t i = 0, e = indices.size(); i < e; ++i) {
      mlir::Value index = indices[e - 1 - i];

      auto dimensionAccess = getDimensionAccess(explicitInductionsPositionMap,
                                                additionalInductions, index);

      if (!dimensionAccess) {
        return mlir::failure();
      }

      dimensionAccesses.push_back(std::move(dimensionAccess));
    }

    auto tensorOp =
        castedOp.getTensor().getDefiningOp<EquationExpressionOpInterface>();

    if (!tensorOp) {
      return mlir::failure();
    }

    return tensorOp.getEquationAccesses(
        accesses, symbolTable, explicitInductionsPositionMap,
        additionalInductions, dimensionAccesses, std::move(path));
  }
};

struct TensorInsertOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::TensorInsertOpInterface, TensorInsertOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<TensorInsertOp>(op);

    ::printExpression(os, castedOp.getValue(), inductions);
    os << " into ";
    ::printExpression(os, castedOp.getDestination(), inductions);
    os << "[";

    llvm::interleaveComma(castedOp.getIndices(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << "]";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(TensorInsertOp)
};

struct ArrayFromElementsOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::ArrayFromElementsOpInterface, ArrayFromElementsOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ArrayFromElementsOp>(op);

    os << "{";

    llvm::interleaveComma(castedOp.getValues(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << "}";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(ArrayFromElementsOp)
};

struct ArrayBroadcastOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::ArrayBroadcastOpInterface, ArrayBroadcastOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ArrayBroadcastOp>(op);

    os << "{";

    for (int64_t i = 0, e = castedOp.getArrayType().getNumElements(); i < e;
         ++i) {
      if (i != 0) {
        os << ", ";
      }

      ::printExpression(os, castedOp.getValue(), inductions);
    }

    os << "}";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(ArrayBroadcastOp)
};

struct ArrayCastOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::ArrayCastOpInterface, ArrayCastOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ArrayCastOp>(op);
    ::printExpression(os, castedOp.getOperand(), inductions);
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(ArrayCastOp)

  mlir::LogicalResult getEquationAccesses(
      mlir::Operation *op, llvm::SmallVectorImpl<VariableAccess> &accesses,
      mlir::SymbolTableCollection &symbolTable,
      llvm::DenseMap<mlir::Value, unsigned int> &explicitInductionsPositionMap,
      AdditionalInductions &additionalInductions,
      llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>
          &dimensionAccesses,
      EquationPath path) const {
    auto castedOp = mlir::cast<ArrayCastOp>(op);
    mlir::Value source = castedOp.getSource();
    auto childOp = source.getDefiningOp();

    if (!childOp) {
      return mlir::success();
    }

    auto expressionInt = mlir::dyn_cast<EquationExpressionOpInterface>(childOp);

    if (!expressionInt) {
      return mlir::failure();
    }

    if (mlir::failed(expressionInt.getEquationAccesses(
            accesses, symbolTable, explicitInductionsPositionMap,
            additionalInductions, dimensionAccesses, path + 0))) {
      return mlir::failure();
    }

    return mlir::success();
  }
};

struct DimOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::DimOpInterface,
                                                          DimOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<DimOp>(op);

    os << "dim(";
    ::printExpression(os, castedOp.getArray(), inductions);
    os << ", ";
    ::printExpression(os, castedOp.getDimension(), inductions);
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(DimOp)

  uint64_t getNumOfExpressionElements(mlir::Operation *op) const { return 1; }

  mlir::Value getExpressionElement(mlir::Operation *op,
                                   const uint64_t position) const {
    auto castedOp = mlir::cast<DimOp>(op);
    assert(position == 0);
    return castedOp.getDimension();
  }
};

struct SubscriptionOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::SubscriptionOpInterface, SubscriptionOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<SubscriptionOp>(op);

    ::printExpression(os, castedOp.getSource(), inductions);
    os << "[";

    llvm::interleaveComma(castedOp.getIndices(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << "]";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(SubscriptionOp)

  mlir::LogicalResult getEquationAccesses(
      mlir::Operation *op, llvm::SmallVectorImpl<VariableAccess> &accesses,
      mlir::SymbolTableCollection &symbolTable,
      llvm::DenseMap<mlir::Value, unsigned int> &explicitInductionsPositionMap,
      AdditionalInductions &additionalInductions,
      llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>
          &dimensionAccesses,
      EquationPath path) const {
    auto castedOp = mlir::cast<SubscriptionOp>(op);
    auto indices = castedOp.getIndices();

    for (size_t i = 0, e = indices.size(); i < e; ++i) {
      mlir::Value index = indices[e - 1 - i];

      auto dimensionAccess = getDimensionAccess(explicitInductionsPositionMap,
                                                additionalInductions, index);

      if (!dimensionAccess) {
        return mlir::failure();
      }

      dimensionAccesses.push_back(std::move(dimensionAccess));
    }

    auto sourceOp =
        castedOp.getSource().getDefiningOp<EquationExpressionOpInterface>();

    if (!sourceOp) {
      return mlir::failure();
    }

    return sourceOp.getEquationAccesses(
        accesses, symbolTable, explicitInductionsPositionMap,
        additionalInductions, dimensionAccesses, std::move(path));
  }
};

struct LoadOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::LoadOpInterface,
                                                          LoadOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<LoadOp>(op);

    ::printExpression(os, castedOp.getArray(), inductions);
    os << "[";

    llvm::interleaveComma(castedOp.getIndices(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << "]";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(LoadOp)

  mlir::LogicalResult getEquationAccesses(
      mlir::Operation *op, llvm::SmallVectorImpl<VariableAccess> &accesses,
      mlir::SymbolTableCollection &symbolTable,
      llvm::DenseMap<mlir::Value, unsigned int> &explicitInductionsPositionMap,
      AdditionalInductions &additionalInductions,
      llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>
          &dimensionAccesses,
      EquationPath path) const {
    auto castedOp = mlir::cast<LoadOp>(op);
    auto indices = castedOp.getIndices();

    for (size_t i = 0, e = indices.size(); i < e; ++i) {
      mlir::Value index = indices[e - 1 - i];

      auto dimensionAccess = getDimensionAccess(explicitInductionsPositionMap,
                                                additionalInductions, index);

      if (!dimensionAccess) {
        return mlir::failure();
      }

      dimensionAccesses.push_back(std::move(dimensionAccess));
    }

    auto arrayOp =
        castedOp.getArray().getDefiningOp<EquationExpressionOpInterface>();

    if (!arrayOp) {
      return mlir::failure();
    }

    return arrayOp.getEquationAccesses(
        accesses, symbolTable, explicitInductionsPositionMap,
        additionalInductions, dimensionAccesses, std::move(path));
  }
};

struct VariableGetOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::VariableGetOpInterface, VariableGetOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<VariableGetOp>(op);
    os << castedOp.getVariable();
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
          &inductionAccesses,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto casted = mlir::cast<VariableGetOp>(op);
    auto otherCasted = mlir::dyn_cast<VariableGetOp>(other);

    if (!otherCasted) {
      return false;
    }

    if (casted.getVariable() != otherCasted.getVariable()) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, other, inductionAccesses,
                                            symbolTableCollection);
  }

  mlir::LogicalResult getEquationAccesses(
      mlir::Operation *op, llvm::SmallVectorImpl<VariableAccess> &accesses,
      mlir::SymbolTableCollection &symbolTable,
      llvm::DenseMap<mlir::Value, unsigned int> &explicitInductionsPositionMap,
      AdditionalInductions &additionalInductions,
      llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>
          &dimensionAccesses,
      EquationPath path) const {
    auto castedOp = mlir::cast<VariableGetOp>(op);

    // Reverse the dimension accesses.
    llvm::SmallVector<std::unique_ptr<DimensionAccess>, 10> reverted;

    for (size_t i = 0, e = dimensionAccesses.size(); i < e; ++i) {
      reverted.push_back(dimensionAccesses[e - i - 1]->clone());
    }

    // Finalize the accesses.
    auto numOfInductions =
        static_cast<uint64_t>(explicitInductionsPositionMap.size());

    if (auto tensorType = mlir::dyn_cast<mlir::TensorType>(castedOp.getType());
        tensorType &&
        tensorType.getRank() > static_cast<int64_t>(reverted.size())) {
      // Access to each scalar variable.
      for (int64_t i = static_cast<int64_t>(reverted.size()),
                   rank = tensorType.getRank();
           i < rank; ++i) {
        const int64_t dimension = tensorType.getDimSize(i);
        assert(dimension != mlir::ShapedType::kDynamic);

        reverted.push_back(std::make_unique<DimensionAccessRange>(
            castedOp.getContext(), Range(0, dimension)));
      }
    }

    accesses.push_back(VariableAccess(
        std::move(path), mlir::SymbolRefAttr::get(castedOp.getVariableAttr()),
        AccessFunction::build(castedOp.getContext(), numOfInductions,
                              reverted)));

    return mlir::success();
  }
};

struct GlobalVariableGetOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::GlobalVariableGetOpInterface, GlobalVariableGetOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<GlobalVariableGetOp>(op);
    os << castedOp.getVariable();
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
          &inductionAccesses,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto casted = mlir::cast<GlobalVariableGetOp>(op);
    auto otherCasted = mlir::dyn_cast<GlobalVariableGetOp>(other);

    if (!otherCasted) {
      return false;
    }

    if (casted.getVariable() != otherCasted.getVariable()) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, other, inductionAccesses,
                                            symbolTableCollection);
  }

  mlir::LogicalResult getEquationAccesses(
      mlir::Operation *op, llvm::SmallVectorImpl<VariableAccess> &accesses,
      mlir::SymbolTableCollection &symbolTable,
      llvm::DenseMap<mlir::Value, unsigned int> &explicitInductionsPositionMap,
      AdditionalInductions &additionalInductions,
      llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>
          &dimensionAccesses,
      EquationPath path) const {
    return mlir::success();
  }
};

struct ConstantOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::ConstantOpInterface,
                                                          ConstantOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ConstantOp>(op);

    if (auto boolAttr = mlir::dyn_cast<BooleanAttr>(castedOp.getValue())) {
      os << (boolAttr.getValue() ? "true" : "false");
      return;
    }

    if (auto integerAttr = mlir::dyn_cast<IntegerAttr>(castedOp.getValue())) {
      os << integerAttr.getValue();
      return;
    }

    if (auto realAttr = mlir::dyn_cast<RealAttr>(castedOp.getValue())) {
      os << realAttr.getValue().convertToDouble();
      return;
    }

    if (auto integerAttr =
            mlir::dyn_cast<mlir::IntegerAttr>(castedOp.getValue())) {
      os << integerAttr.getValue();
      return;
    }

    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(castedOp.getValue())) {
      os << floatAttr.getValueAsDouble();
      return;
    }

    castedOp.getValue().print(os, true);
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
          &inductionAccesses,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto casted = mlir::cast<ConstantOp>(op);
    auto otherCasted = mlir::dyn_cast<ConstantOp>(other);

    if (!otherCasted) {
      return false;
    }

    if (casted.getValue() != otherCasted.getValue()) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, other, inductionAccesses,
                                            symbolTableCollection);
  }
};

struct NegateOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::NegateOpInterface,
                                                          NegateOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<NegateOp>(op);

    os << "(- ";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(NegateOp)
};

struct AddOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::AddOpInterface,
                                                          AddOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<AddOp>(op, os, inductions, "+");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(AddOp)
};

struct AddEWOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::AddEWOpInterface,
                                                          AddEWOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<AddEWOp>(op, os, inductions, ".+");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(AddEWOp)
};

struct SubOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::SubOpInterface,
                                                          SubOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<SubOp>(op, os, inductions, "-");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(SubOp)
};

struct SubEWOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::SubEWOpInterface,
                                                          SubEWOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<SubEWOp>(op, os, inductions, ".-");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(SubEWOp)
};

struct MulOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::MulOpInterface,
                                                          MulOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<MulOp>(op, os, inductions, "*");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(MulOp)
};

struct MulEWOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::MulEWOpInterface,
                                                          MulEWOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<MulEWOp>(op, os, inductions, ".*");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(MulEWOp)
};

struct DivOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::DivOpInterface,
                                                          DivOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<DivOp>(op, os, inductions, "/");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(DivOp)
};

struct DivEWOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::DivEWOpInterface,
                                                          DivEWOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<DivEWOp>(op, os, inductions, "./");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(DivEWOp)
};

struct PowOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::PowOpInterface,
                                                          PowOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<PowOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getBase(), inductions);
    os << " ^ ";
    ::printExpression(os, castedOp.getExponent(), inductions);
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(PowOp)
};

struct PowEWOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::PowEWOpInterface,
                                                          PowEWOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<PowEWOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getBase(), inductions);
    os << " .^ ";
    ::printExpression(os, castedOp.getExponent(), inductions);
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(PowEWOp)
};

struct EqOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::EqOpInterface,
                                                          EqOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<EqOp>(op, os, inductions, "==");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(EqOp)
};

struct NotEqOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::NotEqOpInterface,
                                                          NotEqOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<NotEqOp>(op, os, inductions, "!=");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(NotEqOp)
};

struct GtOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::GtOpInterface,
                                                          GtOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<GtOp>(op, os, inductions, ">");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(GtOp)
};

struct GteOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::GteOpInterface,
                                                          GteOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<GteOp>(op, os, inductions, ">=");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(GteOp)
};

struct LtOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::LtOpInterface,
                                                          LtOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<LtOp>(op, os, inductions, "<");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(LtOp)
};

struct LteOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::LteOpInterface,
                                                          LteOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<LteOp>(op, os, inductions, "<=");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(LteOp)
};

struct NotOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::NotOpInterface,
                                                          NotOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<NotOp>(op, os, inductions, "!");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(NotOp)
};

struct AndOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::AndOpInterface,
                                                          AndOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<AndOp>(op, os, inductions, "&&");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(AndOp)
};

struct OrOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::OrOpInterface,
                                                          OrOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printBinaryExpression<OrOp>(op, os, inductions, "||");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(OrOp)
};

struct SelectOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::SelectOpInterface,
                                                          SelectOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<SelectOp>(op);

    ::printExpression(os, castedOp.getCondition(), inductions);
    os << " ? (";

    llvm::interleaveComma(castedOp.getTrueValues(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << ") : (";

    llvm::interleaveComma(castedOp.getFalseValues(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(SelectOp)
};

struct AbsOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::AbsOpInterface,
                                                          AbsOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<AbsOp>(op, os, inductions, "abs");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(AbsOp)
};

struct AcosOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::AcosOpInterface,
                                                          AcosOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<AcosOp>(op, os, inductions, "acos");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(AcosOp)
};

struct AsinOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::AsinOpInterface,
                                                          AsinOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<AsinOp>(op, os, inductions, "asin");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(AsinOp)
};

struct AtanOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::AtanOpInterface,
                                                          AtanOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<AtanOp>(op, os, inductions, "atan");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(AtanOp)
};

struct Atan2OpInterface
    : public EquationExpressionOpInterface::ExternalModel<::Atan2OpInterface,
                                                          Atan2Op> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<Atan2Op>(op);

    os << "atan2(";
    ::printExpression(os, castedOp.getY(), inductions);
    os << ", ";
    ::printExpression(os, castedOp.getX(), inductions);
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(Atan2Op)
};

struct CeilOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::CeilOpInterface,
                                                          CeilOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<CeilOp>(op, os, inductions, "ceil");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(CeilOp)
};

struct CosOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::CosOpInterface,
                                                          CosOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<CosOp>(op, os, inductions, "cos");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(CosOp)
};

struct CoshOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::CoshOpInterface,
                                                          CoshOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<CoshOp>(op, os, inductions, "cosh");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(CoshOp)
};

struct DiagonalOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::DiagonalOpInterface,
                                                          DiagonalOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<DiagonalOp>(op, os, inductions, "diagonal");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(DiagonalOp)
};

struct DivTruncOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::DivTruncOpInterface,
                                                          DivTruncOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<DivTruncOp>(op);

    os << "div(";
    ::printExpression(os, castedOp.getX(), inductions);
    os << ", ";
    ::printExpression(os, castedOp.getY(), inductions);
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(DivTruncOp)
};

struct ExpOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::ExpOpInterface,
                                                          ExpOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<ExpOp>(op, os, inductions, "exp");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(ExpOp)
};

struct FillOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::FillOpInterface,
                                                          FillOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<FillOp>(op, os, inductions, "fill");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(FillOp)
};

struct FloorOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::FloorOpInterface,
                                                          FloorOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<FloorOp>(op, os, inductions, "floor");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(FloorOp)
};

struct IdentityOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::IdentityOpInterface,
                                                          IdentityOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<IdentityOp>(op, os, inductions, "identity");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(IdentityOp)
};

struct IntegerOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::IntegerOpInterface,
                                                          IntegerOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<IntegerOp>(op, os, inductions, "integer");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(IntegerOp)
};

struct LinspaceOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::LinspaceOpInterface,
                                                          LinspaceOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<LinspaceOp>(op);

    os << "linspace(";
    ::printExpression(os, castedOp.getBegin(), inductions);
    os << ", ";
    ::printExpression(os, castedOp.getEnd(), inductions);
    os << ", ";
    ::printExpression(os, castedOp.getAmount(), inductions);
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(LinspaceOp)
};

struct LogOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::LogOpInterface,
                                                          LogOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<LogOp>(op, os, inductions, "log");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(LogOp)
};

struct Log10OpInterface
    : public EquationExpressionOpInterface::ExternalModel<::Log10OpInterface,
                                                          Log10Op> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<Log10Op>(op, os, inductions, "log10");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(Log10Op)
};

struct MaxOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::MaxOpInterface,
                                                          MaxOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<MaxOp>(op);

    os << "max(";
    ::printExpression(os, castedOp.getFirst(), inductions);

    if (mlir::Value second = castedOp.getSecond()) {
      os << ", ";
      ::printExpression(os, second, inductions);
    }

    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(MaxOp)
};

struct MinOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::MinOpInterface,
                                                          MinOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<MinOp>(op);

    os << "min(";
    ::printExpression(os, castedOp.getFirst(), inductions);

    if (mlir::Value second = castedOp.getSecond()) {
      os << ", ";
      ::printExpression(os, second, inductions);
    }

    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(MinOp)
};

struct ModOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::ModOpInterface,
                                                          ModOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ModOp>(op);

    os << "mod(";
    ::printExpression(os, castedOp.getX(), inductions);
    os << ", ";
    ::printExpression(os, castedOp.getY(), inductions);
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(ModOp)
};

struct NDimsOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::NDimsOpInterface,
                                                          NDimsOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<NDimsOp>(op);

    os << "ndims(";
    ::printExpression(os, castedOp.getArray(), inductions);
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(NDimsOp)
};

struct OnesOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::OnesOpInterface,
                                                          OnesOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<OnesOp>(op);

    os << "ones(";

    llvm::interleaveComma(castedOp.getSizes(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(OnesOp)
};

struct ProductOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::ProductOpInterface,
                                                          ProductOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ProductOp>(op);

    os << "product(";
    ::printExpression(os, castedOp.getArray(), inductions);
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(ProductOp)
};

struct RemOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::RemOpInterface,
                                                          RemOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<RemOp>(op);

    os << "rem(";
    ::printExpression(os, castedOp.getX(), inductions);
    os << ", ";
    ::printExpression(os, castedOp.getY(), inductions);
    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(RemOp)
};

struct SignOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::SignOpInterface,
                                                          SignOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<SignOp>(op, os, inductions, "sign");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(SignOp)
};

struct SinOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::SinOpInterface,
                                                          SinOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<SinOp>(op, os, inductions, "sin");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(SinOp)
};

struct SinhOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::SinhOpInterface,
                                                          SinhOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<SinhOp>(op, os, inductions, "sinh");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(SinhOp)
};

struct SizeOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::SizeOpInterface,
                                                          SizeOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<SizeOp>(op);

    os << "size(";
    ::printExpression(os, castedOp.getArray(), inductions);

    if (mlir::Value dimension = castedOp.getDimension()) {
      os << ", ";
      ::printExpression(os, dimension, inductions);
    }

    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(SizeOp)
};

struct SqrtOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::SqrtOpInterface,
                                                          SqrtOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<SqrtOp>(op, os, inductions, "sqrt");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(SqrtOp)
};

struct SumOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::SumOpInterface,
                                                          SumOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<SumOp>(op, os, inductions, "sum");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(SumOp)
};

struct SymmetricOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::SymmetricOpInterface, SymmetricOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<SymmetricOp>(op, os, inductions, "symmetric");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(SymmetricOp)
};

struct TanOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::TanOpInterface,
                                                          TanOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<TanOp>(op, os, inductions, "tan");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(TanOp)
};

struct TanhOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::TanhOpInterface,
                                                          TanhOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<TanhOp>(op, os, inductions, "tanh");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(TanhOp)
};

struct TransposeOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ::TransposeOpInterface, TransposeOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<TransposeOp>(op, os, inductions, "transpose");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(TransposeOp)
};

struct ZerosOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::ZerosOpInterface,
                                                          ZerosOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ZerosOp>(op);

    os << "zeros(";

    llvm::interleaveComma(castedOp.getSizes(), os, [&](mlir::Value exp) {
      ::printExpression(os, exp, inductions);
    });

    os << ")";
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(ZerosOp)
};

struct DerOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::DerOpInterface,
                                                          DerOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    ::printUnaryExpression<DerOp>(op, os, inductions, "der");
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(DerOp)
};

struct TimeOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::TimeOpInterface,
                                                          TimeOp> {
  void printExpression(mlir::Operation *, llvm::raw_ostream &os,
                       const llvm::DenseMap<mlir::Value, int64_t> &) const {
    os << "time";
  }

  bool isEquivalent(
      mlir::Operation *, mlir::Operation *other,
      llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
          &,
      mlir::SymbolTableCollection &) const {
    return mlir::isa<TimeOp>(other);
  }
};

struct CastOpInterface
    : public EquationExpressionOpInterface::ExternalModel<::CastOpInterface,
                                                          CastOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto casted = mlir::cast<CastOp>(op);
    ::printExpression(os, casted.getValue(), inductions);
  }

  DEFINE_DEFAULT_IS_EQUIVALENT(CastOp)
};
} // namespace

namespace mlir::bmodelica {
void registerEquationExpressionOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context,
                            BaseModelicaDialect *dialect) {
    // clang-format off
    // Equation root.
    EquationSidesOp::attachInterface<::EquationSidesOpInterface>(*context);

    // Tensor operations.
    TensorFromElementsOp::attachInterface<::TensorFromElementsOpInterface>(*context);
    TensorBroadcastOp::attachInterface<::TensorBroadcastOpInterface>(*context);
    TensorViewOp::attachInterface<::TensorViewOpInterface>(*context);
    TensorExtractOp::attachInterface<::TensorExtractOpInterface>(*context);
    TensorInsertOp::attachInterface<::TensorInsertOpInterface>(*context);

    // Array operations.
    ArrayFromElementsOp::attachInterface<::ArrayFromElementsOpInterface>(*context);
    ArrayBroadcastOp::attachInterface<::ArrayBroadcastOpInterface>(*context);
    ArrayCastOp::attachInterface<::ArrayCastOpInterface>(*context);
    DimOp::attachInterface<::DimOpInterface>(*context);
    SubscriptionOp::attachInterface<::SubscriptionOpInterface>(*context);
    LoadOp::attachInterface<::LoadOpInterface>(*context);

    // Variable operations.
    VariableGetOp::attachInterface<::VariableGetOpInterface>(*context);
    GlobalVariableGetOp::attachInterface<::GlobalVariableGetOpInterface>(*context);

    // Math operations.
    ConstantOp::attachInterface<::ConstantOpInterface>(*context);
    NegateOp::attachInterface<::NegateOpInterface>(*context);
    AddOp::attachInterface<::AddOpInterface>(*context);
    AddEWOp::attachInterface<::AddEWOpInterface>(*context);
    SubOp::attachInterface<::SubOpInterface>(*context);
    SubEWOp::attachInterface<::SubEWOpInterface>(*context);
    MulOp::attachInterface<::MulOpInterface>(*context);
    MulEWOp::attachInterface<::MulEWOpInterface>(*context);
    DivOp::attachInterface<::DivOpInterface>(*context);
    DivEWOp::attachInterface<::DivEWOpInterface>(*context);
    PowOp::attachInterface<::PowOpInterface>(*context);
    PowEWOp::attachInterface<::PowEWOpInterface>(*context);

    // Comparison operations.
    EqOp::attachInterface<::EqOpInterface>(*context);
    NotEqOp::attachInterface<::NotEqOpInterface>(*context);
    GtOp::attachInterface<::GtOpInterface>(*context);
    GteOp::attachInterface<::GteOpInterface>(*context);
    LtOp::attachInterface<::LtOpInterface>(*context);
    LteOp::attachInterface<::LteOpInterface>(*context);

    // Logic operations.
    NotOp::attachInterface<::NotOpInterface>(*context);
    AndOp::attachInterface<::AndOpInterface>(*context);
    OrOp::attachInterface<::OrOpInterface>(*context);
    SelectOp::attachInterface<::SelectOpInterface>(*context);

    // Built-in operations
    AbsOp::attachInterface<::AbsOpInterface>(*context);
    AcosOp::attachInterface<::AcosOpInterface>(*context);
    AsinOp::attachInterface<::AsinOpInterface>(*context);
    AtanOp::attachInterface<::AtanOpInterface>(*context);
    Atan2Op::attachInterface<::Atan2OpInterface>(*context);
    CeilOp::attachInterface<::CeilOpInterface>(*context);
    CosOp::attachInterface<::CosOpInterface>(*context);
    CoshOp::attachInterface<::CoshOpInterface>(*context);
    DiagonalOp::attachInterface<::DiagonalOpInterface>(*context);
    DivTruncOp::attachInterface<::DivTruncOpInterface>(*context);
    ExpOp::attachInterface<::ExpOpInterface>(*context);
    FillOp::attachInterface<::FillOpInterface>(*context);
    FloorOp::attachInterface<::FloorOpInterface>(*context);
    IdentityOp::attachInterface<::IdentityOpInterface>(*context);
    IntegerOp::attachInterface<::IntegerOpInterface>(*context);
    LinspaceOp::attachInterface<::LinspaceOpInterface>(*context);
    LogOp::attachInterface<::LogOpInterface>(*context);
    Log10Op::attachInterface<::Log10OpInterface>(*context);
    MaxOp::attachInterface<::MaxOpInterface>(*context);
    MinOp::attachInterface<::MinOpInterface>(*context);
    ModOp::attachInterface<::ModOpInterface>(*context);
    NDimsOp::attachInterface<::NDimsOpInterface>(*context);
    OnesOp::attachInterface<::OnesOpInterface>(*context);
    ProductOp::attachInterface<::ProductOpInterface>(*context);
    RemOp::attachInterface<::RemOpInterface>(*context);
    SignOp::attachInterface<::SignOpInterface>(*context);
    SinOp::attachInterface<::SinOpInterface>(*context);
    SinhOp::attachInterface<::SinhOpInterface>(*context);
    SizeOp::attachInterface<::SizeOpInterface>(*context);
    SqrtOp::attachInterface<::SqrtOpInterface>(*context);
    SumOp::attachInterface<::SumOpInterface>(*context);
    SymmetricOp::attachInterface<::SymmetricOpInterface>(*context);
    TanOp::attachInterface<::TanOpInterface>(*context);
    TanhOp::attachInterface<::TanhOpInterface>(*context);
    TransposeOp::attachInterface<::TransposeOpInterface>(*context);
    ZerosOp::attachInterface<::ZerosOpInterface>(*context);

    // Various operations.
    ReductionOp::attachInterface<::ReductionOpInterface>(*context);
    RangeOp::attachInterface<::RangeOpInterface>(*context);
    DerOp::attachInterface<::DerOpInterface>(*context);
    TimeOp::attachInterface<::TimeOpInterface>(*context);
    CallOp::attachInterface<::CallOpInterface>(*context);
    CastOp::attachInterface<::CastOpInterface>(*context);
    // clang-format on
  });
}
} // namespace mlir::bmodelica
