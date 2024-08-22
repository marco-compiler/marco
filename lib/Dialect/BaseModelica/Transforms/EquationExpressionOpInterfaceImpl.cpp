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

bool areExpressionOperandsEquivalent(
    mlir::ValueRange firstOperands, mlir::ValueRange secondOperands,
    mlir::SymbolTableCollection &symbolTableCollection) {
  if (firstOperands.size() != secondOperands.size()) {
    return false;
  }

  for (auto [firstOperand, secondOperand] :
       llvm::zip(firstOperands, secondOperands)) {
    auto firstExp = firstOperand.getDefiningOp<EquationExpressionOpInterface>();
    auto secondExp =
        secondOperand.getDefiningOp<EquationExpressionOpInterface>();

    if (!firstExp || !secondExp) {
      return false;
    }

    return firstExp.isEquivalent(secondExp, symbolTableCollection);
  }

  return true;
}

bool areEquationExpressionsEquivalent(
    mlir::Operation *firstOp, mlir::Operation *secondOp,
    mlir::SymbolTableCollection &symbolTableCollection) {
  if (firstOp->getResultTypes() != secondOp->getResultTypes()) {
    return false;
  }

  return ::areExpressionOperandsEquivalent(
      firstOp->getOperands(), secondOp->getOperands(), symbolTableCollection);
}
} // namespace

namespace {
struct EquationSidesOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          EquationSidesOpInterface, EquationSidesOp> {
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
          TensorFromElementsOpInterface, TensorFromElementsOp> {
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<TensorFromElementsOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct TensorBroadcastOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          TensorBroadcastOpInterface, TensorBroadcastOp> {
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<TensorBroadcastOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct TensorViewOpInterface
    : public EquationExpressionOpInterface::ExternalModel<TensorViewOpInterface,
                                                          TensorViewOp> {
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<TensorViewOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
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
          TensorExtractOpInterface, TensorExtractOp> {
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<TensorExtractOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
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

struct ArrayFromElementsOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ArrayFromElementsOpInterface, ArrayFromElementsOp> {
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<ArrayFromElementsOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct ArrayBroadcastOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          ArrayBroadcastOpInterface, ArrayBroadcastOp> {
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<ArrayBroadcastOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct ArrayCastOpInterface
    : public EquationExpressionOpInterface::ExternalModel<ArrayCastOpInterface,
                                                          ArrayCastOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ArrayCastOp>(op);
    ::printExpression(os, castedOp.getOperand(), inductions);
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<ArrayCastOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
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
    : public EquationExpressionOpInterface::ExternalModel<DimOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<DimOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }

  uint64_t getNumOfExpressionElements(mlir::Operation *op) const { return 1; }

  mlir::Value getExpressionElement(mlir::Operation *op,
                                   uint64_t position) const {
    auto castedOp = mlir::cast<DimOp>(op);
    assert(position == 0);
    return castedOp.getDimension();
  }
};

struct SubscriptionOpInterface
    : public EquationExpressionOpInterface::ExternalModel<
          SubscriptionOpInterface, SubscriptionOp> {
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<SubscriptionOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
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
    : public EquationExpressionOpInterface::ExternalModel<LoadOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<LoadOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
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
          VariableGetOpInterface, VariableGetOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<VariableGetOp>(op);
    os << castedOp.getVariable();
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto casted = mlir::cast<VariableGetOp>(op);
    auto otherCasted = mlir::dyn_cast<VariableGetOp>(other);

    if (!otherCasted) {
      return false;
    }

    if (casted.getVariable() != otherCasted.getVariable()) {
      return false;
    }

    return ::areEquationExpressionsEquivalent(op, other, symbolTableCollection);
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

    if (auto tensorType = castedOp.getType().dyn_cast<mlir::TensorType>();
        tensorType &&
        tensorType.getRank() > static_cast<int64_t>(reverted.size())) {
      // Access to each scalar variable.
      for (int64_t i = static_cast<int64_t>(reverted.size()),
                   rank = tensorType.getRank();
           i < rank; ++i) {
        int64_t dimension = tensorType.getDimSize(i);
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
          GlobalVariableGetOpInterface, GlobalVariableGetOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<GlobalVariableGetOp>(op);
    os << castedOp.getVariable();
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto casted = mlir::cast<GlobalVariableGetOp>(op);
    auto otherCasted = mlir::dyn_cast<GlobalVariableGetOp>(other);

    if (!otherCasted) {
      return false;
    }

    if (casted.getVariable() != otherCasted.getVariable()) {
      return false;
    }

    return ::areEquationExpressionsEquivalent(op, other, symbolTableCollection);
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
    : public EquationExpressionOpInterface::ExternalModel<ConstantOpInterface,
                                                          ConstantOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ConstantOp>(op);

    if (auto boolAttr = castedOp.getValue().dyn_cast<BooleanAttr>()) {
      os << (boolAttr.getValue() ? "true" : "false");
      return;
    }

    if (auto integerAttr = castedOp.getValue().dyn_cast<IntegerAttr>()) {
      os << integerAttr.getValue();
      return;
    }

    if (auto realAttr = castedOp.getValue().dyn_cast<RealAttr>()) {
      os << realAttr.getValue().convertToDouble();
      return;
    }

    if (auto integerAttr = castedOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
      os << integerAttr.getValue();
      return;
    }

    if (auto floatAttr = castedOp.getValue().dyn_cast<mlir::FloatAttr>()) {
      os << floatAttr.getValueAsDouble();
      return;
    }

    castedOp.getValue().print(os, true);
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto casted = mlir::cast<ConstantOp>(op);
    auto otherCasted = mlir::dyn_cast<ConstantOp>(other);

    if (!otherCasted) {
      return false;
    }

    if (casted.getValue() != otherCasted.getValue()) {
      return false;
    }

    return ::areEquationExpressionsEquivalent(op, other, symbolTableCollection);
  }
};

struct NegateOpInterface
    : public EquationExpressionOpInterface::ExternalModel<NegateOpInterface,
                                                          NegateOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<NegateOp>(op);

    os << "(- ";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<NegateOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct AddOpInterface
    : public EquationExpressionOpInterface::ExternalModel<AddOpInterface,
                                                          AddOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<AddOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " + ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<AddOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct AddEWOpInterface
    : public EquationExpressionOpInterface::ExternalModel<AddEWOpInterface,
                                                          AddEWOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<AddEWOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " .+ ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<AddEWOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct SubOpInterface
    : public EquationExpressionOpInterface::ExternalModel<SubOpInterface,
                                                          SubOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<SubOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " - ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<SubOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct SubEWOpInterface
    : public EquationExpressionOpInterface::ExternalModel<SubEWOpInterface,
                                                          SubEWOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<SubEWOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " .- ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<SubEWOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct MulOpInterface
    : public EquationExpressionOpInterface::ExternalModel<MulOpInterface,
                                                          MulOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<MulOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " * ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<MulOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct MulEWOpInterface
    : public EquationExpressionOpInterface::ExternalModel<MulEWOpInterface,
                                                          MulEWOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<MulEWOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " .* ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<MulEWOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct DivOpInterface
    : public EquationExpressionOpInterface::ExternalModel<DivOpInterface,
                                                          DivOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<DivOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " / ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<DivOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct DivEWOpInterface
    : public EquationExpressionOpInterface::ExternalModel<DivEWOpInterface,
                                                          DivEWOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<DivEWOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " ./ ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<DivEWOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct PowOpInterface
    : public EquationExpressionOpInterface::ExternalModel<PowOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<PowOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct PowEWOpInterface
    : public EquationExpressionOpInterface::ExternalModel<PowEWOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<PowEWOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct EqOpInterface
    : public EquationExpressionOpInterface::ExternalModel<EqOpInterface, EqOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<EqOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " == ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<EqOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct NotEqOpInterface
    : public EquationExpressionOpInterface::ExternalModel<NotEqOpInterface,
                                                          NotEqOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<NotEqOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " != ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<NotEqOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct GtOpInterface
    : public EquationExpressionOpInterface::ExternalModel<GtOpInterface, GtOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<GtOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " > ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<GtOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct GteOpInterface
    : public EquationExpressionOpInterface::ExternalModel<GteOpInterface,
                                                          GteOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<GteOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " >= ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<GteOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct LtOpInterface
    : public EquationExpressionOpInterface::ExternalModel<LtOpInterface, LtOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<LtOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " < ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<LtOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct LteOpInterface
    : public EquationExpressionOpInterface::ExternalModel<LteOpInterface,
                                                          LteOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<LteOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " <= ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<LteOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct NotOpInterface
    : public EquationExpressionOpInterface::ExternalModel<NotOpInterface,
                                                          NotOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<NotOp>(op);

    os << "!(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<NotOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct AndOpInterface
    : public EquationExpressionOpInterface::ExternalModel<AndOpInterface,
                                                          AndOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<AndOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " && ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<AndOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct OrOpInterface
    : public EquationExpressionOpInterface::ExternalModel<OrOpInterface, OrOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<OrOp>(op);

    os << "(";
    ::printExpression(os, castedOp.getLhs(), inductions);
    os << " || ";
    ::printExpression(os, castedOp.getRhs(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<OrOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct SelectOpInterface
    : public EquationExpressionOpInterface::ExternalModel<SelectOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<SelectOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct AbsOpInterface
    : public EquationExpressionOpInterface::ExternalModel<AbsOpInterface,
                                                          AbsOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<AbsOp>(op);

    os << "abs(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<AbsOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct AcosOpInterface
    : public EquationExpressionOpInterface::ExternalModel<AcosOpInterface,
                                                          AcosOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<AcosOp>(op);

    os << "acos(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<AcosOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct AsinOpInterface
    : public EquationExpressionOpInterface::ExternalModel<AsinOpInterface,
                                                          AsinOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<AsinOp>(op);

    os << "asin(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<AsinOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct AtanOpInterface
    : public EquationExpressionOpInterface::ExternalModel<AtanOpInterface,
                                                          AtanOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<AtanOp>(op);

    os << "atan(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<AtanOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct Atan2OpInterface
    : public EquationExpressionOpInterface::ExternalModel<Atan2OpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<Atan2Op>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct CeilOpInterface
    : public EquationExpressionOpInterface::ExternalModel<CeilOpInterface,
                                                          CeilOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<CeilOp>(op);

    os << "ceil(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<CeilOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct CosOpInterface
    : public EquationExpressionOpInterface::ExternalModel<CosOpInterface,
                                                          CosOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<CosOp>(op);

    os << "cos(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<CosOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct CoshOpInterface
    : public EquationExpressionOpInterface::ExternalModel<CoshOpInterface,
                                                          CoshOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<CoshOp>(op);

    os << "cosh(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<CoshOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct DiagonalOpInterface
    : public EquationExpressionOpInterface::ExternalModel<DiagonalOpInterface,
                                                          DiagonalOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<DiagonalOp>(op);

    os << "diagonal(";
    ::printExpression(os, castedOp.getValues(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<DiagonalOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct DivTruncOpInterface
    : public EquationExpressionOpInterface::ExternalModel<DivTruncOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<DivTruncOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct ExpOpInterface
    : public EquationExpressionOpInterface::ExternalModel<ExpOpInterface,
                                                          ExpOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ExpOp>(op);

    os << "exp(";
    ::printExpression(os, castedOp.getExponent(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<ExpOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct FillOpInterface
    : public EquationExpressionOpInterface::ExternalModel<FillOpInterface,
                                                          FillOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<FillOp>(op);

    os << "fill(";
    ::printExpression(os, castedOp.getValue(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<FillOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct FloorOpInterface
    : public EquationExpressionOpInterface::ExternalModel<FloorOpInterface,
                                                          FloorOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<FloorOp>(op);

    os << "floor(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<FloorOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct IdentityOpInterface
    : public EquationExpressionOpInterface::ExternalModel<IdentityOpInterface,
                                                          IdentityOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<IdentityOp>(op);

    os << "identity(";
    ::printExpression(os, castedOp.getSize(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<IdentityOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct IntegerOpInterface
    : public EquationExpressionOpInterface::ExternalModel<IntegerOpInterface,
                                                          IntegerOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<IntegerOp>(op);

    os << "integer(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<IntegerOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct LinspaceOpInterface
    : public EquationExpressionOpInterface::ExternalModel<LinspaceOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<LinspaceOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct LogOpInterface
    : public EquationExpressionOpInterface::ExternalModel<LogOpInterface,
                                                          LogOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<LogOp>(op);

    os << "log(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<LogOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct Log10OpInterface
    : public EquationExpressionOpInterface::ExternalModel<Log10OpInterface,
                                                          Log10Op> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<Log10Op>(op);

    os << "log10(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<Log10Op>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct MaxOpInterface
    : public EquationExpressionOpInterface::ExternalModel<MaxOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<MaxOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct MinOpInterface
    : public EquationExpressionOpInterface::ExternalModel<MinOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<MinOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct ModOpInterface
    : public EquationExpressionOpInterface::ExternalModel<ModOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<ModOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct NDimsOpInterface
    : public EquationExpressionOpInterface::ExternalModel<NDimsOpInterface,
                                                          NDimsOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<NDimsOp>(op);

    os << "ndims(";
    ::printExpression(os, castedOp.getArray(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<NDimsOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct OnesOpInterface
    : public EquationExpressionOpInterface::ExternalModel<OnesOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<OnesOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct ProductOpInterface
    : public EquationExpressionOpInterface::ExternalModel<ProductOpInterface,
                                                          ProductOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<ProductOp>(op);

    os << "product(";
    ::printExpression(os, castedOp.getArray(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<ProductOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct RemOpInterface
    : public EquationExpressionOpInterface::ExternalModel<RemOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<RemOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct SignOpInterface
    : public EquationExpressionOpInterface::ExternalModel<SignOpInterface,
                                                          SignOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<SignOp>(op);

    os << "sign(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<SignOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct SinOpInterface
    : public EquationExpressionOpInterface::ExternalModel<SinOpInterface,
                                                          SinOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<SinOp>(op);

    os << "sin(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<SinOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct SinhOpInterface
    : public EquationExpressionOpInterface::ExternalModel<SinhOpInterface,
                                                          SinhOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<SinhOp>(op);

    os << "sinh(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<SinhOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct SizeOpInterface
    : public EquationExpressionOpInterface::ExternalModel<SizeOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<SizeOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct SqrtOpInterface
    : public EquationExpressionOpInterface::ExternalModel<SqrtOpInterface,
                                                          SqrtOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<SqrtOp>(op);

    os << "sqrt(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<SqrtOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct SumOpInterface
    : public EquationExpressionOpInterface::ExternalModel<SumOpInterface,
                                                          SumOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<SumOp>(op);

    os << "sum(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<SumOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct SymmetricOpInterface
    : public EquationExpressionOpInterface::ExternalModel<SymmetricOpInterface,
                                                          SymmetricOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<SymmetricOp>(op);

    os << "symmetric(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<SymmetricOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct TanOpInterface
    : public EquationExpressionOpInterface::ExternalModel<TanOpInterface,
                                                          TanOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<TanOp>(op);

    os << "tan(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<TanOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct TanhOpInterface
    : public EquationExpressionOpInterface::ExternalModel<TanhOpInterface,
                                                          TanhOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<TanhOp>(op);

    os << "tanh(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<TanhOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct TransposeOpInterface
    : public EquationExpressionOpInterface::ExternalModel<TransposeOpInterface,
                                                          TransposeOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<TransposeOp>(op);

    os << "transpose(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<TransposeOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct ZerosOpInterface
    : public EquationExpressionOpInterface::ExternalModel<ZerosOpInterface,
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

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<ZerosOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct ReductionOpInterface
    : public EquationExpressionOpInterface::ExternalModel<ReductionOpInterface,
                                                          ReductionOp> {
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
                                          symbolTableCollection)) {
      return false;
    }

    auto yieldOp = mlir::cast<YieldOp>(casted.getBody()->getTerminator());
    auto otherYieldOp =
        mlir::cast<YieldOp>(otherCasted.getBody()->getTerminator());
    return areExpressionOperandsEquivalent(
        yieldOp.getValues(), otherYieldOp.getValues(), symbolTableCollection);
  }

  uint64_t getNumOfExpressionElements(mlir::Operation *op) const {
    auto castedOp = mlir::cast<ReductionOp>(op);

    auto terminator = mlir::cast<YieldOp>(castedOp.getBody()->getTerminator());

    return terminator.getValues().size();
  }

  mlir::Value getExpressionElement(mlir::Operation *op,
                                   uint64_t element) const {
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
         llvm::zip(castedOp.getInductions(), castedOp.getIterables())) {
      auto constantOp = iterable.getDefiningOp<ConstantOp>();

      if (!constantOp) {
        return mlir::failure();
      }

      auto iterableAttr = constantOp.getValue();

      if (auto rangeAttr = iterableAttr.dyn_cast<IntegerRangeAttr>()) {
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

      if (auto rangeAttr = iterableAttr.dyn_cast<RealRangeAttr>()) {
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

struct DerOpInterface
    : public EquationExpressionOpInterface::ExternalModel<DerOpInterface,
                                                          DerOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto castedOp = mlir::cast<DerOp>(op);

    os << "der(";
    ::printExpression(os, castedOp.getOperand(), inductions);
    os << ")";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<DerOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
};

struct TimeOpInterface
    : public EquationExpressionOpInterface::ExternalModel<TimeOpInterface,
                                                          TimeOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    os << "time";
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    return mlir::isa<TimeOp>(other);
  }
};

struct CallOpInterface
    : public EquationExpressionOpInterface::ExternalModel<CallOpInterface,
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

      for (const auto &entry : argNamesPos) {
        mlir::Value arg = casted.getArgs()[entry.getValue()];
        mlir::Value otherArg =
            otherCasted.getArgs()[otherArgNamesPos[entry.getKey()]];

        if (!::areExpressionOperandsEquivalent(arg, otherArg,
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
                                   symbolTableCollection)) {
        return false;
      }
    } else if (otherArgNames) {
      if (mlir::failed(
              getArgNamesPos(casted, symbolTableCollection, argNamesPos))) {
        return false;
      }

      if (!compareNamedUnnamedArgs(otherCasted.getArgs(), otherArgNamesPos,
                                   casted.getArgs(), argNamesPos,
                                   symbolTableCollection)) {
        return false;
      }
    } else {
      if (!::areExpressionOperandsEquivalent(
              casted.getArgs(), otherCasted.getArgs(), symbolTableCollection)) {
        return false;
      }
    }

    return true;
  }

  void getArgNamesPos(mlir::ArrayAttr argNames,
                      llvm::StringMap<size_t> &pos) const {
    for (auto argName : llvm::enumerate(argNames)) {
      auto name = argName.value().cast<mlir::FlatSymbolRefAttr>().getValue();
      pos[name] = argName.index();
    }
  }

  mlir::LogicalResult
  getArgNamesPos(CallOp callOp,
                 mlir::SymbolTableCollection &symbolTableCollection,
                 llvm::StringMap<size_t> &pos) const {
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
    return llvm::all_of(child, [&](const auto &entry) {
      return parent.contains(entry.getKey());
    });
  }

  bool haveSameArgNames(const llvm::StringMap<size_t> &first,
                        const llvm::StringMap<size_t> &second) const {
    return containsArgNames(first, second) && containsArgNames(second, first);
  }

  bool compareNamedUnnamedArgs(
      mlir::ValueRange namedArgs, const llvm::StringMap<size_t> &namedArgsPos,
      mlir::ValueRange unnamedArgs,
      const llvm::StringMap<size_t> &unnamedArgsPos,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    if (namedArgs.size() != unnamedArgs.size()) {
      return false;
    }

    llvm::DenseMap<size_t, std::string> inverseUnnamedArgsPos;

    for (const auto &entry : unnamedArgsPos) {
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

      if (!::areExpressionOperandsEquivalent(
              namedArg, unnamedArg.value(), symbolTableCollection)) {
        return false;
      }
    }

    return true;
  }
};

struct CastOpInterface
    : public EquationExpressionOpInterface::ExternalModel<CastOpInterface,
                                                          CastOp> {
  void printExpression(
      mlir::Operation *op, llvm::raw_ostream &os,
      const llvm::DenseMap<mlir::Value, int64_t> &inductions) const {
    auto casted = mlir::cast<CastOp>(op);
    ::printExpression(os, casted.getValue(), inductions);
  }

  bool isEquivalent(
      mlir::Operation *op, mlir::Operation *other,
      mlir::SymbolTableCollection &symbolTableCollection) const {
    auto otherCasted = mlir::dyn_cast<CastOp>(other);

    if (!otherCasted) {
      return false;
    }

    return areEquationExpressionsEquivalent(op, otherCasted.getOperation(),
                                            symbolTableCollection);
  }
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
    DerOp::attachInterface<::DerOpInterface>(*context);
    TimeOp::attachInterface<TimeOpInterface>(*context);
    CallOp::attachInterface<::CallOpInterface>(*context);
    CastOp::attachInterface<::CastOpInterface>(*context);
    // clang-format on
  });
}
} // namespace mlir::bmodelica
