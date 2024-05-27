#include "marco/Dialect/BaseModelica/Transforms/EquationExpressionOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

using namespace ::mlir::bmodelica;

static void printExpression(
    llvm::raw_ostream& os,
    mlir::Value value,
    const llvm::DenseMap<mlir::Value, int64_t>& inductions)
{
  mlir::Operation* op = value.getDefiningOp();

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

namespace
{
  struct EquationSidesOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
            EquationSidesOpInterface, EquationSidesOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<EquationSidesOp>(op);

      os << "{";

      llvm::interleaveComma(
          castedOp.getLhsValues(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << "} = {";

      llvm::interleaveComma(
          castedOp.getRhsValues(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << "}";
    }
  };

  struct TensorFromElementsOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
            TensorFromElementsOpInterface, TensorFromElementsOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<TensorFromElementsOp>(op);

      os << "{";

      llvm::interleaveComma(
          castedOp.getValues(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << "}";
    }
  };

  struct TensorBroadcastOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
            TensorBroadcastOpInterface, TensorBroadcastOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
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
  };

  struct TensorViewOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
            TensorViewOpInterface, TensorViewOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<TensorViewOp>(op);

      ::printExpression(os, castedOp.getSource(), inductions);
      os << "[";

      llvm::interleaveComma(
          castedOp.getSubscriptions(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << "]";
    }

    mlir::LogicalResult getEquationAccesses(
        mlir::Operation* op,
        llvm::SmallVectorImpl<VariableAccess>& accesses,
        mlir::SymbolTableCollection& symbolTable,
        llvm::DenseMap<mlir::Value, unsigned int>& explicitInductionsPositionMap,
        AdditionalInductions& additionalInductions,
        llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>& dimensionAccesses,
        EquationPath path) const
    {
      auto castedOp = mlir::cast<TensorViewOp>(op);
      auto indices = castedOp.getSubscriptions();

      for (size_t i = 0, e = indices.size(); i < e; ++i) {
        mlir::Value index = indices[e - 1 - i];

        auto dimensionAccess = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, index);

        if (!dimensionAccess) {
          return mlir::failure();
        }

        dimensionAccesses.push_back(std::move(dimensionAccess));
      }

      auto sourceOp = castedOp.getSource()
                          .getDefiningOp<EquationExpressionOpInterface>();

      if (!sourceOp) {
        return mlir::failure();
      }

      return sourceOp.getEquationAccesses(
          accesses, symbolTable, explicitInductionsPositionMap,
          additionalInductions, dimensionAccesses,
          std::move(path));
    }
  };

  struct TensorExtractOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
            TensorExtractOpInterface, TensorExtractOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<TensorExtractOp>(op);

      ::printExpression(os, castedOp.getTensor(), inductions);
      os << "[";

      llvm::interleaveComma(
          castedOp.getIndices(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << "]";
    }

    mlir::LogicalResult getEquationAccesses(
        mlir::Operation* op,
        llvm::SmallVectorImpl<VariableAccess>& accesses,
        mlir::SymbolTableCollection& symbolTable,
        llvm::DenseMap<mlir::Value, unsigned int>& explicitInductionsPositionMap,
        AdditionalInductions& additionalInductions,
        llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>& dimensionAccesses,
        EquationPath path) const
    {
      auto castedOp = mlir::cast<TensorExtractOp>(op);
      auto indices = castedOp.getIndices();

      for (size_t i = 0, e = indices.size(); i < e; ++i) {
        mlir::Value index = indices[e - 1 - i];

        auto dimensionAccess = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, index);

        if (!dimensionAccess) {
          return mlir::failure();
        }

        dimensionAccesses.push_back(std::move(dimensionAccess));
      }

      auto tensorOp = castedOp.getTensor()
                          .getDefiningOp<EquationExpressionOpInterface>();

      if (!tensorOp) {
        return mlir::failure();
      }

      return tensorOp.getEquationAccesses(
          accesses, symbolTable, explicitInductionsPositionMap,
          additionalInductions, dimensionAccesses,
          std::move(path));
    }
  };

  struct ArrayFromElementsOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
            ArrayFromElementsOpInterface, ArrayFromElementsOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<ArrayFromElementsOp>(op);

      os << "{";

      llvm::interleaveComma(
          castedOp.getValues(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << "}";
    }
  };

  struct ArrayBroadcastOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
            ArrayBroadcastOpInterface, ArrayBroadcastOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<ArrayBroadcastOp>(op);

      os << "{";

      for (int64_t i = 0, e = castedOp.getArrayType().getNumElements();
           i < e; ++i) {
        if (i != 0) {
          os << ", ";
        }

        ::printExpression(os, castedOp.getValue(), inductions);
      }

      os << "}";
    }
  };

  struct ArrayCastOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
            ArrayCastOpInterface, ArrayCastOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<ArrayCastOp>(op);
      ::printExpression(os, castedOp.getOperand(), inductions);
    }

    mlir::LogicalResult getEquationAccesses(
        mlir::Operation* op,
        llvm::SmallVectorImpl<VariableAccess>& accesses,
        mlir::SymbolTableCollection& symbolTable,
        llvm::DenseMap<mlir::Value, unsigned int>& explicitInductionsPositionMap,
        AdditionalInductions& additionalInductions,
        llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>& dimensionAccesses,
        EquationPath path) const
    {
      auto castedOp = mlir::cast<ArrayCastOp>(op);
      mlir::Value source = castedOp.getSource();
      auto childOp = source.getDefiningOp();

      if (!childOp) {
        return mlir::success();
      }

      auto expressionInt =
          mlir::dyn_cast<EquationExpressionOpInterface>(childOp);

      if (!expressionInt) {
        return mlir::failure();
      }

      if (mlir::failed(expressionInt.getEquationAccesses(
              accesses, symbolTable,
              explicitInductionsPositionMap,
              additionalInductions,
              dimensionAccesses,
              path + 0))) {
        return mlir::failure();
      }

      return mlir::success();
    }
  };

  struct DimOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          DimOpInterface, DimOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<DimOp>(op);

      os << "dim(";
      ::printExpression(os, castedOp.getArray(), inductions);
      os << ", ";
      ::printExpression(os, castedOp.getDimension(), inductions);
      os << ")";
    }

    uint64_t getNumOfExpressionElements(mlir::Operation* op) const
    {
      return 1;
    }

    mlir::Value getExpressionElement(
        mlir::Operation* op,
        uint64_t position) const
    {
      auto castedOp = mlir::cast<DimOp>(op);
      assert(position == 0);
      return castedOp.getDimension();
    }
  };

  struct SubscriptionOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
            SubscriptionOpInterface, SubscriptionOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<SubscriptionOp>(op);

      ::printExpression(os, castedOp.getSource(), inductions);
      os << "[";

      llvm::interleaveComma(
          castedOp.getIndices(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << "]";
    }

    mlir::LogicalResult getEquationAccesses(
        mlir::Operation* op,
        llvm::SmallVectorImpl<VariableAccess>& accesses,
        mlir::SymbolTableCollection& symbolTable,
        llvm::DenseMap<mlir::Value, unsigned int>& explicitInductionsPositionMap,
        AdditionalInductions& additionalInductions,
        llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>& dimensionAccesses,
        EquationPath path) const
    {
      auto castedOp = mlir::cast<SubscriptionOp>(op);
      auto indices = castedOp.getIndices();

      for (size_t i = 0, e = indices.size(); i < e; ++i) {
        mlir::Value index = indices[e - 1 - i];

        auto dimensionAccess = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, index);

        if (!dimensionAccess) {
          return mlir::failure();
        }

        dimensionAccesses.push_back(std::move(dimensionAccess));
      }

      auto sourceOp = castedOp.getSource()
                          .getDefiningOp<EquationExpressionOpInterface>();

      if (!sourceOp) {
        return mlir::failure();
      }

      return sourceOp.getEquationAccesses(
          accesses, symbolTable, explicitInductionsPositionMap,
          additionalInductions, dimensionAccesses,
          std::move(path));
    }
  };

  struct LoadOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          LoadOpInterface, LoadOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<LoadOp>(op);

      ::printExpression(os, castedOp.getArray(), inductions);
      os << "[";

      llvm::interleaveComma(
          castedOp.getIndices(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << "]";
    }

    mlir::LogicalResult getEquationAccesses(
        mlir::Operation* op,
        llvm::SmallVectorImpl<VariableAccess>& accesses,
        mlir::SymbolTableCollection& symbolTable,
        llvm::DenseMap<mlir::Value, unsigned int>& explicitInductionsPositionMap,
        AdditionalInductions& additionalInductions,
        llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>& dimensionAccesses,
        EquationPath path) const
    {
      auto castedOp = mlir::cast<LoadOp>(op);
      auto indices = castedOp.getIndices();

      for (size_t i = 0, e = indices.size(); i < e; ++i) {
        mlir::Value index = indices[e - 1 - i];

        auto dimensionAccess = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, index);

        if (!dimensionAccess) {
          return mlir::failure();
        }

        dimensionAccesses.push_back(std::move(dimensionAccess));
      }

      auto arrayOp = castedOp.getArray()
                         .getDefiningOp<EquationExpressionOpInterface>();

      if (!arrayOp) {
        return mlir::failure();
      }

      return arrayOp.getEquationAccesses(
          accesses, symbolTable, explicitInductionsPositionMap,
          additionalInductions, dimensionAccesses,
          std::move(path));
    }
  };

  struct VariableGetOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          VariableGetOpInterface, VariableGetOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<VariableGetOp>(op);
      os << castedOp.getVariable();
    }

    mlir::LogicalResult getEquationAccesses(
        mlir::Operation* op,
        llvm::SmallVectorImpl<VariableAccess>& accesses,
        mlir::SymbolTableCollection& symbolTable,
        llvm::DenseMap<mlir::Value, unsigned int>& explicitInductionsPositionMap,
        AdditionalInductions& additionalInductions,
        llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>& dimensionAccesses,
        EquationPath path) const
    {
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
                     rank = tensorType.getRank(); i < rank; ++i) {
          int64_t dimension = tensorType.getDimSize(i);
          assert(dimension != mlir::ShapedType::kDynamic);

          reverted.push_back(std::make_unique<DimensionAccessRange>(
              castedOp.getContext(), Range(0, dimension)));
        }
      }

      accesses.push_back(VariableAccess(
          std::move(path),
          mlir::SymbolRefAttr::get(castedOp.getVariableAttr()),
          AccessFunction::build(
              castedOp.getContext(), numOfInductions, reverted)));

      return mlir::success();
    }
  };

  struct ConstantOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          ConstantOpInterface, ConstantOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
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

      if (auto integerAttr =
              castedOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
        os << integerAttr.getValue();
        return;
      }

      if (auto floatAttr = castedOp.getValue().dyn_cast<mlir::FloatAttr>()) {
        os << floatAttr.getValueAsDouble();
        return;
      }

      castedOp.getValue().print(os, true);
    }
  };

  struct NegateOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          NegateOpInterface, NegateOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<NegateOp>(op);

      os << "(- ";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct AddOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          AddOpInterface, AddOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<AddOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " + ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct AddEWOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          AddEWOpInterface, AddEWOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<AddEWOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " .+ ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct SubOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          SubOpInterface, SubOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<SubOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " - ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct SubEWOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          SubEWOpInterface, SubEWOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<SubEWOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " .- ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct MulOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          MulOpInterface, MulOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<MulOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " * ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct MulEWOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          MulEWOpInterface, MulEWOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<MulEWOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " .* ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct DivOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          DivOpInterface, DivOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<DivOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " / ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct DivEWOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          DivEWOpInterface, DivEWOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<DivEWOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " ./ ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct PowOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          PowOpInterface, PowOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<PowOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getBase(), inductions);
      os << " ^ ";
      ::printExpression(os, castedOp.getExponent(), inductions);
      os << ")";
    }
  };

  struct PowEWOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          PowEWOpInterface, PowEWOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<PowEWOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getBase(), inductions);
      os << " .^ ";
      ::printExpression(os, castedOp.getExponent(), inductions);
      os << ")";
    }
  };

  struct EqOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          EqOpInterface, EqOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<EqOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " == ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct NotEqOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          NotEqOpInterface, NotEqOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<NotEqOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " != ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct GtOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          GtOpInterface, GtOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<GtOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " > ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct GteOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          GteOpInterface, GteOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<GteOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " >= ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct LtOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          LtOpInterface, LtOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<LtOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " < ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct LteOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          LteOpInterface, LteOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<LteOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " <= ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct NotOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          NotOpInterface, NotOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<NotOp>(op);

      os << "!(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct AndOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          AndOpInterface, AndOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<AndOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " && ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct OrOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          OrOpInterface, OrOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<OrOp>(op);

      os << "(";
      ::printExpression(os, castedOp.getLhs(), inductions);
      os << " || ";
      ::printExpression(os, castedOp.getRhs(), inductions);
      os << ")";
    }
  };

  struct SelectOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          SelectOpInterface, SelectOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<SelectOp>(op);

      ::printExpression(os, castedOp.getCondition(), inductions);
      os << " ? (";

      llvm::interleaveComma(
          castedOp.getTrueValues(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << ") : (";

      llvm::interleaveComma(
          castedOp.getFalseValues(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << ")";
    }
  };

  struct AbsOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          AbsOpInterface, AbsOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<AbsOp>(op);

      os << "abs(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct AcosOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          AcosOpInterface, AcosOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<AcosOp>(op);

      os << "acos(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct AsinOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          AsinOpInterface, AsinOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<AsinOp>(op);

      os << "asin(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct AtanOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          AtanOpInterface, AtanOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<AtanOp>(op);

      os << "atan(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct Atan2OpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          Atan2OpInterface, Atan2Op>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<Atan2Op>(op);

      os << "atan2(";
      ::printExpression(os, castedOp.getY(), inductions);
      os << ", ";
      ::printExpression(os, castedOp.getX(), inductions);
      os << ")";
    }
  };

  struct CeilOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          CeilOpInterface, CeilOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<CeilOp>(op);

      os << "ceil(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct CosOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          CosOpInterface, CosOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<CosOp>(op);

      os << "cos(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct CoshOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          CoshOpInterface, CoshOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<CoshOp>(op);

      os << "cosh(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct DiagonalOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          DiagonalOpInterface, DiagonalOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<DiagonalOp>(op);

      os << "diagonal(";
      ::printExpression(os, castedOp.getValues(), inductions);
      os << ")";
    }
  };

  struct DivTruncOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
            DivTruncOpInterface, DivTruncOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<DivTruncOp>(op);

      os << "div(";
      ::printExpression(os, castedOp.getX(), inductions);
      os << ", ";
      ::printExpression(os, castedOp.getY(), inductions);
      os << ")";
    }
  };

  struct ExpOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          ExpOpInterface, ExpOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<ExpOp>(op);

      os << "exp(";
      ::printExpression(os, castedOp.getExponent(), inductions);
      os << ")";
    }
  };

  struct FillOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          FillOpInterface, FillOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<FillOp>(op);

      os << "fill(";
      ::printExpression(os, castedOp.getValue(), inductions);
      os << ")";
    }
  };

  struct FloorOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          FloorOpInterface, FloorOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<FloorOp>(op);

      os << "floor(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct IdentityOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          IdentityOpInterface, IdentityOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<IdentityOp>(op);

      os << "identity(";
      ::printExpression(os, castedOp.getSize(), inductions);
      os << ")";
    }
  };

  struct IntegerOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          IntegerOpInterface, IntegerOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<IntegerOp>(op);

      os << "integer(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct LinspaceOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          LinspaceOpInterface, LinspaceOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<LinspaceOp>(op);

      os << "linspace(";
      ::printExpression(os, castedOp.getBegin(), inductions);
      os << ", ";
      ::printExpression(os, castedOp.getEnd(), inductions);
      os << ", ";
      ::printExpression(os, castedOp.getAmount(), inductions);
      os << ")";
    }
  };

  struct LogOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          LogOpInterface, LogOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<LogOp>(op);

      os << "log(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct Log10OpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          Log10OpInterface, Log10Op>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<Log10Op>(op);

      os << "log10(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct MaxOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          MaxOpInterface, MaxOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<MaxOp>(op);

      os << "max(";
      ::printExpression(os, castedOp.getFirst(), inductions);

      if (mlir::Value second = castedOp.getSecond()) {
        os << ", ";
        ::printExpression(os, second, inductions);
      }

      os << ")";
    }
  };

  struct MinOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          MinOpInterface, MinOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<MinOp>(op);

      os << "min(";
      ::printExpression(os, castedOp.getFirst(), inductions);

      if (mlir::Value second = castedOp.getSecond()) {
        os << ", ";
        ::printExpression(os, second, inductions);
      }

      os << ")";
    }
  };

  struct ModOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          ModOpInterface, ModOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<ModOp>(op);

      os << "mod(";
      ::printExpression(os, castedOp.getX(), inductions);
      os << ", ";
      ::printExpression(os, castedOp.getY(), inductions);
      os << ")";
    }
  };

  struct NDimsOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          NDimsOpInterface, NDimsOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<NDimsOp>(op);

      os << "ndims(";
      ::printExpression(os, castedOp.getArray(), inductions);
      os << ")";
    }
  };

  struct OnesOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          OnesOpInterface, OnesOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<OnesOp>(op);

      os << "ones(";

      llvm::interleaveComma(
          castedOp.getSizes(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << ")";
    }
  };

  struct ProductOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          ProductOpInterface, ProductOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<ProductOp>(op);

      os << "product(";
      ::printExpression(os, castedOp.getArray(), inductions);
      os << ")";
    }
  };

  struct RemOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          RemOpInterface, RemOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<RemOp>(op);

      os << "rem(";
      ::printExpression(os, castedOp.getX(), inductions);
      os << ", ";
      ::printExpression(os, castedOp.getY(), inductions);
      os << ")";
    }
  };

  struct SignOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          SignOpInterface, SignOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<SignOp>(op);

      os << "sign(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct SinOpInterface
       : public EquationExpressionOpInterface::ExternalModel<
           SinOpInterface, SinOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<SinOp>(op);

      os << "sin(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct SinhOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          SinhOpInterface, SinhOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<SinhOp>(op);

      os << "sinh(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct SizeOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          SizeOpInterface, SizeOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<SizeOp>(op);

      os << "size(";
      ::printExpression(os, castedOp.getArray(), inductions);

      if (mlir::Value dimension = castedOp.getDimension()) {
        os << ", ";
        ::printExpression(os, dimension, inductions);
      }

      os << ")";
    }
  };

  struct SqrtOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          SqrtOpInterface, SqrtOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<SqrtOp>(op);

      os << "sqrt(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct SumOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          SumOpInterface, SumOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<SumOp>(op);

      os << "sum(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct SymmetricOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          SymmetricOpInterface, SymmetricOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<SymmetricOp>(op);

      os << "symmetric(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct TanOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          TanOpInterface, TanOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<TanOp>(op);

      os << "tan(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct TanhOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          TanhOpInterface, TanhOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<TanhOp>(op);

      os << "tanh(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct TransposeOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          TransposeOpInterface, TransposeOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<TransposeOp>(op);

      os << "transpose(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct ZerosOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          ZerosOpInterface, ZerosOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<ZerosOp>(op);

      os << "zeros(";

      llvm::interleaveComma(
          castedOp.getSizes(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << ")";
    }
  };

  struct ReductionOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          ReductionOpInterface, ReductionOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
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

      auto terminator = mlir::cast<YieldOp>(
          castedOp.getBody()->getTerminator());

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

    uint64_t getNumOfExpressionElements(mlir::Operation* op) const
    {
      auto castedOp = mlir::cast<ReductionOp>(op);

      auto terminator = mlir::cast<YieldOp>(
          castedOp.getBody()->getTerminator());

      return terminator.getValues().size();
    }

    mlir::Value getExpressionElement(
        mlir::Operation* op, uint64_t element) const
    {
      auto castedOp = mlir::cast<ReductionOp>(op);

      auto terminator = mlir::cast<YieldOp>(
          castedOp.getBody()->getTerminator());

      return terminator.getValues()[element];
    }

    llvm::SmallVector<mlir::Value> getAdditionalInductions(
        mlir::Operation* op) const
    {
      auto castedOp = mlir::cast<ReductionOp>(op);
      llvm::SmallVector<mlir::Value> result;
      auto inductions = castedOp.getInductions();
      result.append(inductions.begin(), inductions.end());
      return result;
    }

    mlir::LogicalResult mapAdditionalInductions(
        mlir::Operation* op,
        AdditionalInductions& additionalInductions) const
    {
      auto castedOp = mlir::cast<ReductionOp>(op);

      IndexSet indices;
      llvm::SmallVector<std::pair<mlir::Value, uint64_t>> inductionsMap;

      for (const auto& [induction, iterable] :
           llvm::zip(castedOp.getInductions(), castedOp.getIterables())) {
        auto constantOp = iterable.getDefiningOp<ConstantOp>();

        if (!constantOp) {
          return mlir::failure();
        }

        auto iterableAttr = constantOp.getValue();

        if (auto rangeAttr = iterableAttr.dyn_cast<IntegerRangeAttr>()) {
          assert(rangeAttr.getStep() == 1);

          auto lowerBound = static_cast<Range::data_type>(
              rangeAttr.getLowerBound());

          auto upperBound = static_cast<Range::data_type>(
              rangeAttr.getUpperBound());

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
      : public EquationExpressionOpInterface::ExternalModel<
          DerOpInterface, DerOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto castedOp = mlir::cast<DerOp>(op);

      os << "der(";
      ::printExpression(os, castedOp.getOperand(), inductions);
      os << ")";
    }
  };

  struct TimeOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          TimeOpInterface, TimeOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      os << "time";
    }
  };

  struct CallOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          CallOpInterface, CallOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto casted = mlir::cast<CallOp>(op);
      os << casted.getCallee() << "(";

      llvm::interleaveComma(
          casted.getArgs(), os,
          [&](mlir::Value exp) {
            ::printExpression(os, exp, inductions);
          });

      os << ")";
    }
  };

  struct CastOpInterface
      : public EquationExpressionOpInterface::ExternalModel<
          CastOpInterface, CastOp>
  {
    void printExpression(
        mlir::Operation* op,
        llvm::raw_ostream& os,
        const llvm::DenseMap<mlir::Value, int64_t>& inductions) const
    {
      auto casted = mlir::cast<CastOp>(op);
      ::printExpression(os, casted.getValue(), inductions);
    }
  };
}

namespace mlir::bmodelica
{
  void registerEquationExpressionOpInterfaceExternalModels(
      mlir::DialectRegistry& registry)
  {
    registry.addExtension(+[](mlir::MLIRContext* context,
                              BaseModelicaDialect* dialect) {
      // Equation root.
      EquationSidesOp::attachInterface<::EquationSidesOpInterface>(*context);

      // Tensor operations.
      TensorFromElementsOp::attachInterface<
          ::TensorFromElementsOpInterface>(*context);

      TensorBroadcastOp::attachInterface<
          ::TensorBroadcastOpInterface>(*context);

      TensorViewOp::attachInterface<::TensorViewOpInterface>(*context);
      TensorExtractOp::attachInterface<::TensorExtractOpInterface>(*context);

      // Array operations.
      ArrayFromElementsOp::attachInterface<
          ::ArrayFromElementsOpInterface>(*context);

      ArrayBroadcastOp::attachInterface<::ArrayBroadcastOpInterface>(*context);
      ArrayCastOp::attachInterface<::ArrayCastOpInterface>(*context);
      DimOp::attachInterface<::DimOpInterface>(*context);
      SubscriptionOp::attachInterface<::SubscriptionOpInterface>(*context);
      LoadOp::attachInterface<::LoadOpInterface>(*context);

      // Variable operations.
      VariableGetOp::attachInterface<::VariableGetOpInterface>(*context);

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
    });
  }
}
