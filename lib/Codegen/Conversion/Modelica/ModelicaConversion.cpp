#include "marco/Codegen/Conversion/Modelica/ModelicaConversion.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Conversion/Modelica/TypeConverter.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

static bool isNumeric(mlir::Type type)
{
  return type.isa<mlir::IndexType, BooleanType, IntegerType, RealType>();
}

static bool isNumeric(mlir::Value value)
{
  return isNumeric(value.getType());
}

static mlir::Attribute getZeroAttr(mlir::Type type)
{
  if (type.isa<BooleanType>()) {
    return BooleanAttr::get(type.getContext(), 0);
  }

  if (type.isa<IntegerType>()) {
    return IntegerAttr::get(type.getContext(), 0);
  }

  if (type.isa<RealType>()) {
    return RealAttr::get(type.getContext(), 0);
  }

  llvm_unreachable("Unknown type");
  return mlir::Attribute();
}

static std::vector<mlir::Value> getArrayDynamicDimensions(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value array)
{
  std::vector<mlir::Value> result;

  assert(array.getType().isa<ArrayType>());
  auto arrayType = array.getType().cast<ArrayType>();

  for (const auto& dimension : llvm::enumerate(arrayType.getShape())) {
    if (dimension.value() == -1) {
      mlir::Value dim = builder.create<ConstantOp>(loc, builder.getIndexAttr(dimension.index()));
      result.push_back(builder.create<DimOp>(loc, array, dim));
    }
  }

  return result;
}

static void iterateArray(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::Value array,
    std::function<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)> callback)
{
  assert(array.getType().isa<ArrayType>());
  auto arrayType = array.getType().cast<ArrayType>();

  mlir::Value zero = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(0));
  mlir::Value one = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(1));

  llvm::SmallVector<mlir::Value, 3> lowerBounds(arrayType.getRank(), zero);
  llvm::SmallVector<mlir::Value, 3> upperBounds;
  llvm::SmallVector<mlir::Value, 3> steps(arrayType.getRank(), one);

  for (unsigned int i = 0, e = arrayType.getRank(); i < e; ++i) {
    mlir::Value dim = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(i));
    upperBounds.push_back(builder.create<DimOp>(loc, array, dim));
  }

  // Create nested loops in order to iterate on each dimension of the array
  mlir::scf::buildLoopNest(builder, loc, lowerBounds, upperBounds, steps, callback);
}

static mlir::FuncOp getOrDeclareFunction(mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::TypeRange results, mlir::TypeRange args)
{
	if (auto foo = module.lookupSymbol<mlir::FuncOp>(name)) {
    return foo;
  }

	mlir::PatternRewriter::InsertionGuard insertGuard(builder);
	builder.setInsertionPointToStart(module.getBody());
	auto foo = builder.create<mlir::FuncOp>(module.getLoc(), name, builder.getFunctionType(args, results));
	foo.setPrivate();
	return foo;
}

static mlir::FuncOp getOrDeclareFunction(mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::TypeRange results, mlir::ValueRange args)
{
	return getOrDeclareFunction(builder, module, name, results, args.getTypes());
}

static mlir::Type castToMostGenericType(
    mlir::OpBuilder& builder, mlir::ValueRange values, llvm::SmallVectorImpl<mlir::Value>& castedValues)
{
	mlir::Type resultType = nullptr;
	mlir::Type resultBaseType = nullptr;

	for (const auto& value : values) {
		mlir::Type type = value.getType();
		mlir::Type baseType = type;

		if (resultType == nullptr) {
			resultType = type;
			resultBaseType = type;

			while (resultBaseType.isa<ArrayType>()) {
        resultBaseType = resultBaseType.cast<ArrayType>().getElementType();
      }

			continue;
		}

		if (type.isa<ArrayType>()) {
			while (baseType.isa<ArrayType>()) {
        baseType = baseType.cast<ArrayType>().getElementType();
      }
		}

		if (resultBaseType.isa<mlir::IndexType>() || baseType.isa<RealType>()) {
			resultType = type;
			resultBaseType = baseType;
		}
	}

	llvm::SmallVector<mlir::Type, 3> types;

	for (const auto& value : values)
	{
		mlir::Type type = value.getType();

		if (type.isa<ArrayType>())
		{
			auto arrayType = type.cast<ArrayType>();
			auto shape = arrayType.getShape();
			types.emplace_back(ArrayType::get(arrayType.getContext(), resultBaseType, shape));
		}
		else
			types.emplace_back(resultBaseType);
	}

	for (const auto& [value, type] : llvm::zip(values, types))
	{
		mlir::Value castedValue = builder.create<CastOp>(value.getLoc(), type, value);
		castedValues.push_back(castedValue);
	}

	return types[0];
}

namespace
{
  template<typename Op>
  class ModelicaOpRewritePattern : public mlir::OpRewritePattern<Op>
  {
    public:
      ModelicaOpRewritePattern(mlir::MLIRContext* ctx, ModelicaConversionOptions options)
          : mlir::OpRewritePattern<Op>(ctx),
            options(std::move(options))
      {
      }
    
    protected:
      ModelicaConversionOptions options;
  };

  struct AssignmentOpScalarLowering : public ModelicaOpRewritePattern<AssignmentOp>
  {
    using ModelicaOpRewritePattern<AssignmentOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(AssignmentOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      if (!isNumeric(op.value())) {
        return rewriter.notifyMatchFailure(op, "Source value has not a numeric type");
      }

      auto destinationBaseType = op.destination().getType().cast<ArrayType>().getElementType();
      mlir::Value value = rewriter.create<CastOp>(loc, destinationBaseType, op.value());
      rewriter.replaceOpWithNewOp<StoreOp>(op, value, op.destination(), llvm::None);

      return mlir::success();
    }
  };

  struct AssignmentOpArrayLowering : public ModelicaOpRewritePattern<AssignmentOp>
  {
    using ModelicaOpRewritePattern<AssignmentOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(AssignmentOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      if (!op.value().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Source value is not an array");
      }

      iterateArray(rewriter, op.getLoc(), op.value(),
                   [&](mlir::OpBuilder& nestedBuilder, mlir::Location, mlir::ValueRange position) {
                     mlir::Value value = rewriter.create<LoadOp>(loc, op.value(), position);
                     value = rewriter.create<CastOp>(value.getLoc(), op.destination().getType().cast<ArrayType>().getElementType(), value);
                     rewriter.create<StoreOp>(loc, value, op.destination(), position);
                   });

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };

  struct NotOpArrayLowering : public ModelicaOpRewritePattern<NotOp>
  {
    using ModelicaOpRewritePattern<NotOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(NotOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operand is compatible
      if (!op.operand().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Operand is not an array");
      }

      if (auto operandArrayType = op.operand().getType().cast<ArrayType>(); !operandArrayType.getElementType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Operand is not an array of booleans");
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.operand());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array element
      iterateArray(
          rewriter, loc, op.operand(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value value = nestedBuilder.create<LoadOp>(location, op.operand(), indices);
            mlir::Value negated = nestedBuilder.create<NotOp>(location, resultArrayType.getElementType(), value);
            nestedBuilder.create<StoreOp>(location, negated, result, indices);
          });

      return mlir::success();
    }
  };

  template<typename Op>
  struct BinaryLogicOpArrayLowering : public ModelicaOpRewritePattern<Op>
  {
    using ModelicaOpRewritePattern<Op>::ModelicaOpRewritePattern;

    virtual mlir::Value getLhs(Op op) const = 0;
    virtual mlir::Value getRhs(Op op) const = 0;

    virtual mlir::Value scalarize(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elementType, mlir::Value lhs, mlir::Value rhs) const = 0;

    mlir::LogicalResult matchAndRewrite(Op op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!getLhs(op).getType().template isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array");
      }

      auto lhsArrayType = getLhs(op).getType().template cast<ArrayType>();

      if (!lhsArrayType.getElementType().template isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");
      }

      if (!getRhs(op).getType().template isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not an array");
      }

      auto rhsArrayType = getRhs(op).getType().template cast<ArrayType>();

      if (!rhsArrayType.getElementType().template isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not an array of booleans");
      }

      assert(lhsArrayType.getRank() == rhsArrayType.getRank());

      if (this->options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == -1 || rhsShape[i] == -1) {
            mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, getLhs(op), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, getRhs(op), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().template cast<ArrayType>();
      auto lhsDynamicDimensions = getArrayDynamicDimensions(rewriter, loc, getLhs(op));
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, lhsDynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, result,
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(loc, getLhs(op), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(loc, getRhs(op), indices);
            mlir::Value scalarResult = scalarize(nestedBuilder, location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(loc, scalarResult, result, indices);
          });

      return mlir::success();
    }
  };

  struct AndOpArrayLowering : public BinaryLogicOpArrayLowering<AndOp>
  {
    using BinaryLogicOpArrayLowering<AndOp>::BinaryLogicOpArrayLowering;

    mlir::Value getLhs(AndOp op) const override
    {
      return op.lhs();
    }

    mlir::Value getRhs(AndOp op) const override
    {
      return op.rhs();
    }

    mlir::Value scalarize(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elementType, mlir::Value lhs, mlir::Value rhs) const override
    {
      return builder.create<AndOp>(loc, elementType, lhs, rhs);
    }
  };

  struct OrOpArrayLowering : public BinaryLogicOpArrayLowering<OrOp>
  {
    using BinaryLogicOpArrayLowering<OrOp>::BinaryLogicOpArrayLowering;

    mlir::Value getLhs(OrOp op) const override
    {
      return op.lhs();
    }

    mlir::Value getRhs(OrOp op) const override
    {
      return op.rhs();
    }

    mlir::Value scalarize(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elementType, mlir::Value lhs, mlir::Value rhs) const override
    {
      return builder.create<OrOp>(loc, elementType, lhs, rhs);
    }
  };

  struct NegateOpArrayLowering : public ModelicaOpRewritePattern<NegateOp>
  {
    using ModelicaOpRewritePattern<NegateOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(NegateOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operand is compatible
      if (!op.operand().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Value is not an array");
      }

      auto operandArrayType = op.operand().getType().cast<ArrayType>();

      if (!isNumeric(operandArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Array has not numeric elements");
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.operand());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.operand(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value source = nestedBuilder.create<LoadOp>(location, op.operand(), indices);
            mlir::Value value = nestedBuilder.create<NegateOp>(location, resultArrayType.getElementType(), source);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct AddOpArrayLowering : public ModelicaOpRewritePattern<AddOp>
  {
    using ModelicaOpRewritePattern<AddOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(AddOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");
      }

      if (!op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");
      }

      auto lhsArrayType = op.lhs().getType().cast<ArrayType>();
      auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

      for (auto pair : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        auto lhsDimension = std::get<0>(pair);
        auto rhsDimension = std::get<1>(pair);

        if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension) {
          return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
        }
      }

      if (!isNumeric(lhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");
      }

      if (!isNumeric(rhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");
      }

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        assert(lhsArrayType.getRank() == rhsArrayType.getRank());

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == -1 || rhsShape[i] == -1) {
            mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(location, op.lhs(), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(location, op.rhs(), indices);
            mlir::Value value = nestedBuilder.create<AddOp>(location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct AddOpMixedLowering : public ModelicaOpRewritePattern<AddOp>
  {
    using ModelicaOpRewritePattern<AddOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(AddOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>() && !op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "None of the operands is an array");
      }

      if (op.lhs().getType().isa<ArrayType>() && isNumeric(op.rhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a scalar");
      }

      if (op.rhs().getType().isa<ArrayType>() && isNumeric(op.lhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a scalar");
      }

      mlir::Value array = op.lhs().getType().isa<ArrayType>() ? op.lhs() : op.rhs();
      mlir::Value scalar = op.lhs().getType().isa<ArrayType>() ? op.rhs() : op.lhs();

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, array);
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, array, indices);
            mlir::Value value = nestedBuilder.create<AddOp>(location, resultArrayType.getElementType(), arrayValue, scalar);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct AddEWOpArrayLowering : public ModelicaOpRewritePattern<AddEWOp>
  {
    using ModelicaOpRewritePattern<AddEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(AddEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");
      }

      if (!op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");
      }

      auto lhsArrayType = op.lhs().getType().cast<ArrayType>();
      auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

      for (auto pair : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        auto lhsDimension = std::get<0>(pair);
        auto rhsDimension = std::get<1>(pair);

        if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension) {
          return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
        }
      }

      if (!isNumeric(lhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");
      }

      if (!isNumeric(rhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");
      }

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        assert(lhsArrayType.getRank() == rhsArrayType.getRank());

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == -1 || rhsShape[i] == -1) {
            mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(location, op.lhs(), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(location, op.rhs(), indices);
            mlir::Value value = nestedBuilder.create<AddOp>(location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct AddEWOpMixedLowering : public ModelicaOpRewritePattern<AddEWOp>
  {
    using ModelicaOpRewritePattern<AddEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(AddEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>() && !op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "None of the operands is an array");
      }

      if (op.lhs().getType().isa<ArrayType>() && !isNumeric(op.rhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a scalar");
      }

      if (op.rhs().getType().isa<ArrayType>() && !isNumeric(op.lhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a scalar");
      }

      mlir::Value array = op.lhs().getType().isa<ArrayType>() ? op.lhs() : op.rhs();
      mlir::Value scalar = op.lhs().getType().isa<ArrayType>() ? op.rhs() : op.lhs();

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, array);
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, array, indices);
            mlir::Value value = nestedBuilder.create<AddOp>(location, resultArrayType.getElementType(), arrayValue, scalar);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct SubOpArrayLowering : public ModelicaOpRewritePattern<SubOp>
  {
    using ModelicaOpRewritePattern<SubOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SubOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");
      }

      if (!op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");
      }

      auto lhsArrayType = op.lhs().getType().cast<ArrayType>();
      auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

      for (auto pair : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        auto lhsDimension = std::get<0>(pair);
        auto rhsDimension = std::get<1>(pair);

        if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension) {
          return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
        }
      }

      if (!isNumeric(lhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");
      }

      if (!isNumeric(rhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");
      }

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        assert(lhsArrayType.getRank() == rhsArrayType.getRank());

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == -1 || rhsShape[i] == -1) {
            mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(location, op.lhs(), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(location, op.rhs(), indices);
            mlir::Value value = nestedBuilder.create<SubOp>(location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct SubOpMixedLowering : public ModelicaOpRewritePattern<SubOp>
  {
    using ModelicaOpRewritePattern<SubOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SubOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>() && !op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "None of the operands is an array");
      }

      if (op.lhs().getType().isa<ArrayType>() && isNumeric(op.rhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a scalar");
      }

      if (op.rhs().getType().isa<ArrayType>() && isNumeric(op.lhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a scalar");
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs().getType().isa<ArrayType>() ? op.lhs() : op.rhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            if (op.lhs().getType().isa<ArrayType>()) {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.lhs(), indices);
              mlir::Value value = nestedBuilder.create<SubOp>(location, resultArrayType.getElementType(), arrayValue, op.rhs());
              nestedBuilder.create<StoreOp>(location, value, result, indices);
            } else {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.rhs(), indices);
              mlir::Value value = nestedBuilder.create<SubOp>(location, resultArrayType.getElementType(), op.lhs(), arrayValue);
              nestedBuilder.create<StoreOp>(location, value, result, indices);
            }
          });

      return mlir::success();
    }
  };

  struct SubEWOpArrayLowering : public ModelicaOpRewritePattern<SubEWOp>
  {
    using ModelicaOpRewritePattern<SubEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SubEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");
      }

      if (!op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");
      }

      auto lhsArrayType = op.lhs().getType().cast<ArrayType>();
      auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

      for (auto pair : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        auto lhsDimension = std::get<0>(pair);
        auto rhsDimension = std::get<1>(pair);

        if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension) {
          return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
        }
      }

      if (!isNumeric(lhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");
      }

      if (!isNumeric(rhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");
      }

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        assert(lhsArrayType.getRank() == rhsArrayType.getRank());

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == -1 || rhsShape[i] == -1) {
            mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(location, op.lhs(), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(location, op.rhs(), indices);
            mlir::Value value = nestedBuilder.create<SubOp>(location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct SubEWOpMixedLowering : public ModelicaOpRewritePattern<SubEWOp>
  {
    using ModelicaOpRewritePattern<SubEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SubEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>() && !op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "None of the operands is an array");
      }

      if (op.lhs().getType().isa<ArrayType>() && !isNumeric(op.rhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a scalar");
      }

      if (op.rhs().getType().isa<ArrayType>() && !isNumeric(op.lhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a scalar");
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs().getType().isa<ArrayType>() ? op.lhs() : op.rhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            if (op.lhs().getType().isa<ArrayType>()) {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.lhs(), indices);
              mlir::Value value = nestedBuilder.create<SubOp>(location, resultArrayType.getElementType(), arrayValue, op.rhs());
              nestedBuilder.create<StoreOp>(location, value, result, indices);
            } else {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.rhs(), indices);
              mlir::Value value = nestedBuilder.create<SubOp>(location, resultArrayType.getElementType(), op.lhs(), arrayValue);
              nestedBuilder.create<StoreOp>(location, value, result, indices);
            }
          });

      return mlir::success();
    }
  };

  /// Product between a scalar and an array.
  struct MulOpScalarProductLowering : public ModelicaOpRewritePattern<MulOp>
  {
    using ModelicaOpRewritePattern<MulOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>() && !op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Scalar-array product: none of the operands is an array");
      }

      if (op.lhs().getType().isa<ArrayType>() && !isNumeric(op.rhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Scalar-array product: right-hand side operand is not a scalar");
      }

      if (op.rhs().getType().isa<ArrayType>() && !isNumeric(op.lhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Scalar-array product: left-hand side operand is not a scalar");
      }

      mlir::Value array = op.lhs().getType().isa<ArrayType>() ? op.lhs() : op.rhs();
      mlir::Value scalar = op.lhs().getType().isa<ArrayType>() ? op.rhs() : op.lhs();

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, array);
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Multiply each array element by the scalar value
      iterateArray(
          rewriter, loc, array,
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, array, indices);
            mlir::Value value = nestedBuilder.create<MulOp>(location, resultArrayType.getElementType(), scalar, arrayValue);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  /// Cross product of two 1-D arrays.
  /// Result is a scalar.
  ///
  /// [ x1, x2, x3 ] * [ y1, y2, y3 ] = x1 * y1 + x2 * y2 + x3 * y3
  struct MulOpCrossProductLowering : public ModelicaOpRewritePattern<MulOp>
  {
    using ModelicaOpRewritePattern<MulOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Cross product: left-hand side value is not an array");
      }

      auto lhsArrayType = op.lhs().getType().cast<ArrayType>();

      if (lhsArrayType.getRank() != 1) {
        return rewriter.notifyMatchFailure(op, "Cross product: left-hand side arrays is not 1-D");
      }

      if (!op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Cross product: right-hand side value is not an array");
      }

      auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

      if (rhsArrayType.getRank() != 1) {
        return rewriter.notifyMatchFailure(op, "Cross product: right-hand side arrays is not 1-D");
      }

      if (lhsArrayType.getShape()[0] != -1 && rhsArrayType.getShape()[0] != -1) {
        if (lhsArrayType.getShape()[0] != rhsArrayType.getShape()[0]) {
          return rewriter.notifyMatchFailure(op, "Cross product: the two arrays have different shape");
        }
      }

      assert(lhsArrayType.getRank() == 1);
      assert(rhsArrayType.getRank() == 1);

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        if (lhsShape[0] == -1 || rhsShape[0] == -1) {
          mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
          mlir::Value rhsDimensionSize =  rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
          mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
          rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
        }
      }

      // Compute the result
      mlir::Type resultType = op.getResult().getType();

      mlir::Value lowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value upperBound = rewriter.create<DimOp>(loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

      mlir::Value init = rewriter.create<ConstantOp>(loc, getZeroAttr(resultType));

      // Iterate on the two arrays at the same time, and propagate the
      // progressive result to the next loop iteration.
      auto loop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);

      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loop.getBody());

        mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), loop.getInductionVar());
        mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), loop.getInductionVar());
        mlir::Value product = rewriter.create<MulOp>(loc, resultType, lhs, rhs);
        mlir::Value sum = rewriter.create<AddOp>(loc, resultType, product, loop.getRegionIterArgs()[0]);
        rewriter.create<mlir::scf::YieldOp>(loc, sum);
      }

      rewriter.replaceOp(op, loop.getResult(0));
      return mlir::success();
    }
  };

  /// Product of a vector (1-D array) and a matrix (2-D array).
  ///
  /// [ x1 ]  *  [ y11, y12 ]  =  [ (x1 * y11 + x2 * y21 + x3 * y31), (x1 * y12 + x2 * y22 + x3 * y32) ]
  /// [ x2 ]		 [ y21, y22 ]
  /// [ x3 ]		 [ y31, y32 ]
  struct MulOpVectorMatrixLowering : public ModelicaOpRewritePattern<MulOp>
  {
    using ModelicaOpRewritePattern<MulOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Vector-matrix product: left-hand side value is not an array");
      }

      auto lhsArrayType = op.lhs().getType().cast<ArrayType>();

      if (lhsArrayType.getRank() != 1) {
        return rewriter.notifyMatchFailure(op, "Vector-matrix product: left-hand size array is not 1-D");
      }

      if (!op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Vector-matrix product: right-hand side value is not an array");
      }

      auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

      if (rhsArrayType.getRank() != 2) {
        return rewriter.notifyMatchFailure(op, "Vector-matrix product: right-hand side matrix is not 2-D");
      }

      if (lhsArrayType.getShape()[0] != -1 && rhsArrayType.getShape()[0] != -1) {
        if (lhsArrayType.getShape()[0] != rhsArrayType.getShape()[0]) {
          return rewriter.notifyMatchFailure(op, "Vector-matrix product: incompatible shapes");
        }
      }

      assert(lhsArrayType.getRank() == 1);
      assert(rhsArrayType.getRank() == 2);

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        if (lhsShape[0] == -1 || rhsShape[0] == -1) {
          mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), zero);
          mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), zero);
          mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
          rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto shape = resultArrayType.getShape();
      assert(shape.size() == 1);

      llvm::SmallVector<mlir::Value, 1> dynamicDimensions;

      if (shape[0] == -1) {
        dynamicDimensions.push_back(rewriter.create<DimOp>(
            loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1))));
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Iterate on the columns
      mlir::Value columnsLowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value columnsUpperBound = rewriter.create<DimOp>(loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1)));
      mlir::Value columnsStep = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto outerLoop = rewriter.create<mlir::scf::ForOp>(loc, columnsLowerBound, columnsUpperBound, columnsStep);
      rewriter.setInsertionPointToStart(outerLoop.getBody());

      // Product between the vector and the current column
      mlir::Value lowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value upperBound = rewriter.create<DimOp>(loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

      mlir::Value init = rewriter.create<ConstantOp>(loc, getZeroAttr(resultArrayType.getElementType()));

      auto innerLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);
      rewriter.setInsertionPointToStart(innerLoop.getBody());

      mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), innerLoop.getInductionVar());
      mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), mlir::ValueRange({ innerLoop.getInductionVar(), outerLoop.getInductionVar() }));
      mlir::Value product = rewriter.create<MulOp>(loc, resultArrayType.getElementType(), lhs, rhs);
      mlir::Value sum = rewriter.create<AddOp>(loc, resultArrayType.getElementType(), product, innerLoop.getRegionIterArgs()[0]);
      rewriter.create<mlir::scf::YieldOp>(loc, sum);

      // Store the product in the result array
      rewriter.setInsertionPointAfter(innerLoop);
      mlir::Value productResult = innerLoop.getResult(0);
      rewriter.create<StoreOp>(loc, productResult, result, outerLoop.getInductionVar());

      rewriter.setInsertionPointAfter(outerLoop);
      return mlir::success();
    }
  };

  /// Product of a matrix (2-D array) and a vector (1-D array).
  ///
  /// [ x11, x12 ] * [ y1, y2 ] = [ x11 * y1 + x12 * y2 ]
  /// [ x21, x22 ]							  [ x21 * y1 + x22 * y2 ]
  /// [ x31, x32 ]								[ x31 * y1 + x22 * y2 ]
  struct MulOpMatrixVectorLowering : public ModelicaOpRewritePattern<MulOp>
  {
    using ModelicaOpRewritePattern<MulOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Matrix-vector product: left-hand side value is not an array");
      }

      auto lhsArrayType = op.lhs().getType().cast<ArrayType>();

      if (lhsArrayType.getRank() != 2) {
        return rewriter.notifyMatchFailure(op, "Matrix-vector product: left-hand size array is not 2-D");
      }

      if (!op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Matrix-vector product: right-hand side value is not an array");
      }

      auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

      if (rhsArrayType.getRank() != 1) {
        return rewriter.notifyMatchFailure(op, "Matrix-vector product: right-hand side matrix is not 1-D");
      }

      if (lhsArrayType.getShape()[1] != -1 && rhsArrayType.getShape()[0] != -1) {
        if (lhsArrayType.getShape()[1] != rhsArrayType.getShape()[0]) {
          return rewriter.notifyMatchFailure(op, "Matrix-vector product: incompatible shapes");
        }
      }

      assert(lhsArrayType.getRank() == 2);
      assert(rhsArrayType.getRank() == 1);

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        if (lhsShape[1] == -1 || rhsShape[0] == -1) {
          mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
          mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), one);
          mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), zero);
          mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
          rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto shape = resultArrayType.getShape();

      llvm::SmallVector<mlir::Value, 1> dynamicDimensions;

      if (shape[0] == -1) {
        dynamicDimensions.push_back(rewriter.create<DimOp>(
            loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0))));
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Iterate on the rows
      mlir::Value rowsLowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value rowsUpperBound = rewriter.create<DimOp>(loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value rowsStep = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto outerLoop = rewriter.create<mlir::scf::ParallelOp>(loc, rowsLowerBound, rowsUpperBound, rowsStep);
      rewriter.setInsertionPointToStart(outerLoop.getBody());

      // Product between the current row and the vector
      mlir::Value lowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value upperBound = rewriter.create<DimOp>(loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));
      mlir::Value init = rewriter.create<mlir::ConstantOp>(loc, getZeroAttr(resultArrayType.getElementType()));

      auto innerLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);
      rewriter.setInsertionPointToStart(innerLoop.getBody());

      mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), mlir::ValueRange({ outerLoop.getInductionVars()[0], innerLoop.getInductionVar() }));
      mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), innerLoop.getInductionVar());
      mlir::Value product = rewriter.create<MulOp>(loc, resultArrayType.getElementType(), lhs, rhs);
      mlir::Value sum = rewriter.create<AddOp>(loc, resultArrayType.getElementType(), product, innerLoop.getRegionIterArgs()[0]);
      rewriter.create<mlir::scf::YieldOp>(loc, sum);

      // Store the product in the result array
      rewriter.setInsertionPointAfter(innerLoop);
      mlir::Value productResult = innerLoop.getResult(0);
      rewriter.create<StoreOp>(loc, productResult, result, outerLoop.getInductionVars()[0]);

      rewriter.setInsertionPointAfter(outerLoop);

      return mlir::success();
    }
  };

  /// Product of two matrices (2-D arrays).
  ///
  /// [ x11, x12, x13 ] * [ y11, y12 ]  =  [ x11 * y11 + x12 * y21 + x13 * y31, x11 * y12 + x12 * y22 + x13 * y32 ]
  /// [ x21, x22, x23 ]   [ y21, y22 ]		 [ x21 * y11 + x22 * y21 + x23 * y31, x21 * y12 + x22 * y22 + x23 * y32 ]
  /// [ x31, x32, x33 ]	  [ y31, y32 ]		 [ x31 * y11 + x32 * y21 + x33 * y31, x31 * y12 + x32 * y22 + x33 * y32 ]
  /// [ x41, x42, x43 ]
  struct MulOpMatrixLowering : public ModelicaOpRewritePattern<MulOp>
  {
    using ModelicaOpRewritePattern<MulOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Matrix product: left-hand side value is not an array");
      }

      auto lhsArrayType = op.lhs().getType().cast<ArrayType>();

      if (lhsArrayType.getRank() != 2) {
        return rewriter.notifyMatchFailure(op, "Matrix product: left-hand size array is not 2-D");
      }

      if (!op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Matrix product: right-hand side value is not an array");
      }

      auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

      if (rhsArrayType.getRank() != 2) {
        return rewriter.notifyMatchFailure(op, "Matrix product: right-hand side matrix is not 2-D");
      }

      if (lhsArrayType.getShape()[1] != -1 && rhsArrayType.getShape()[0] != -1) {
        if (lhsArrayType.getShape()[1] != rhsArrayType.getShape()[0]) {
          return rewriter.notifyMatchFailure(op, "Matrix-vector product: incompatible shapes");
        }
      }

      assert(lhsArrayType.getRank() == 2);
      assert(rhsArrayType.getRank() == 2);

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        if (lhsShape[1] == -1 || rhsShape[0] == -1) {
          mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
          mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), one);
          mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), zero);
          mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
          rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto shape = resultArrayType.getShape();

      llvm::SmallVector<mlir::Value, 2> dynamicDimensions;

      if (shape[0] == -1) {
        dynamicDimensions.push_back(rewriter.create<DimOp>(
            loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0))));
      }

      if (shape[1] == -1) {
        dynamicDimensions.push_back(rewriter.create<DimOp>(
            loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1))));
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Iterate on the rows
      mlir::Value rowsLowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value rowsUpperBound = rewriter.create<DimOp>(loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value rowsStep = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto rowsLoop = rewriter.create<mlir::scf::ForOp>(loc, rowsLowerBound, rowsUpperBound, rowsStep);
      rewriter.setInsertionPointToStart(rowsLoop.getBody());

      // Iterate on the columns
      mlir::Value columnsLowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value columnsUpperBound = rewriter.create<DimOp>(loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1)));
      mlir::Value columnsStep = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto columnsLoop = rewriter.create<mlir::scf::ForOp>(loc, columnsLowerBound, columnsUpperBound, columnsStep);
      rewriter.setInsertionPointToStart(columnsLoop.getBody());

      // Product between the current row and the current column
      mlir::Value lowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value upperBound = rewriter.create<DimOp>(loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

      mlir::Value init = rewriter.create<ConstantOp>(loc, getZeroAttr(resultArrayType.getElementType()));

      auto innerLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);
      rewriter.setInsertionPointToStart(innerLoop.getBody());

      mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), mlir::ValueRange({ rowsLoop.getInductionVar(), innerLoop.getInductionVar() }));
      mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), mlir::ValueRange({ innerLoop.getInductionVar(), columnsLoop.getInductionVar() }));
      mlir::Value product = rewriter.create<MulOp>(loc, resultArrayType.getElementType(), lhs, rhs);
      mlir::Value sum = rewriter.create<AddOp>(loc, resultArrayType.getElementType(), product, innerLoop.getRegionIterArgs()[0]);
      rewriter.create<mlir::scf::YieldOp>(loc, sum);

      // Store the product in the result array
      rewriter.setInsertionPointAfter(innerLoop);
      mlir::Value productResult = innerLoop.getResult(0);
      rewriter.create<StoreOp>(loc, productResult, result, mlir::ValueRange({ rowsLoop.getInductionVar(), columnsLoop.getInductionVar() }));

      rewriter.setInsertionPointAfter(rowsLoop);

      return mlir::success();
    }
  };

  struct MulEWOpArrayLowering : public ModelicaOpRewritePattern<MulEWOp>
  {
    using ModelicaOpRewritePattern<MulEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(MulEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");
      }

      if (!op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");
      }

      auto lhsArrayType = op.lhs().getType().cast<ArrayType>();
      auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

      for (auto pair : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        auto lhsDimension = std::get<0>(pair);
        auto rhsDimension = std::get<1>(pair);

        if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension) {
          return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
        }
      }

      if (!isNumeric(lhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");
      }

      if (!isNumeric(rhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");
      }

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        assert(lhsArrayType.getRank() == rhsArrayType.getRank());

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == -1 || rhsShape[i] == -1) {
            mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(location, op.lhs(), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(location, op.rhs(), indices);
            mlir::Value value = nestedBuilder.create<MulOp>(location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct MulEWOpMixedLowering : public ModelicaOpRewritePattern<MulEWOp>
  {
    using ModelicaOpRewritePattern<MulEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(MulEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>() && !op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "None of the operands is an array");
      }

      if (op.lhs().getType().isa<ArrayType>() && !isNumeric(op.rhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a scalar");
      }

      if (op.rhs().getType().isa<ArrayType>() && !isNumeric(op.lhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a scalar");
      }

      mlir::Value array = op.lhs().getType().isa<ArrayType>() ? op.lhs() : op.rhs();
      mlir::Value scalar = op.lhs().getType().isa<ArrayType>() ? op.rhs() : op.lhs();

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, array);
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, array, indices);
            mlir::Value value = nestedBuilder.create<MulOp>(location, resultArrayType.getElementType(), arrayValue, scalar);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  /// Division between an array and a scalar value.
  struct DivOpArrayLowering : public ModelicaOpRewritePattern<DivOp>
  {
    using ModelicaOpRewritePattern<DivOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(DivOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Array-scalar division: left-hand size value is not an array");
      }

      if (!isNumeric(op.lhs().getType().cast<ArrayType>().getElementType())) {
        return rewriter.notifyMatchFailure(op, "Array-scalar division: right-hand side array has not numeric elements");
      }

      if (!isNumeric(op.rhs())) {
        return rewriter.notifyMatchFailure(op, "Array-scalar division: right-hand size value is not a scalar");
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Divide each array element by the scalar value
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.lhs(), indices);
            mlir::Value value = nestedBuilder.create<DivOp>(location, resultArrayType.getElementType(), arrayValue, op.rhs());
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct DivEWOpArrayLowering : public ModelicaOpRewritePattern<DivEWOp>
  {
    using ModelicaOpRewritePattern<DivEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(DivEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");
      }

      if (!op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");
      }

      auto lhsArrayType = op.lhs().getType().cast<ArrayType>();
      auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

      for (auto pair : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        auto lhsDimension = std::get<0>(pair);
        auto rhsDimension = std::get<1>(pair);

        if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension) {
          return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
        }
      }

      if (!isNumeric(lhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");
      }

      if (!isNumeric(rhsArrayType.getElementType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");
      }

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        assert(lhsArrayType.getRank() == rhsArrayType.getRank());

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == -1 || rhsShape[i] == -1) {
            mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(location, op.lhs(), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(location, op.rhs(), indices);
            mlir::Value value = nestedBuilder.create<DivOp>(location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct DivEWOpMixedLowering : public ModelicaOpRewritePattern<DivEWOp>
  {
    using ModelicaOpRewritePattern<DivEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(DivEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<ArrayType>() && !op.rhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "None of the operands is an array");
      }

      if (op.lhs().getType().isa<ArrayType>() && !isNumeric(op.rhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a scalar");
      }

      if (op.rhs().getType().isa<ArrayType>() && !isNumeric(op.lhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a scalar");
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs().getType().isa<ArrayType>() ? op.lhs() : op.rhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.lhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            if (op.lhs().getType().isa<ArrayType>()) {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.lhs(), indices);
              mlir::Value value = nestedBuilder.create<DivOp>(location, resultArrayType.getElementType(), arrayValue, op.rhs());
              nestedBuilder.create<StoreOp>(location, value, result, indices);
            } else {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.rhs(), indices);
              mlir::Value value = nestedBuilder.create<DivOp>(location, resultArrayType.getElementType(), op.lhs(), arrayValue);
              nestedBuilder.create<StoreOp>(location, value, result, indices);
            }
          });

      return mlir::success();
    }
  };

  struct PowOpMatrixLowering: public ModelicaOpRewritePattern<PowOp>
  {
    using ModelicaOpRewritePattern<PowOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(PowOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.base().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Base is not an array");
      }

      auto baseArrayType = op.base().getType().cast<ArrayType>();

      if (baseArrayType.getRank() != 2) {
        return rewriter.notifyMatchFailure(op, "Base array is not 2-D");
      }

      if (baseArrayType.getShape()[0] != -1 && baseArrayType.getShape()[1] != -1) {
        if (baseArrayType.getShape()[0] != baseArrayType.getShape()[1]) {
          return rewriter.notifyMatchFailure(op, "Base is not a square matrix");
        }
      }

      if (!op.exponent().getType().isa<IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Exponent is not an integer");
      }

      assert(baseArrayType.getRank() == 2);

      if (options.assertions) {
        // Check if the matrix is a square one
        auto shape = baseArrayType.getShape();

        if (shape[0] == -1 || shape[1] == -1) {
          mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
          mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.base(), one);
          mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.base(), zero);
          mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
          rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Base matrix is not squared"));
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.base());
      mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
      mlir::Value size = rewriter.create<DimOp>(loc, op.base(), one);
      mlir::Value result = rewriter.replaceOpWithNewOp<IdentityOp>(op, resultArrayType, size);

      // Compute the result
      mlir::Value exponent = rewriter.create<CastOp>(loc, rewriter.getIndexType(), op.exponent());
      mlir::Value lowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));
      mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto forLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, exponent, step);

      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forLoop.getBody());
        mlir::Value next = rewriter.create<MulOp>(loc, result.getType(), result, op.base());
        rewriter.create<AssignmentOp>(loc, result, next);
      }

      return mlir::success();
    }
  };

  struct NDimsOpLowering : public ModelicaOpRewritePattern<NDimsOp>
  {
    using ModelicaOpRewritePattern<NDimsOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(NDimsOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      auto arrayType = op.array().getType().cast<ArrayType>();
      mlir::Value result = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(arrayType.getRank()));
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct SizeOpArrayLowering : public ModelicaOpRewritePattern<SizeOp>
  {
    using ModelicaOpRewritePattern<SizeOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SizeOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op->getLoc();

      if (op.hasDimension()) {
        return rewriter.notifyMatchFailure(op, "Index specified");
      }

      assert(op.getResult().getType().isa<ArrayType>());
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, llvm::None);

      // Iterate on each dimension
      mlir::Value zeroValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));

      auto arrayType = op.array().getType().cast<ArrayType>();
      mlir::Value rank = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(arrayType.getRank()));
      mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto loop = rewriter.create<mlir::scf::ForOp>(loc, zeroValue, rank, step);

      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loop.getBody());

        // Get the size of the current dimension
        mlir::Value dimensionSize = rewriter.create<SizeOp>(loc, resultArrayType.getElementType(), op.array(), loop.getInductionVar());

        // Store it into the result array
        rewriter.create<StoreOp>(loc, dimensionSize, result, loop.getInductionVar());
      }

      return mlir::success();
    }
  };

  struct SizeOpDimensionLowering : public ModelicaOpRewritePattern<SizeOp>
  {
    using ModelicaOpRewritePattern<SizeOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SizeOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      if (!op.hasDimension()) {
        return rewriter.notifyMatchFailure(op, "No index specified");
      }

      mlir::Value index = rewriter.create<CastOp>(loc, rewriter.getIndexType(), op.dimension());
      mlir::Value result = rewriter.create<DimOp>(loc, op.array(), index);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };
}

namespace
{
  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class ModelicaOpConversionPattern : public mlir::OpConversionPattern<Op>
  {
    protected:
      using Adaptor = typename Op::Adaptor;

    public:
      ModelicaOpConversionPattern(mlir::MLIRContext* ctx, TypeConverter& typeConverter, ModelicaConversionOptions options)
          : mlir::OpConversionPattern<Op>(typeConverter, ctx, 1),
            options(std::move(options))
      {
      }

      mlir::modelica::TypeConverter& typeConverter() const
      {
        return *static_cast<mlir::modelica::TypeConverter *>(this->getTypeConverter());
      }

      mlir::Type convertType(mlir::Type type) const
      {
        return typeConverter().convertType(type);
      }

      mlir::Value materializeTargetConversion(mlir::OpBuilder& builder, mlir::Value value) const
      {
        mlir::Type type = this->getTypeConverter()->convertType(value.getType());
        return this->getTypeConverter()->materializeTargetConversion(builder, value.getLoc(), type, value);
      }

      void materializeTargetConversion(mlir::OpBuilder& builder, llvm::SmallVectorImpl<mlir::Value>& values) const
      {
        for (auto& value : values)
          value = materializeTargetConversion(builder, value);
      }

      std::string getMangledFunctionName(llvm::StringRef name, llvm::Optional<mlir::Type> returnType, mlir::ValueRange args) const
      {
        return getMangledFunctionName(name, returnType, args.getTypes());
      }

      std::string getMangledFunctionName(llvm::StringRef name, llvm::Optional<mlir::Type> returnType, mlir::TypeRange argsTypes) const
      {
        std::string resultType = returnType.hasValue() ? getMangledType(*returnType) : "void";
        std::string result = "_M" + name.str() + "_" + resultType;

        for (const auto& argType : argsTypes) {
          result += "_" + getMangledType(argType);
        }

        return result;
      }

    private:
      std::string getMangledType(mlir::Type type) const
      {
        if (auto booleanType = type.dyn_cast<BooleanType>()) {
          return "i1";
        }

        if (auto integerType = type.dyn_cast<IntegerType>()) {
          return "i" + std::to_string(convertType(integerType).getIntOrFloatBitWidth());
        }

        if (auto realType = type.dyn_cast<RealType>()) {
          return "f" + std::to_string(convertType(realType).getIntOrFloatBitWidth());
        }

        if (auto arrayType = type.dyn_cast<UnsizedArrayType>()) {
          return "a" + getMangledType(arrayType.getElementType());
        }

        if (auto indexType = type.dyn_cast<mlir::IndexType>()) {
          return getMangledType(this->getTypeConverter()->convertType(type));
        }

        if (auto integerType = type.dyn_cast<mlir::IntegerType>()) {
          return "i" + std::to_string(integerType.getWidth());
        }

        if (auto floatType = type.dyn_cast<mlir::FloatType>()) {
          return "f" + std::to_string(floatType.getWidth());
        }

        llvm_unreachable("Unknown type for mangling");
        return "unknown";
      }

    protected:
      ModelicaConversionOptions options;
  };

  class ConstantOpLowering : public ModelicaOpConversionPattern<ConstantOp>
  {
    public:
      using ModelicaOpConversionPattern<ConstantOp>::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(ConstantOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
      {
        auto attribute = convertAttribute(rewriter, op.getResult().getType(), op.value());
        rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, attribute);
        return mlir::success();
      }

    private:
      mlir::Attribute convertAttribute(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Attribute attribute) const
      {
        if (attribute.getType().isa<mlir::IndexType>()) {
          return attribute;
        }

        resultType = getTypeConverter()->convertType(resultType);

        if (auto booleanAttribute = attribute.dyn_cast<BooleanAttr>()) {
          return builder.getBoolAttr(booleanAttribute.getValue());
        }

        if (auto integerAttribute = attribute.dyn_cast<IntegerAttr>()) {
          return builder.getIntegerAttr(resultType, integerAttribute.getValue());
        }

        if (auto realAttribute = attribute.dyn_cast<RealAttr>()) {
          return builder.getFloatAttr(resultType, realAttribute.getValue());
        }

        llvm_unreachable("Unknown attribute type");
        return nullptr;
      }
  };

  struct NotOpLowering : public ModelicaOpConversionPattern<NotOp>
  {
    using ModelicaOpConversionPattern<NotOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(NotOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      Adaptor transformed(operands);

      // Check if the operand is compatible
      if (!op.operand().getType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Operand is not a Boolean");
      }

      // There is no native negate operation in LLVM IR, so we need to leverage
      // a property of the XOR operation: x XOR true = NOT x
      mlir::Value trueValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(true));
      mlir::Value result = rewriter.create<mlir::XOrOp>(loc, trueValue, transformed.operand());
      result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);

      return mlir::success();
    }
  };

  struct AndOpLowering : public ModelicaOpConversionPattern<AndOp>
  {
    using ModelicaOpConversionPattern<AndOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AndOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      Adaptor transformed(operands);

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a Boolean");
      }

      if (!op.rhs().getType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a Boolean");
      }

      // Compute the result
      mlir::Value result = rewriter.create<mlir::AndOp>(loc, transformed.lhs(), transformed.rhs());
      result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);

      return mlir::success();
    }
  };

  struct OrOpLowering : public ModelicaOpConversionPattern<OrOp>
  {
    using ModelicaOpConversionPattern<OrOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(OrOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      Adaptor transformed(operands);

      // Check if the operands are compatible
      if (!op.lhs().getType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a Boolean");
      }

      if (!op.rhs().getType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a Boolean");
      }

      // Compute the result
      mlir::Value result = rewriter.create<mlir::OrOp>(loc, transformed.lhs(), transformed.rhs());
      result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);

      return mlir::success();
    }
  };

  template<typename Op>
  struct ComparisonOpLowering : public ModelicaOpConversionPattern<Op>
  {
    using Adaptor = typename ModelicaOpConversionPattern<Op>::Adaptor;
    using ModelicaOpConversionPattern<Op>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!isNumeric(op.lhs()) || !isNumeric(op.rhs()))
        return rewriter.notifyMatchFailure(op, "Unsupported types");

      // Cast the operands to the most generic type, in order to avoid
      // information loss.
      llvm::SmallVector<mlir::Value, 3> castedOperands;
      mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
      Adaptor adaptor(castedOperands);

      // Compute the result
      if (type.isa<mlir::IndexType, BooleanType, IntegerType>()) {
        mlir::Value result = compareIntegers(rewriter, loc, adaptor.lhs(), adaptor.rhs());
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      if (type.isa<RealType>()) {
        mlir::Value result = compareReals(rewriter, loc, adaptor.lhs(), adaptor.rhs());
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      return mlir::failure();
    }

    virtual mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const = 0;
    virtual mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const = 0;
  };

  struct EqOpLowering : public ComparisonOpLowering<EqOp>
  {
    using ComparisonOpLowering<EqOp>::ComparisonOpLowering;

    mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpIOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpIPredicate::eq,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }

    mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpFOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpFPredicate::OEQ,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }
  };

  struct NotEqOpLowering : public ComparisonOpLowering<NotEqOp>
  {
    using ComparisonOpLowering<NotEqOp>::ComparisonOpLowering;

    mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpIOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpIPredicate::ne,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }

    mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpFOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpFPredicate::ONE,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }
  };

  struct GtOpLowering : public ComparisonOpLowering<GtOp>
  {
    using ComparisonOpLowering<GtOp>::ComparisonOpLowering;

    mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpIOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpIPredicate::sgt,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }

    mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpFOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpFPredicate::OGT,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }
  };

  struct GteOpLowering : public ComparisonOpLowering<GteOp>
  {
    using ComparisonOpLowering<GteOp>::ComparisonOpLowering;

    mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpIOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpIPredicate::sge,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }

    mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpFOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpFPredicate::OGE,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }
  };

  struct LtOpLowering : public ComparisonOpLowering<LtOp>
  {
    using ComparisonOpLowering<LtOp>::ComparisonOpLowering;

    mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpIOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpIPredicate::slt,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }

    mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpFOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpFPredicate::OLT,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }
  };

  struct LteOpLowering : public ComparisonOpLowering<LteOp>
  {
    using ComparisonOpLowering<LteOp>::ComparisonOpLowering;

    mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpIOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpIPredicate::sle,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }

    mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
    {
      mlir::Value result = builder.create<mlir::CmpFOp>(
          loc,
          builder.getIntegerType(1),
          mlir::CmpFPredicate::OLE,
          materializeTargetConversion(builder, lhs),
          materializeTargetConversion(builder, rhs));

      return getTypeConverter()->materializeSourceConversion(
          builder, loc, BooleanType::get(result.getContext()), result);
    }
  };

  struct NegateOpLowering : public ModelicaOpConversionPattern<NegateOp>
  {
    using ModelicaOpConversionPattern<NegateOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(NegateOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operand is compatible
      if (!isNumeric(op.operand())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not a scalar");
      }

      Adaptor transformed(operands);
      mlir::Type type = op.operand().getType();

      // Compute the result
      if (type.isa<mlir::IndexType, BooleanType, IntegerType>()) {
        mlir::Value zeroValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getZeroAttr(transformed.operand().getType()));
        mlir::Value result = rewriter.create<mlir::SubIOp>(loc, zeroValue, transformed.operand());
        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      if (type.isa<RealType>()) {
        mlir::Value result = rewriter.create<mlir::NegFOp>(loc, transformed.operand());
        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown type");
    }
  };
  
  template<typename Op>
  struct AddOpLikeLowering : public ModelicaOpConversionPattern<Op>
  {
    using ModelicaOpConversionPattern<Op>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!isNumeric(op.lhs())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not a scalar");
      }

      if (!isNumeric(op.rhs())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not a scalar");
      }

      // Cast the operands to the most generic type, in order to avoid information loss.
      llvm::SmallVector<mlir::Value, 3> castedOperands;
      mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
      this->materializeTargetConversion(rewriter, castedOperands);
      typename ModelicaOpConversionPattern<Op>::Adaptor transformed(castedOperands);

      // Compute the result
      if (type.isa<mlir::IndexType, BooleanType, IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::AddIOp>(loc, transformed.lhs(), transformed.rhs());
        result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      if (type.isa<RealType>()) {
        mlir::Value result = rewriter.create<mlir::AddFOp>(loc, transformed.lhs(), transformed.rhs());
        result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown type");
    }
  };

  struct AddOpLowering : public AddOpLikeLowering<AddOp>
  {
    using AddOpLikeLowering<AddOp>::AddOpLikeLowering;
  };

  struct AddEWOpLowering : public AddOpLikeLowering<AddEWOp>
  {
    using AddOpLikeLowering<AddEWOp>::AddOpLikeLowering;
  };
  
  template<typename Op>
  struct SubOpLikeLowering : public ModelicaOpConversionPattern<Op>
  {
    using ModelicaOpConversionPattern<Op>::ModelicaOpConversionPattern;
    
    mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!isNumeric(op.lhs())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not a scalar");
      }

      if (!isNumeric(op.rhs())) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not a scalar");
      }

      // Cast the operands to the most generic type, in order to avoid
      // information loss.
      llvm::SmallVector<mlir::Value, 3> castedOperands;
      mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
      this->materializeTargetConversion(rewriter, castedOperands);
      typename ModelicaOpConversionPattern<Op>::Adaptor transformed(castedOperands);

      // Compute the result
      if (type.isa<mlir::IndexType, BooleanType, IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::SubIOp>(loc, transformed.lhs(), transformed.rhs());
        result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      if (type.isa<RealType>()) {
        mlir::Value result = rewriter.create<mlir::SubFOp>(loc, transformed.lhs(), transformed.rhs());
        result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown type");
    }
  };

  struct SubOpLowering : public SubOpLikeLowering<SubOp>
  {
    using SubOpLikeLowering<SubOp>::SubOpLikeLowering;
  };

  struct SubEWOpLowering : public SubOpLikeLowering<SubEWOp>
  {
    using SubOpLikeLowering<SubEWOp>::SubOpLikeLowering;
  };
  
  template<typename Op>
  struct MulOpLikeLowering : public ModelicaOpConversionPattern<Op>
  {
    using ModelicaOpConversionPattern<Op>::ModelicaOpConversionPattern;
    
    mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!isNumeric(op.lhs())) {
        return rewriter.notifyMatchFailure(op, "Scalar-scalar product: left-hand side value is not a scalar");
      }

      if (!isNumeric(op.rhs())) {
        return rewriter.notifyMatchFailure(op, "Scalar-scalar product: right-hand side value is not a scalar");
      }

      // Cast the operands to the most generic type, in order to avoid
      // information loss.
      llvm::SmallVector<mlir::Value, 3> castedOperands;
      mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
      this->materializeTargetConversion(rewriter, castedOperands);
      typename ModelicaOpConversionPattern<Op>::Adaptor transformed(castedOperands);

      // Compute the result
      if (type.isa<mlir::IndexType, BooleanType, IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::MulIOp>(loc, transformed.lhs(), transformed.rhs());
        result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      if (type.isa<RealType>()) {
        mlir::Value result = rewriter.create<mlir::MulFOp>(loc, transformed.lhs(), transformed.rhs());
        result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown type");
    }
  };

  struct MulOpLowering : public MulOpLikeLowering<MulOp>
  {
    using MulOpLikeLowering<MulOp>::MulOpLikeLowering;
  };

  struct MulEWOpLowering : public MulOpLikeLowering<MulEWOp>
  {
    using MulOpLikeLowering<MulEWOp>::MulOpLikeLowering;
  };
  
  template<typename Op>
  struct DivOpLikeLowering : public ModelicaOpConversionPattern<Op>
  {
    using ModelicaOpConversionPattern<Op>::ModelicaOpConversionPattern;
    
    mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!isNumeric(op.lhs())) {
        return rewriter.notifyMatchFailure(op, "Scalar-scalar division: left-hand side value is not a scalar");
      }

      if (!isNumeric(op.rhs())) {
        return rewriter.notifyMatchFailure(op, "Scalar-scalar division: right-hand side value is not a scalar");
      }

      // Cast the operands to the most generic type, in order to avoid
      // information loss.
      llvm::SmallVector<mlir::Value, 3> castedOperands;
      mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
      this->materializeTargetConversion(rewriter, castedOperands);
      typename ModelicaOpConversionPattern<Op>::Adaptor transformed(castedOperands);

      // Compute the result
      if (type.isa<mlir::IndexType, BooleanType, IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::SignedDivIOp>(loc, transformed.lhs(), transformed.rhs());
        result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      if (type.isa<RealType>()) {
        mlir::Value result = rewriter.create<mlir::DivFOp>(loc, transformed.lhs(), transformed.rhs());
        result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown type");
    }
  };

  struct DivOpLowering : public DivOpLikeLowering<DivOp>
  {
    using DivOpLikeLowering<DivOp>::DivOpLikeLowering;
  };

  struct DivEWOpLowering : public DivOpLikeLowering<DivEWOp>
  {
    using DivOpLikeLowering<DivEWOp>::DivOpLikeLowering;
  };

  struct PowOpLowering: public ModelicaOpConversionPattern<PowOp>
  {
    using ModelicaOpConversionPattern<PowOp>::ModelicaOpConversionPattern;
    
    mlir::LogicalResult matchAndRewrite(PowOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      // Check if the operands are compatible
      if (!isNumeric(op.base())) {
        return rewriter.notifyMatchFailure(op, "Base is not a scalar");
      }

      if (!isNumeric(op.exponent())) {
        return rewriter.notifyMatchFailure(op, "Base is not a scalar");
      }

      // Compute the result
      llvm::SmallVector<mlir::Value, 3> args;
      args.push_back(op.base());
      args.push_back(op.exponent());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("pow", op.getResult().getType(), args),
          op.getResult().getType(),
          mlir::ValueRange(args).getTypes());

      rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), op.getResult().getType(), args);
      return mlir::success();
    }
  };

  struct AbsOpLowering : public ModelicaOpConversionPattern<AbsOp>
  {
    using ModelicaOpConversionPattern<AbsOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AbsOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("abs", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct AcosOpLowering : public ModelicaOpConversionPattern<AcosOp>
  {
    using ModelicaOpConversionPattern<AcosOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AcosOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("acos", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct AsinOpLowering : public ModelicaOpConversionPattern<AsinOp>
  {
    using ModelicaOpConversionPattern<AsinOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AsinOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("asin", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct AtanOpLowering : public ModelicaOpConversionPattern<AtanOp>
  {
    using ModelicaOpConversionPattern<AtanOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AtanOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("atan", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct Atan2OpLowering : public ModelicaOpConversionPattern<Atan2Op>
  {
    using ModelicaOpConversionPattern<Atan2Op>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(Atan2Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      llvm::SmallVector<mlir::Value, 3> args;
      args.push_back(rewriter.create<CastOp>(loc, realType, op.y()));
      args.push_back(rewriter.create<CastOp>(loc, realType, op.x()));

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("atan2", realType, args),
          realType,
          args);

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, args).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct CosOpLowering : public ModelicaOpConversionPattern<CosOp>
  {
    using ModelicaOpConversionPattern<CosOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(CosOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("cos", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct CoshOpLowering : public ModelicaOpConversionPattern<CoshOp>
  {
    using ModelicaOpConversionPattern<CoshOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(CoshOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("cosh", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct DiagonalOpLowering : public ModelicaOpConversionPattern<DiagonalOp>
  {
    using ModelicaOpConversionPattern<DiagonalOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(DiagonalOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto arrayType = op.getResult().getType().cast<ArrayType>();

      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
      mlir::Value castedSize = nullptr;

      for (const auto& size : arrayType.getShape()) {
        if (size == -1) {
          if (castedSize == nullptr) {
            assert(op.values().getType().cast<ArrayType>().getRank() == 1);
            mlir::Value zeroValue = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
            castedSize = rewriter.create<DimOp>(loc, op.values(), zeroValue);
          }

          dynamicDimensions.push_back(castedSize);
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, arrayType, dynamicDimensions);

      llvm::SmallVector<mlir::Value, 3> args;

      args.push_back(rewriter.create<ArrayCastOp>(
          loc, result.getType().cast<ArrayType>().toUnsized(), result));

      args.push_back(rewriter.create<ArrayCastOp>(
          loc, op.values().getType().cast<ArrayType>().toUnsized(), op.values()));

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("diagonal", llvm::None, args),
          llvm::None,
          args);

      rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
      return mlir::success();
    }
  };

  struct ExpOpLowering : public ModelicaOpConversionPattern<ExpOp>
  {
    using ModelicaOpConversionPattern<ExpOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(ExpOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.exponent());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("exp", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct FillOpLowering : public ModelicaOpConversionPattern<ArrayFillOp>
  {
    using ModelicaOpConversionPattern<ArrayFillOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(ArrayFillOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto arrayType = op.array().getType().cast<ArrayType>();

      llvm::SmallVector<mlir::Value, 3> args;
      args.push_back(rewriter.create<ArrayCastOp>(loc, arrayType.toUnsized(), op.array()));
      args.push_back(rewriter.create<CastOp>(loc, arrayType.getElementType(), op.value()));

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("fill", llvm::None, args),
          llvm::None,
          mlir::ValueRange(args).getTypes());

      rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };

  struct IdentityOpLowering : public ModelicaOpConversionPattern<IdentityOp>
  {
    using ModelicaOpConversionPattern<IdentityOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(IdentityOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto arrayType = op.getResult().getType().cast<ArrayType>();

      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
      mlir::Value castedSize = nullptr;

      for (const auto& size : arrayType.getShape()) {
        if (size == -1) {
          if (castedSize == nullptr) {
            castedSize = rewriter.create<CastOp>(loc, rewriter.getIndexType(), op.size());
          }

          dynamicDimensions.push_back(castedSize);
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, arrayType, dynamicDimensions);

      mlir::Value arg = rewriter.create<ArrayCastOp>(
          loc, result.getType().cast<ArrayType>().toUnsized(), result);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("identity", llvm::None, arg),
          llvm::None,
          arg.getType());

      rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, arg);
      return mlir::success();
    }
  };

  struct LinspaceOpLowering : public ModelicaOpConversionPattern<LinspaceOp>
  {
    using ModelicaOpConversionPattern<LinspaceOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(LinspaceOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto arrayType = op.getResult().getType().cast<ArrayType>();

      assert(arrayType.getRank() == 1);
      mlir::Value size = rewriter.create<CastOp>(loc, rewriter.getIndexType(), op.amount());

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, arrayType, size);

      llvm::SmallVector<mlir::Value, 3> args;

      args.push_back(rewriter.create<ArrayCastOp>(
          loc, result.getType().cast<ArrayType>().toUnsized(), result));

      args.push_back(rewriter.create<CastOp>(
          loc, RealType::get(op->getContext()), op.begin()));

      args.push_back(rewriter.create<CastOp>(
          loc, RealType::get(op->getContext()), op.end()));

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("linspace", llvm::None, args),
          llvm::None,
          args);

      rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
      return mlir::success();
    }
  };

  struct LogOpLowering : public ModelicaOpConversionPattern<LogOp>
  {
    using ModelicaOpConversionPattern<LogOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(LogOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("log", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct Log10OpLowering : public ModelicaOpConversionPattern<Log10Op>
  {
    using ModelicaOpConversionPattern<Log10Op>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(Log10Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("log10", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct OnesOpLowering : public ModelicaOpConversionPattern<OnesOp>
  {
    using ModelicaOpConversionPattern<OnesOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(OnesOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto arrayType = op.getResult().getType().cast<ArrayType>();

      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (auto size : llvm::enumerate(arrayType.getShape())) {
        if (size.value() == -1) {
          dynamicDimensions.push_back(rewriter.create<CastOp>(loc, rewriter.getIndexType(), op.sizes()[size.index()]));
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, arrayType, dynamicDimensions);

      llvm::SmallVector<mlir::Value, 3> args;

      args.push_back(rewriter.create<ArrayCastOp>(
          loc, result.getType().cast<ArrayType>().toUnsized(), result));

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("ones", llvm::None, args),
          llvm::None,
          args);

      rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
      return mlir::success();
    }
  };

  struct MaxOpArrayLowering : public ModelicaOpConversionPattern<MaxOp>
  {
    using ModelicaOpConversionPattern<MaxOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(MaxOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      if (op.getNumOperands() != 1) {
        return rewriter.notifyMatchFailure(op, "Operand is not an array");
      }

      mlir::Value operand = op.first();

      // If there is just one operand, then it is for sure an array, thanks
      // to the operation verification.

      assert(operand.getType().isa<ArrayType>() && isNumeric(operand.getType().cast<ArrayType>().getElementType()));

      auto arrayType = operand.getType().cast<ArrayType>();
      operand = rewriter.create<ArrayCastOp>(loc, arrayType.toUnsized(), operand);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("max", arrayType.getElementType(), operand),
          arrayType.getElementType(),
          operand);

      auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), arrayType.getElementType(), operand);
      assert(call.getNumResults() == 1);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), call->getResult(0));

      return mlir::success();
    }
  };

  struct MaxOpScalarsLowering : public ModelicaOpConversionPattern<MaxOp>
  {
    using ModelicaOpConversionPattern<MaxOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(MaxOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      if (op.getNumOperands() != 2) {
        return rewriter.notifyMatchFailure(op, "Operands are not scalars");
      }

      // If there are two operands then they are for sure scalars, thanks
      // to the operation verification.

      llvm::SmallVector<mlir::Value, 2> values;
      values.push_back(op.first());
      values.push_back(op.second());
      assert(isNumeric(values[0]) && isNumeric(values[1]));

      llvm::SmallVector<mlir::Value, 2> castedOperands;
      mlir::Type type = castToMostGenericType(rewriter, values, castedOperands);
      materializeTargetConversion(rewriter, castedOperands);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("max", type, castedOperands),
          type,
          castedOperands);

      auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), type, castedOperands);
      assert(call.getNumResults() == 1);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), call->getResult(0));

      return mlir::success();
    }
  };

  struct MinOpArrayLowering : public ModelicaOpConversionPattern<MinOp>
  {
    using ModelicaOpConversionPattern<MinOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(MinOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      if (op.getNumOperands() != 1) {
        return rewriter.notifyMatchFailure(op, "Operand is not an array");
      }

      mlir::Value operand = op.first();

      // If there is just one operand, then it is for sure an array, thanks
      // to the operation verification.

      assert(operand.getType().isa<ArrayType>() && isNumeric(operand.getType().cast<ArrayType>().getElementType()));

      auto arrayType = operand.getType().cast<ArrayType>();
      operand = rewriter.create<ArrayCastOp>(loc, arrayType.toUnsized(), operand);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("min", arrayType.getElementType(), operand),
          arrayType.getElementType(),
          operand);

      auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), arrayType.getElementType(), operand);
      assert(call.getNumResults() == 1);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), call->getResult(0));

      return mlir::success();
    }
  };

  struct MinOpScalarsLowering : public ModelicaOpConversionPattern<MinOp>
  {
    using ModelicaOpConversionPattern<MinOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(MinOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      if (op.getNumOperands() != 2) {
        return rewriter.notifyMatchFailure(op, "Operands are not scalars");
      }

      // If there are two operands then they are for sure scalars, thanks
      // to the operation verification.

      llvm::SmallVector<mlir::Value, 2> values;
      values.push_back(op.first());
      values.push_back(op.second());

      assert(isNumeric(values[0]) && isNumeric(values[1]));

      llvm::SmallVector<mlir::Value, 3> castedOperands;
      mlir::Type type = castToMostGenericType(rewriter, values, castedOperands);
      materializeTargetConversion(rewriter, castedOperands);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("min", type, castedOperands),
          type,
          castedOperands);

      auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), type, castedOperands);
      assert(call.getNumResults() == 1);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), call->getResult(0));

      return mlir::success();
    }
  };

  struct ProductOpLowering : public ModelicaOpConversionPattern<ProductOp>
  {
    using ModelicaOpConversionPattern<ProductOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(ProductOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto arrayType = op.array().getType().cast<ArrayType>();

      mlir::Value arg = rewriter.create<ArrayCastOp>(
          loc, op.array().getType().cast<ArrayType>().toUnsized(), op.array());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("product", arrayType.getElementType(), arg),
          arrayType.getElementType(),
          arg);

      auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), arrayType.getElementType(), arg);
      assert(call.getNumResults() == 1);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), call->getResult(0));

      return mlir::success();
    }
  };

  struct SignOpLowering : public ModelicaOpConversionPattern<SignOp>
  {
    using ModelicaOpConversionPattern<SignOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SignOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto integerType = IntegerType::get(op.getContext());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("sign", integerType, op.operand()),
          integerType,
          op.operand().getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), integerType, op.operand()).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct SinOpLowering : public ModelicaOpConversionPattern<SinOp>
  {
    using ModelicaOpConversionPattern<SinOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SinOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("sin", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct SinhOpLowering : public ModelicaOpConversionPattern<SinhOp>
  {
    using ModelicaOpConversionPattern<SinhOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SinhOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("sinh", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct SqrtOpLowering : public ModelicaOpConversionPattern<SqrtOp>
  {
    using ModelicaOpConversionPattern<SqrtOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SqrtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("sqrt", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct SumOpLowering : public ModelicaOpConversionPattern<SumOp>
  {
    using ModelicaOpConversionPattern<SumOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SumOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto arrayType = op.array().getType().cast<ArrayType>();

      mlir::Value arg = rewriter.create<ArrayCastOp>(
          loc, op.array().getType().cast<ArrayType>().toUnsized(), op.array());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("sum", arrayType.getElementType(), arg),
          arrayType.getElementType(),
          arg);

      auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), arrayType.getElementType(), arg);
      assert(call.getNumResults() == 1);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), call->getResult(0));

      return mlir::success();
    }
  };

  struct SymmetricOpLowering : public ModelicaOpConversionPattern<SymmetricOp>
  {
    using ModelicaOpConversionPattern<SymmetricOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SymmetricOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto arrayType = op.matrix().getType().cast<ArrayType>();
      auto shape = arrayType.getShape();

      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (auto size : llvm::enumerate(shape)) {
        if (size.value() == -1) {
          mlir::Value index = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(size.index()));
          dynamicDimensions.push_back(rewriter.create<DimOp>(loc, op.matrix(), index));
        }
      }

      if (options.assertions) {
        // Check if the matrix is a square one
        if (shape[0] == -1 || shape[1] == -1) {
          mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
          mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.matrix(), one));
          mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.matrix(), zero));
          mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
          rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Base matrix is not squared"));
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, arrayType, dynamicDimensions);

      llvm::SmallVector<mlir::Value, 3> args;

      args.push_back(rewriter.create<ArrayCastOp>(
          loc, result.getType().cast<ArrayType>().toUnsized(), result));

      args.push_back(rewriter.create<ArrayCastOp>(
          loc, op.matrix().getType().cast<ArrayType>().toUnsized(), op.matrix()));

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("symmetric", llvm::None, args),
          llvm::None,
          args);

      rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
      return mlir::success();
    }
  };

  struct TanOpLowering : public ModelicaOpConversionPattern<TanOp>
  {
    using ModelicaOpConversionPattern<TanOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(TanOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("tan", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct TanhOpLowering : public ModelicaOpConversionPattern<TanhOp>
  {
    using ModelicaOpConversionPattern<TanhOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(TanhOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value arg = rewriter.create<CastOp>(loc, realType, op.operand());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("tanh", realType, arg),
          realType,
          arg.getType());

      mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct TransposeOpLowering : public ModelicaOpConversionPattern<TransposeOp>
  {
    using ModelicaOpConversionPattern<TransposeOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(TransposeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto arrayType = op.getResult().getType().cast<ArrayType>();
      auto shape = arrayType.getShape();

      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (auto size : llvm::enumerate(shape)) {
        if (size.value() == -1) {
          mlir::Value index = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(shape.size() - size.index() - 1));
          mlir::Value dim = rewriter.create<DimOp>(loc, op.matrix(), index);
          dynamicDimensions.push_back(dim);
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, arrayType, dynamicDimensions);

      llvm::SmallVector<mlir::Value, 3> args;

      args.push_back(rewriter.create<ArrayCastOp>(
          loc, result.getType().cast<ArrayType>().toUnsized(), result));

      args.push_back(rewriter.create<ArrayCastOp>(
          loc, op.matrix().getType().cast<ArrayType>().toUnsized(), op.matrix()));

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("transpose", llvm::None, args),
          llvm::None,
          args);

      rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
      return mlir::success();
    }
  };

  struct ZerosOpLowering : public ModelicaOpConversionPattern<ZerosOp>
  {
    using ModelicaOpConversionPattern<ZerosOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(ZerosOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto arrayType = op.getResult().getType().cast<ArrayType>();

      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (auto size : llvm::enumerate(arrayType.getShape()))
        if (size.value() == -1)
          dynamicDimensions.push_back(rewriter.create<CastOp>(loc, rewriter.getIndexType(), op.sizes()[size.index()]));

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, arrayType, dynamicDimensions);

      llvm::SmallVector<mlir::Value, 3> args;

      args.push_back(rewriter.create<ArrayCastOp>(
          loc, result.getType().cast<ArrayType>().toUnsized(), result));

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("zeros", llvm::None, args),
          llvm::None,
          args);

      rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
      return mlir::success();
    }
  };

  struct PrintOpLowering : public ModelicaOpConversionPattern<PrintOp>
  {
    using ModelicaOpConversionPattern<PrintOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(PrintOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      mlir::Value arg = op.value();

      if (auto arrayType = arg.getType().dyn_cast<ArrayType>()) {
        arg = rewriter.create<ArrayCastOp>(loc, arrayType.toUnsized(), arg);
      }

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("print", llvm::None, arg),
          llvm::None,
          arg.getType());

      rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, arg);
      return mlir::success();
    }
  };
}

static void populateModelicaConversionPatterns(
		mlir::OwningRewritePatternList& patterns,
		mlir::MLIRContext* context,
		mlir::modelica::TypeConverter& typeConverter,
		ModelicaConversionOptions options)
{
  patterns.insert<
      AssignmentOpScalarLowering,
      AssignmentOpArrayLowering,
      NotOpArrayLowering,
      AndOpArrayLowering,
      OrOpArrayLowering,
      NegateOpArrayLowering,
      AddOpArrayLowering,
      AddOpMixedLowering,
      AddEWOpArrayLowering,
      AddEWOpMixedLowering,
      SubOpArrayLowering,
      SubOpMixedLowering,
      SubEWOpArrayLowering,
      SubEWOpMixedLowering,
      MulOpScalarProductLowering,
      MulOpCrossProductLowering,
      MulOpVectorMatrixLowering,
      MulOpMatrixVectorLowering,
      MulOpMatrixLowering,
      MulEWOpArrayLowering,
      MulEWOpMixedLowering,
      DivOpArrayLowering,
      DivEWOpArrayLowering,
      DivEWOpMixedLowering,
      PowOpMatrixLowering,
      NDimsOpLowering,
      SizeOpArrayLowering,
      SizeOpDimensionLowering>(context, options);

  patterns.insert<
      ConstantOpLowering,
      NotOpLowering,
      AndOpLowering,
      OrOpLowering,
      EqOpLowering,
      NotEqOpLowering,
      GtOpLowering,
      GteOpLowering,
      LtOpLowering,
      LteOpLowering,
      NegateOpLowering,
      AddOpLowering,
      AddEWOpLowering,
      SubOpLowering,
      SubEWOpLowering,
      MulOpLowering,
      MulEWOpLowering,
      DivOpLowering,
      DivEWOpLowering,
      PowOpLowering,
      AbsOpLowering,
      AcosOpLowering,
      AsinOpLowering,
      AtanOpLowering,
      Atan2OpLowering,
      CosOpLowering,
      CoshOpLowering,
      DiagonalOpLowering,
      ExpOpLowering,
      FillOpLowering,
      IdentityOpLowering,
      LinspaceOpLowering,
      LogOpLowering,
      Log10OpLowering,
      OnesOpLowering,
      MaxOpArrayLowering,
      MaxOpScalarsLowering,
      MinOpArrayLowering,
      MinOpScalarsLowering,
      ProductOpLowering,
      SignOpLowering,
      SinOpLowering,
      SinhOpLowering,
      SqrtOpLowering,
      SumOpLowering,
      SymmetricOpLowering,
      TanOpLowering,
      TanhOpLowering,
      TransposeOpLowering,
      ZerosOpLowering,
      PrintOpLowering>(context, typeConverter, options);
}

class ModelicaConversionPass : public mlir::PassWrapper<ModelicaConversionPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
    ModelicaConversionPass(ModelicaConversionOptions options, unsigned int bitWidth)
        : options(std::move(options)),
          bitWidth(bitWidth)
    {
    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override
    {
      registry.insert<ModelicaDialect>();
      registry.insert<mlir::StandardOpsDialect>();
      registry.insert<mlir::scf::SCFDialect>();
      registry.insert<mlir::LLVM::LLVMDialect>();
    }

    void runOnOperation() override
	{
		if (mlir::failed(convertOperations())) {
			mlir::emitError(getOperation().getLoc(), "Error in converting the Modelica operations");
			return signalPassFailure();
		}
	}

	private:
    mlir::LogicalResult convertOperations()
    {
      auto module = getOperation();
      mlir::ConversionTarget target(getContext());

      target.addIllegalOp<
          ConstantOp,
          AssignmentOp,
          ArrayFillOp,
          NotOp, AndOp, OrOp,
          EqOp,
          NotEqOp,
          GtOp,
          GteOp,
          LtOp,
          LteOp,
          NegateOp,
          AddOp,
          AddEWOp,
          SubOp,
          SubEWOp,
          MulOp,
          MulEWOp,
          DivOp,
          DivEWOp,
          PowOp,
          PowEWOp>();

      target.addIllegalOp<
          AbsOp,
          AcosOp,
          AsinOp,
          AtanOp,
          Atan2Op,
          CosOp,
          CoshOp,
          DiagonalOp,
          ExpOp,
          IdentityOp,
          ArrayFillOp,
          LinspaceOp,
          LogOp,
          Log10Op,
          MaxOp,
          MinOp,
          NDimsOp,
          OnesOp,
          ProductOp,
          SignOp,
          SinOp,
          SinhOp,
          SizeOp,
          SqrtOp,
          SumOp,
          SymmetricOp,
          TanOp,
          TanhOp,
          TransposeOp,
          ZerosOp>();

      target.addIllegalOp<PrintOp>();

      target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
        return true;
      });

      mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
      TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

      mlir::OwningRewritePatternList patterns(&getContext());
      populateModelicaConversionPatterns(patterns, &getContext(), typeConverter, options);

      return applyPartialConversion(module, target, std::move(patterns));
    }

    private:
      ModelicaConversionOptions options;
      unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> marco::codegen::createModelicaConversionPass(ModelicaConversionOptions options, unsigned int bitWidth)
{
	return std::make_unique<ModelicaConversionPass>(options, bitWidth);
}
