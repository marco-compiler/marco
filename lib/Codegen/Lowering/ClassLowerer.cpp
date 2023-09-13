#include "marco/Codegen/Lowering/ClassLowerer.h"
#include "marco/Codegen/Lowering/ClassDependencyGraph.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ClassLowerer::ClassLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void ClassLowerer::declare(const ast::Class& cls)
  {
    if (auto model = cls.dyn_cast<ast::Model>()) {
      return declare(*model);
    }

    if (auto package = cls.dyn_cast<ast::Package>()) {
      return declare(*package);
    }

    if (auto function = cls.dyn_cast<ast::PartialDerFunction>()) {
      return declare(*function);
    }

    if (auto record = cls.dyn_cast<ast::Record>()) {
      return declare(*record);
    }

    if (auto standardFunction = cls.dyn_cast<ast::StandardFunction>()) {
      return declare(*standardFunction);
    }

    llvm_unreachable("Unknown class type");
  }

  void ClassLowerer::declareVariables(const ast::Class& cls)
  {
    if (auto model = cls.dyn_cast<ast::Model>()) {
      return declareVariables(*model);
    }

    if (auto package = cls.dyn_cast<ast::Package>()) {
      return declareVariables(*package);
    }

    if (auto function = cls.dyn_cast<ast::PartialDerFunction>()) {
      return declareVariables(*function);
    }

    if (auto record = cls.dyn_cast<ast::Record>()) {
      return declareVariables(*record);
    }

    if (auto standardFunction = cls.dyn_cast<ast::StandardFunction>()) {
      return declareVariables(*standardFunction);
    }

    llvm_unreachable("Unknown class type");
  }

  void ClassLowerer::declare(const ast::Member& variable)
  {
    mlir::Location location = loc(variable.getLocation());

    VariableType variableType = getVariableType(
        *variable.getType(), *variable.getTypePrefix());

    llvm::SmallVector<llvm::StringRef> dimensionsConstraints;

    if (auto arrayType = variableType.unwrap().dyn_cast<ArrayType>()) {
      for (size_t dim = 0, rank = variable.getType()->getRank(); dim < rank;
           ++dim) {
        const ast::ArrayDimension* dimension = (*variable.getType())[dim];

        if (dimension->isDynamic()) {
          if (dimension->hasExpression()) {
            dimensionsConstraints.push_back(
                VariableOp::kDimensionConstraintFixed);
          } else {
            dimensionsConstraints.push_back(
                VariableOp::kDimensionConstraintUnbounded);
          }
        }
      }
    }

    auto variableOp = builder().create<VariableOp>(
        location, variable.getName(), variableType,
        builder().getStrArrayAttr(dimensionsConstraints));

    mlir::SymbolTable& symbolTable = getSymbolTable().getSymbolTable(
        variableOp->getParentOfType<ClassInterface>());

    symbolTable.insert(variableOp);
  }

  void ClassLowerer::lower(const ast::Class& cls)
  {
    if (auto model = cls.dyn_cast<ast::Model>()) {
      return lower(*model);
    }

    if (auto package = cls.dyn_cast<ast::Package>()) {
      return lower(*package);
    }

    if (auto function = cls.dyn_cast<ast::PartialDerFunction>()) {
      return lower(*function);
    }

    if (auto record = cls.dyn_cast<ast::Record>()) {
      return lower(*record);
    }

    if (auto standardFunction = cls.dyn_cast<ast::StandardFunction>()) {
      return lower(*standardFunction);
    }

    llvm_unreachable("Unknown class type");
  }

  VariableType ClassLowerer::getVariableType(
      const ast::VariableType& variableType,
      const ast::TypePrefix& typePrefix)
  {
    llvm::SmallVector<int64_t, 3> shape;

    for (size_t i = 0, rank = variableType.getRank(); i < rank; ++i) {
      const ast::ArrayDimension* dimension = variableType[i];

      if (dimension->isDynamic()) {
        shape.push_back(VariableType::kDynamic);
      } else {
        shape.push_back(dimension->getNumericSize());
      }
    }

    mlir::Type baseType;

    if (auto builtInType = variableType.dyn_cast<ast::BuiltInType>()) {
      if (builtInType->getBuiltInTypeKind() ==
          ast::BuiltInType::Kind::Boolean) {
        baseType = BooleanType::get(builder().getContext());
      } else if (builtInType->getBuiltInTypeKind() ==
                 ast::BuiltInType::Kind::Integer) {
        baseType = IntegerType::get(builder().getContext());
      } else if (builtInType->getBuiltInTypeKind() ==
                 ast::BuiltInType::Kind::Real) {
        baseType = RealType::get(builder().getContext());
      } else {
        llvm_unreachable("Unknown built-in type");
        return nullptr;
      }
    } else if (auto userDefinedType =
                   variableType.dyn_cast<ast::UserDefinedType>()) {
      auto symbolOp = resolveType(*userDefinedType, getLookupScope());

      if (symbolOp == nullptr) {
        llvm_unreachable("Unknown variable type");
        return nullptr;
      }

      if (mlir::isa<RecordOp>(symbolOp)) {
        baseType = RecordType::get(
            builder().getContext(), getSymbolRefFromRoot(symbolOp));
      } else {
        llvm_unreachable("Unknown variable type");
        return nullptr;
      }
    } else {
      llvm_unreachable("Unknown variable type");
      return nullptr;
    }

    VariabilityProperty variabilityProperty = VariabilityProperty::none;
    IOProperty ioProperty = IOProperty::none;

    if (typePrefix.isDiscrete()) {
      variabilityProperty = VariabilityProperty::discrete;
    } else if (typePrefix.isParameter()) {
      variabilityProperty = VariabilityProperty::parameter;
    } else if (typePrefix.isConstant()) {
      variabilityProperty = VariabilityProperty::constant;
    }

    if (typePrefix.isInput()) {
      ioProperty = IOProperty::input;
    } else if (typePrefix.isOutput()) {
      ioProperty = IOProperty::output;
    }

    return VariableType::get(shape, baseType, variabilityProperty, ioProperty);
  }

  void ClassLowerer::lowerClassBody(const ast::Class& cls)
  {
    // Lower the constraints for the dynamic dimensions of the variables.
    for (const auto& variable : cls.getVariables()) {
      lowerVariableDimensionConstraints(
          getSymbolTable().getSymbolTable(getClass(cls)),
          *variable->cast<ast::Member>());
    }

    // Create the equations.
    for (const auto& blockNode : cls.getEquationsBlocks()) {
      const ast::EquationsBlock* block =
          blockNode->cast<ast::EquationsBlock>();

      for (size_t i = 0, e = block->getNumOfEquations(); i < e; ++i) {
        lower(*block->getEquation(i), false);
      }

      for (size_t i = 0, e = block->getNumOfForEquations(); i < e; ++i) {
        lower(*block->getForEquation(i), false);
      }
    }

    // Create the initial equations.
    for (const auto& blockNode : cls.getInitialEquationsBlocks()) {
      const ast::EquationsBlock* block =
          blockNode->cast<ast::EquationsBlock>();

      for (size_t i = 0, e = block->getNumOfEquations(); i < e; ++i) {
        lower(*block->getEquation(i), true);
      }

      for (size_t i = 0, e = block->getNumOfForEquations(); i < e; ++i) {
        lower(*block->getForEquation(i), true);
      }
    }

    // Create the algorithms.
    for (const auto& algorithm : cls.getAlgorithms()) {
      lower(*algorithm->cast<ast::Algorithm>());
    }
  }

  void ClassLowerer::createBindingEquation(
      const ast::Member& variable, const ast::Expression& expression)
  {
    mlir::Location location = loc(expression.getLocation());

    auto bindingEquationOp = builder().create<BindingEquationOp>(
        location, variable.getName());

    mlir::OpBuilder::InsertionGuard guard(builder());

    assert(bindingEquationOp.getBodyRegion().empty());

    mlir::Block* bodyBlock = builder().createBlock(
        &bindingEquationOp.getBodyRegion());

    builder().setInsertionPointToStart(bodyBlock);

    // Lower the expression and yield its value.
    mlir::Location expressionLoc = loc(expression.getLocation());
    auto expressionValues = lower(expression);
    assert(expressionValues.size() == 1);

    builder().create<YieldOp>(
        location, expressionValues[0].get(expressionLoc));
  }

  void ClassLowerer::lowerStartAttribute(
      const ast::Member& member,
      const ast::Expression& expression,
      bool fixed,
      bool each)
  {
    mlir::Location location = loc(expression.getLocation());

    auto startOp = builder().create<StartOp>(
        location, member.getName(), fixed, each);

    mlir::OpBuilder::InsertionGuard guard(builder());

    assert(startOp.getBodyRegion().empty());
    mlir::Block* bodyBlock = builder().createBlock(&startOp.getBodyRegion());
    builder().setInsertionPointToStart(bodyBlock);

    mlir::Location valueLoc = loc(expression.getLocation());
    auto value = lower(expression);
    assert(value.size() == 1);

    builder().create<YieldOp>(location, value[0].get(valueLoc));
  }

  void ClassLowerer::lowerVariableDimensionConstraints(
      mlir::SymbolTable& symbolTable,
      const ast::Member& variable)
  {
    mlir::Location location = loc(variable.getLocation());
    auto variableOp = symbolTable.lookup<VariableOp>(variable.getName());
    assert(variableOp != nullptr);

    if (variableOp.getNumOfFixedDimensions() != 0) {
      mlir::OpBuilder::InsertionGuard guard(builder());

      builder().setInsertionPointToStart(
          &variableOp.getConstraintsRegion().emplaceBlock());

      llvm::SmallVector<mlir::Value> fixedDimensions;

      for (size_t dim = 0, rank = variable.getType()->getRank(); dim < rank;
           ++dim) {
        const ast::ArrayDimension* dimension = (*variable.getType())[dim];

        if (dimension->hasExpression()) {
          const ast::Expression* sizeExpression = dimension->getExpression();
          mlir::Location sizeLoc = loc(sizeExpression->getLocation());
          mlir::Value size = lower(*sizeExpression)[0].get(sizeLoc);

          if (!size.getType().isa<mlir::IndexType>()) {
            size = builder().create<CastOp>(
                location, builder().getIndexType(), size);
          }

          fixedDimensions.push_back(size);
        }
      }

      builder().create<YieldOp>(location, fixedDimensions);
    }
  }
}
