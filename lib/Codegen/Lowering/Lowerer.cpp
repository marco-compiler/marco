#include "marco/Codegen/Lowering/Lowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  Lowerer::Lowerer(LoweringContext* context, BridgeInterface* bridge)
      : context_(context),
        bridge_(bridge)
  {
  }

  mlir::Location Lowerer::loc(const SourcePosition& location)
  {
    return mlir::FileLineColLoc::get(
        builder().getIdentifier(*location.file),
        location.line,
        location.column);
  }

  mlir::Location Lowerer::loc(const SourceRange& location)
  {
    return loc(location.getStartPosition());
  }

  LoweringContext* Lowerer::context()
  {
    return context_;
  }

  mlir::OpBuilder& Lowerer::builder()
  {
    return context_->builder;
  }

  LoweringContext::SymbolTable& Lowerer::symbolTable()
  {
    return context_->symbolTable;
  }

  std::vector<mlir::Operation*> Lowerer::lower(const ast::Class& cls)
  {
    return bridge_->lower(cls);
  }

  Results Lowerer::lower(const ast::Expression& expression)
  {
    return bridge_->lower(expression);
  }

  void Lowerer::lower(const ast::Statement& statement)
  {
    bridge_->lower(statement);
  }

  void Lowerer::lower(const ast::Equation& equation)
  {
    bridge_->lower(equation);
  }

  void Lowerer::lower(const ast::ForEquation& forEquation)
  {
    bridge_->lower(forEquation);
  }

  mlir::Type Lowerer::lower(const ast::Type& type)
  {
    return type.visit([&](const auto& obj) -> mlir::Type {
      auto baseType = lower(obj);

      if (!type.isScalar()) {
        llvm::SmallVector<long, 3> shape;

        for (const auto& dimension : type.getDimensions()) {
          if (dimension.isDynamic()) {
            shape.push_back(-1);
          } else {
            shape.push_back(dimension.getNumericSize());
          }
        }

        return ArrayType::get(builder().getContext(), baseType, shape);
      }

      return baseType;
    });
  }

  mlir::Type Lowerer::lower(const ast::BuiltInType& type)
  {
    switch (type) {
      case BuiltInType::None:
        return builder().getNoneType();

      case BuiltInType::Integer:
        return IntegerType::get(builder().getContext());

      case BuiltInType::Float:
        return RealType::get(builder().getContext());

      case BuiltInType::Boolean:
        return BooleanType::get(builder().getContext());

      default:
        llvm_unreachable("Unknown built-in type");
        return builder().getNoneType();
    }
  }

  mlir::Type Lowerer::lower(const ast::PackedType& type)
  {
    llvm::SmallVector<mlir::Type, 3> types;

    for (const auto& subType : type) {
      types.push_back(lower(subType));
    }

    return builder().getTupleType(types);
  }

  mlir::Type Lowerer::lower(const ast::UserDefinedType& type)
  {
    llvm::SmallVector<mlir::Type, 3> types;

    for (const auto& subType : type) {
      types.push_back(lower(subType));
    }

    return builder().getTupleType(types);
  }
}
