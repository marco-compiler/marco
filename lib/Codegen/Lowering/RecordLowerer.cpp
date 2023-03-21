#include "marco/Codegen/Lowering/RecordLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  RecordLowerer::RecordLowerer(LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge)
  {
  }

  std::vector<mlir::Operation*> RecordLowerer::lower(const Record& record)
  {
    std::vector<mlir::Operation*> result;

    mlir::OpBuilder::InsertionGuard guard(builder());
    Lowerer::SymbolScope varScope(symbolTable());

    mlir::Location location = loc(record.getLocation());

    llvm::SmallVector<mlir::Type, 3> variableTypes;
    llvm::SmallVector<mlir::Location, 3> variableLocations;

    // Determine the type of each variable.
    llvm::SmallVector<mlir::Attribute, 3> variableNames;

    for (const auto& member : record.getMembers()) {
      mlir::Type type = lower(member->getType());

      if (!type.isa<ArrayType>()) {
        type = ArrayType::get(llvm::None, type);
      }

      variableTypes.push_back(type);

      mlir::StringAttr nameAttribute = builder().getStringAttr(member->getName());
      variableNames.push_back(nameAttribute);

      variableLocations.push_back(loc(member->getLocation()));
    }

    // Create the model operation and its blocks.
    auto recordOp = builder().create<RecordOp>(location, record.getName());

    // Simulation variables.
    builder().setInsertionPointToStart(recordOp.bodyBlock());

    // First, add the variables to the symbol table.
    for (const auto& member : record.getMembers()) {
      mlir::Location variableLoc = loc(member->getLocation());
      mlir::Type type = lower(member->getType());

      symbolTable().insert(
          member->getName(),
          Reference::variable(
              builder(), variableLoc, member->getName(), type));
    }

    // Then, lower them.
    for (const auto& member : record.getMembers()) {
      lower(*member);
    }

    // Equations.
    symbolTable().insert("time", Reference::time(builder(), location));

    // Add the model operation to the list of top-level operations.
    result.push_back(recordOp);

    // Process the inner classes.
    builder().setInsertionPointAfter(recordOp);

    for (const auto& innerClass : record.getInnerClasses()) {
      for (auto& loweredInnerClass : lower(*innerClass)) {
        result.push_back(std::move(loweredInnerClass));
      }
    }

    return result;
  }

  void RecordLowerer::lower(const Member& member)
  {
    // TODO refactor, it is the same for both functions and models.
    mlir::Location location = loc(member.getLocation());
    mlir::Type type = lower(member.getType());

    VariabilityProperty variabilityProperty = VariabilityProperty::none;
    IOProperty ioProperty = IOProperty::none;

    if (member.isDiscrete()) {
      variabilityProperty = VariabilityProperty::discrete;
    } else if (member.isParameter()) {
      variabilityProperty = VariabilityProperty::parameter;
    } else if (member.isConstant()) {
      variabilityProperty = VariabilityProperty::constant;
    }

    if (member.isInput()) {
      ioProperty = IOProperty::input;
    } else if (member.isOutput()) {
      ioProperty = IOProperty::output;
    }

    auto variableType = VariableType::wrap(type, variabilityProperty, ioProperty);

    llvm::SmallVector<llvm::StringRef> dimensionsConstraints;
    bool hasFixedDimensions = false;

    if (auto arrayType = type.dyn_cast<ArrayType>()) {
      for (const auto& dimension : member.getType().getDimensions()) {
        if (dimension.hasExpression()) {
          dimensionsConstraints.push_back(
              VariableOp::kDimensionConstraintFixed);

          hasFixedDimensions = true;
        } else {
          dimensionsConstraints.push_back(
              VariableOp::kDimensionConstraintUnbounded);
        }
      }
    }

    auto var = builder().create<VariableOp>(
        location, member.getName(), variableType,
        builder().getStrArrayAttr(dimensionsConstraints),
        nullptr);

    if (hasFixedDimensions) {
      mlir::OpBuilder::InsertionGuard guard(builder());

      builder().setInsertionPointToStart(
          &var.getConstraintsRegion().emplaceBlock());

      llvm::SmallVector<mlir::Value> fixedDimensions;

      for (const auto& dimension : member.getType().getDimensions()) {
        if (dimension.hasExpression()) {
          mlir::Location dimensionLoc =
              loc(dimension.getExpression()->getLocation());

          mlir::Value size =
              lower(*dimension.getExpression())[0].get(dimensionLoc);

          size = builder().create<CastOp>(
              location, builder().getIndexType(), size);

          fixedDimensions.push_back(size);
        }
      }

      builder().create<YieldOp>(location, fixedDimensions);
    }
  }
}
