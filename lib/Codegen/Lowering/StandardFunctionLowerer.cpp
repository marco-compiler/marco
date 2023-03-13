#include "marco/Codegen/Lowering/StandardFunctionLowerer.h"
#include "marco/AST/Analysis/DynamicDimensionsGraph.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  StandardFunctionLowerer::StandardFunctionLowerer(
      LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge)
  {
  }

  std::vector<mlir::Operation*> StandardFunctionLowerer::lower(
      const StandardFunction& function)
  {
    std::vector<mlir::Operation*> result;

    mlir::OpBuilder::InsertionGuard guard(builder());
    Lowerer::SymbolScope scope(symbolTable());

    mlir::Location location = loc(function.getLocation());

    // Input variables.
    llvm::SmallVector<llvm::StringRef, 3> argNames;

    for (const auto& member : function.getArgs()) {
      argNames.emplace_back(member->getName());
    }

    // Output variables.
    llvm::SmallVector<llvm::StringRef, 1> returnNames;
    auto outputMembers = function.getResults();

    for (const auto& member : outputMembers) {
      returnNames.emplace_back(member->getName());
    }

    // Create the function.
    auto functionOp = builder().create<FunctionOp>(location, function.getName());

    // Process the annotations.
    if (function.hasAnnotation()) {
      const auto* annotation = function.getAnnotation();

      // Inline attribute.
      functionOp->setAttr(
          "inline",
          builder().getBoolAttr(
              function.getAnnotation()->getInlineProperty()));

      // Inverse functions attribute.
      auto inverseFunctionAnnotation =
          annotation->getInverseFunctionAnnotation();

      InverseFunctionsMap map;

      // Create a map of the function members indexes for faster retrieval.
      llvm::StringMap<unsigned int> indexes;

      for (const auto& name : llvm::enumerate(argNames)) {
        indexes[name.value()] = name.index();
      }

      for (const auto& name : llvm::enumerate(returnNames)) {
        indexes[name.value()] = argNames.size() + name.index();
      }

      mlir::StorageUniquer::StorageAllocator allocator;

      // Iterate over the input arguments and for each invertible one
      // add the function to the inverse map.
      for (const auto& arg : argNames) {
        if (!inverseFunctionAnnotation.isInvertible(arg)) {
          continue;
        }

        auto inverseArgs = inverseFunctionAnnotation.getInverseArgs(arg);
        llvm::SmallVector<unsigned int, 3> permutation;

        for (const auto& inverseArg : inverseArgs) {
          assert(indexes.find(inverseArg) != indexes.end());
          permutation.push_back(indexes[inverseArg]);
        }

        map[indexes[arg]] = std::make_pair(
            inverseFunctionAnnotation.getInverseFunction(arg),
            allocator.copyInto(llvm::ArrayRef<unsigned int>(permutation)));
      }

      if (!map.empty()) {
        auto inverseFunctionAttribute =
            InverseFunctionsAttr::get(builder().getContext(), map);

        functionOp->setAttr("inverse", inverseFunctionAttribute);
      }

      if (annotation->hasDerivativeAnnotation()) {
        auto derivativeAnnotation = annotation->getDerivativeAnnotation();

        auto derivativeAttribute = DerivativeAttr::get(
            builder().getContext(),
            derivativeAnnotation.getName(),
            derivativeAnnotation.getOrder());

        functionOp->setAttr("derivative", derivativeAttribute);
      }
    }

    // Create the body of the function.
    mlir::Block* entryBlock = builder().createBlock(&functionOp.getBody());
    builder().setInsertionPointToStart(entryBlock);

    // Add the variables to the symbol table.
    for (const auto& member : function.getMembers()) {
      mlir::Location variableLoc = loc(member->getLocation());
      mlir::Type type = lower(member->getType());

      symbolTable().insert(
          member->getName(),
          Reference::variable(
              builder(), variableLoc, member->getName(), type));
    }

    // Then, lower them.
    for (const auto& member : function.getMembers()) {
      lower(*member);

      if (member->hasExpression()) {
        lowerVariableDefaultValue(
            member->getName(), *member->getExpression());
      }
    }

    // Emit the body of the function.
    if (auto algorithms = function.getAlgorithms(); algorithms.size() > 0) {
      assert(algorithms.size() == 1);

      const auto& algorithm = function.getAlgorithms()[0];

      if (!algorithm->empty()) {
        auto algorithmOp = builder().create<AlgorithmOp>(
            loc(algorithm->getLocation()));

        algorithmOp.getBodyRegion().emplaceBlock();
        builder().setInsertionPointToStart(algorithmOp.bodyBlock());

        for (const auto& statement : *algorithm) {
          lower(*statement);
        }
      }
    }

    result.push_back(functionOp);
    return result;
  }

  void StandardFunctionLowerer::lower(const Member& member)
  {
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

    auto memberType = MemberType::wrap(type, variabilityProperty, ioProperty);

    llvm::SmallVector<llvm::StringRef> dimensionsConstraints;
    bool hasFixedDimensions = false;

    if (auto arrayType = type.dyn_cast<ArrayType>()) {
      for (const auto& dimension : member.getType().getDimensions()) {
        if (dimension.isDynamic()) {
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
    }

    auto var = builder().create<VariableOp>(
        location, member.getName(), memberType,
        builder().getStrArrayAttr(dimensionsConstraints));

    if (hasFixedDimensions) {
      mlir::OpBuilder::InsertionGuard guard(builder());

      builder().setInsertionPointToStart(
          &var.getConstraintsRegion().emplaceBlock());

      llvm::SmallVector<mlir::Value> fixedDimensions;

      for (const auto& dimension : member.getType().getDimensions()) {
        if (dimension.hasExpression()) {
          mlir::Value size = lower(*dimension.getExpression())[0].get(location);

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

  void StandardFunctionLowerer::lowerVariableDefaultValue(
   llvm::StringRef variable, const ast::Expression& expression)
  {
    mlir::Location expressionLoc = loc(expression.getLocation());
    auto defaultOp = builder().create<DefaultOp>(expressionLoc, variable);

    mlir::OpBuilder::InsertionGuard guard(builder());
    mlir::Block* bodyBlock = builder().createBlock(&defaultOp.getBodyRegion());
    builder().setInsertionPointToStart(bodyBlock);

    mlir::Value value = lower(expression)[0].get(expressionLoc);
    builder().create<YieldOp>(expressionLoc, value);
  }
}
