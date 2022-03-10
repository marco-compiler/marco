#include "llvm/ADT/SmallVector.h"
#include "marco/AST/AST.h"
#include "marco/Codegen/NewBridge.h"
#include "marco/Codegen/Lowering/FunctionCallBridge.h"
#include "marco/Codegen/Lowering/OperationBridge.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/Verifier.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  NewLoweringBridge::NewLoweringBridge(mlir::MLIRContext& context, CodegenOptions options)
      : builder(&context),
        options(options)
  {
    context.loadDialect<ModelicaDialect>();
    context.loadDialect<mlir::StandardOpsDialect>();
  }

  mlir::Location NewLoweringBridge::loc(SourcePosition location)
  {
    return mlir::FileLineColLoc::get(
        builder.getIdentifier(*location.file),
        location.line,
        location.column);
  }

  mlir::Location NewLoweringBridge::loc(SourceRange location)
  {
    return loc(location.getStartPosition());
  }

  llvm::Optional<mlir::ModuleOp> NewLoweringBridge::run(llvm::ArrayRef<std::unique_ptr<Class>> classes)
  {
    /*
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    llvm::SmallVector<mlir::Operation*, 3> operations;

    for (const auto& cls : classes) {
      auto* op = cls->visit([&](const auto& obj) {
        return lower(obj);
      });

      if (op != nullptr)
        operations.push_back(op);
    }

    if (operations.size() == 1 && mlir::isa<mlir::ModuleOp>(operations[0]))
      module = mlir::cast<mlir::ModuleOp>(operations[0]);
    else
    {
      module = mlir::ModuleOp::create(builder.getUnknownLoc());

      for (const auto& op : operations)
        module.push_back(op);
    }

    if (failed(mlir::verify(module)))
      return llvm::None;

    return module;
     */
  }

  mlir::Operation* NewLoweringBridge::lower(const ast::Class& cls)
  {
    return cls.visit([&](const auto& obj) {
      return lower(obj);
    });
  }

  mlir::Operation* NewLoweringBridge::lower(const ast::PartialDerFunction& function)
  {
    /*
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto location = loc(function.getLocation());

    llvm::StringRef derivedFunction = function.getDerivedFunction()->get<ReferenceAccess>()->getName();
    llvm::SmallVector<llvm::StringRef, 3> independentVariables;

    for (const auto& independentVariable : function.getIndependentVariables())
      independentVariables.push_back(independentVariable->get<ReferenceAccess>()->getName());

    return builder.create<DerFunctionOp>(location, function.getName(), derivedFunction, independentVariables);
     */
  }

  mlir::Operation* NewLoweringBridge::lower(const ast::StandardFunction& function)
  {
    /*
    mlir::OpBuilder::InsertionGuard guard(builder);
    llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);

    auto location = loc(function.getLocation());

    llvm::SmallVector<llvm::StringRef, 3> argNames;
    llvm::SmallVector<mlir::Type, 3> argTypes;

    for (const auto& member : function.getArgs())
    {
      argNames.emplace_back(member->getName());

      mlir::Type type = lower(member->getType(), BufferAllocationScope::unknown);

      if (auto arrayType = type.dyn_cast<ArrayType>())
        type = arrayType.toUnknownAllocationScope();

      argTypes.emplace_back(type);
    }

    llvm::SmallVector<llvm::StringRef, 3> returnNames;
    llvm::SmallVector<mlir::Type, 3> returnTypes;
    auto outputMembers = function.getResults();

    for (const auto& member : outputMembers)
    {
      const auto& frontendType = member->getType();
      mlir::Type type = lower(member->getType(), BufferAllocationScope::heap);

      if (auto arrayType = type.dyn_cast<ArrayType>())
        type = arrayType.toAllocationScope(BufferAllocationScope::heap);

      returnNames.emplace_back(member->getName());
      returnTypes.emplace_back(type);
    }

    auto functionType = builder.getFunctionType(argTypes, returnTypes);
    auto functionOp = builder.create<FunctionOp>(location, function.getName(), functionType, argNames, returnNames);

    if (function.hasAnnotation())
    {
      const auto* annotation = function.getAnnotation();

      // Inline attribute
      functionOp->setAttr("inline", builder.getBoolAttr(function.getAnnotation()->getInlineProperty()));

      {
        // Inverse functions attribute
        auto inverseFunctionAnnotation = annotation->getInverseFunctionAnnotation();
        InverseFunctionsAttribute::Map map;

        // Create a map of the function members indexes for faster retrieval
        llvm::StringMap<unsigned int> indexes;

        for (const auto& name : llvm::enumerate(argNames))
          indexes[name.value()] = name.index();

        for (const auto& name : llvm::enumerate(returnNames))
          indexes[name.value()] = argNames.size() + name.index();

        mlir::StorageUniquer::StorageAllocator allocator;

        // Iterate over the input arguments and for each invertible one
        // add the function to the inverse map.
        for (const auto& arg : argNames)
        {
          if (!inverseFunctionAnnotation.isInvertible(arg))
            continue;

          auto inverseArgs = inverseFunctionAnnotation.getInverseArgs(arg);
          llvm::SmallVector<unsigned int, 3> permutation;

          for (const auto& inverseArg : inverseArgs)
          {
            assert(indexes.find(inverseArg) != indexes.end());
            permutation.push_back(indexes[inverseArg]);
          }

          map[indexes[arg]] = std::make_pair(
              inverseFunctionAnnotation.getInverseFunction(arg),
              allocator.copyInto(llvm::ArrayRef<unsigned int>(permutation)));
        }

        if (!map.empty())
        {
          auto inverseFunctionAttribute = builder.getInverseFunctionsAttribute(map);
          functionOp->setAttr("inverse", inverseFunctionAttribute);
        }
      }

      if (annotation->hasDerivativeAnnotation())
      {
        auto derivativeAnnotation = annotation->getDerivativeAnnotation();
        auto derivativeAttribute = builder.getDerivativeAttribute(derivativeAnnotation.getName(), derivativeAnnotation.getOrder());
        functionOp->setAttr("derivative", derivativeAttribute);
      }
    }

    // If the function doesn't have a body, it means it is just a declaration
    if (function.getAlgorithms().empty())
      return { functionOp };

    // Start the body of the function.
    auto& entryBlock = *functionOp.addEntryBlock();

    // Declare all the function arguments in the symbol table
    for (const auto& [name, value] : llvm::zip(argNames, entryBlock.getArguments()))
      symbolTable.insert(name, Reference::ssa(&builder, value));

    builder.setInsertionPointToStart(&entryBlock);

    // Initialize members
    for (const auto& member : function.getMembers())
      lower<Function>(*member);

    // Emit the body of the function
    const auto& algorithm = function.getAlgorithms()[0];

    // Lower the statements
    lower(*function.getAlgorithms()[0]);

    builder.create<FunctionTerminatorOp>(location);
    return functionOp;
     */
  }

  mlir::Operation* NewLoweringBridge::lower(const ast::Model& model)
  {
    /*
    mlir::OpBuilder::InsertionGuard guard(builder);
    llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);

    auto location = loc(model.getLocation());

    llvm::SmallVector<mlir::Type, 3> args;

    // Time variable
    args.push_back(builder.getArrayType(BufferAllocationScope::unknown, builder.getRealType()));

    // Other variables
    llvm::SmallVector<mlir::Attribute, 3> variableNames;

    for (const auto& member : model.getMembers())
    {
      mlir::Type type = lower(member->getType(), BufferAllocationScope::unknown);

      if (auto arrayType = type.dyn_cast<ArrayType>())
        type = arrayType.toUnknownAllocationScope();
      else
        type = builder.getArrayType(BufferAllocationScope::unknown, type);

      args.push_back(type);

      mlir::StringAttr nameAttribute = builder.getStringAttr(member->getName());
      variableNames.push_back(nameAttribute);
    }

    llvm::ArrayRef<mlir::Attribute> attributeArray(variableNames);
    mlir::ArrayAttr variableNamesAttribute = builder.getArrayAttr(attributeArray);

    // Create the operation
    auto modelOp = builder.create<ModelOp>(
        location,
        variableNamesAttribute,
        builder.getRealAttribute(options.startTime),
        builder.getRealAttribute(options.endTime),
        builder.getRealAttribute(options.timeStep),
        // TODO
        //builder.getRealAttribute(options.relativeTolerance),
        //builder.getRealAttribute(options.absoluteTolerance),
        args);

    {
      // Simulation variables
      builder.setInsertionPointToStart(&modelOp.init().front());
      llvm::SmallVector<mlir::Value, 3> vars;

      auto memberType = MemberType::get(builder.getContext(), MemberAllocationScope::heap, builder.getRealType());
      mlir::Value time = builder.create<MemberCreateOp>(location, "time", memberType, llvm::None, false);
      vars.push_back(time);

      for (const auto& member : model.getMembers())
      {
        lower<ast::Model>(*member);
        vars.push_back(symbolTable.lookup(member->getName()).getReference());
      }

      builder.create<YieldOp>(location, vars);
    }

    {
      // Body
      builder.setInsertionPointToStart(&modelOp.body().front());

      mlir::Value time = modelOp.time();
      symbolTable.insert("time", Reference::memory(&builder, time));

      for (const auto& member : llvm::enumerate(model.getMembers()))
        symbolTable.insert(member.value()->getName(), Reference::memory(&builder, modelOp.body().getArgument(member.index() + 1)));

      for (const auto& equation : model.getEquations())
        lower(*equation);

      for (const auto& forEquation : model.getForEquations())
        lower(*forEquation);

      builder.create<YieldOp>(location);
    }

    return modelOp;
     */
  }

  mlir::Operation* NewLoweringBridge::lower(const ast::Package& package)
  {
    /*
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (const auto& innerClass : package.getInnerClasses())
    {
      auto* op = innerClass->visit([&](const auto& obj) {
        return lower(obj);
      });

      if (op != nullptr)
        module.push_back(op);
    }

    return module;
     */
  }

  mlir::Operation* NewLoweringBridge::lower(const ast::Record& record)
  {
    /*
    /*
    llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
    auto location = loc(record.getLocation());

    // Whenever a record is defined, a record constructor function with the
    // same name and in the same scope as the record class must be implicitly
    // defined, so that the record can then be instantiated.

    llvm::SmallVector<mlir::Type, 3> argsTypes;
    llvm::SmallVector<mlir::Type, 3> recordTypes;

    for (const auto& member : record)
    {
      argsTypes.push_back(lower(member.getType(), BufferAllocationScope::unknown));
      recordTypes.push_back(lower(member.getType(), BufferAllocationScope::heap));
    }

    RecordType resultType = builder.getRecordType(recordTypes);

    auto functionType = builder.getFunctionType(argsTypes, resultType);
    auto function = mlir::FuncOp::create(location, record.getName(), functionType);

    auto& entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    llvm::SmallVector<mlir::Value, 3> results;

    for (const auto& [arg, type] : llvm::zip(entryBlock.getArguments(), recordTypes))
    {
      if (auto arrayType = type.dyn_cast<ArrayType>())
        results.push_back(builder.create<ArrayCloneOp>(location, arg, arrayType, false));
      else
        results.push_back(arg);
    }

    mlir::Value result = builder.create<RecordOp>(location, resultType, results);
    builder.create<mlir::ReturnOp>(location, result);

    return { function };
     */

    return nullptr;
  }

  mlir::Type NewLoweringBridge::lower(const Type& type, ArrayAllocationScope desiredAllocationScope)
  {
    /*
    auto visitor = [&](const auto& obj) -> mlir::Type
    {
      auto baseType = lower(obj, desiredAllocationScope);

      if (!type.isScalar())
      {
        const auto& dimensions = type.getDimensions();
        llvm::SmallVector<long, 3> shape;

        for (const auto& dimension : type.getDimensions())
        {
          if (dimension.isDynamic())
            shape.emplace_back(-1);
          else
            shape.emplace_back(dimension.getNumericSize());
        }

        return builder.getArrayType(desiredAllocationScope, baseType, shape).toMinAllowedAllocationScope();
      }

      return baseType;
    };

    return type.visit(visitor);
     */
  }

  mlir::Type NewLoweringBridge::lower(const BuiltInType& type, ArrayAllocationScope desiredAllocationScope)
  {
    /*
    switch (type)
    {
      case BuiltInType::None:
        return builder.getNoneType();
      case BuiltInType::Integer:
        return builder.getIntegerType();
      case BuiltInType::Float:
        return builder.getRealType();
      case BuiltInType::Boolean:
        return builder.getBooleanType();
      default:
        assert(false && "Unexpected type");
        return builder.getNoneType();
    }
     */
  }

  mlir::Type NewLoweringBridge::lower(const PackedType& type, ArrayAllocationScope desiredAllocationScope)
  {
    /*
    llvm::SmallVector<mlir::Type, 3> types;

    for (const auto& subType : type)
      types.push_back(lower(subType, desiredAllocationScope));

    return builder.getTupleType(move(types));
     */
  }

  mlir::Type NewLoweringBridge::lower(const UserDefinedType& type, ArrayAllocationScope desiredAllocationScope)
  {
    /*
    llvm::SmallVector<mlir::Type, 3> types;

    for (const auto& subType : type)
      types.push_back(lower(subType, desiredAllocationScope));

    return builder.getTupleType(move(types));
     */
  }

  template<>
  void NewLoweringBridge::lower<ast::Model>(const Member& member)
  {
    /*
    auto location = loc(member.getLocation());

    const auto& frontendType = member.getType();
    mlir::Type type = lower(frontendType, BufferAllocationScope::heap);

    auto memberType = type.isa<ArrayType>() ?
                      MemberType::get(type.cast<ArrayType>()) :
                      MemberType::get(builder.getContext(), MemberAllocationScope::heap, type);

    bool isConstant = member.isParameter();
    mlir::Value memberOp = builder.create<MemberCreateOp>(location, member.getName(), memberType, llvm::None, isConstant);
    symbolTable.insert(member.getName(), Reference::member(&builder, memberOp));

    Reference ref = symbolTable.lookup(member.getName());

    if (member.hasStartOverload())
    {
      auto values = lower<Expression>(*member.getStartOverload());
      assert(values.size() == 1);

      if (auto arrayType = type.dyn_cast<ArrayType>())
        builder.create<FillOp>(location, *values[0], *ref);
      else
        ref.set(*values[0]);
    }
    else if (member.hasInitializer())
    {
      mlir::Value value = *lower<Expression>(*member.getInitializer())[0];
      ref.set(value);
    }
    else
    {
      if (auto arrayType = type.dyn_cast<ArrayType>())
      {
        mlir::Value zero = builder.create<ConstantOp>(location, builder.getZeroAttribute(arrayType.getElementType()));
        builder.create<FillOp>(location, zero, *ref);
      }
      else
      {
        mlir::Value zero = builder.create<ConstantOp>(location, builder.getZeroAttribute(type));
        ref.set(zero);
      }
    }
     */
  }

  /// Lower a member of a function.
  ///
  /// Input members are ignored because they are supposed to be unmodifiable
  /// as per the Modelica standard, and thus don't need a local copy.
  /// Output arrays are always allocated on the heap and eventually moved to
  /// input arguments by the dedicated pass. Protected arrays, instead, are
  /// allocated according to the ArrayType allocation logic.
  template<>
  void NewLoweringBridge::lower<Function>(const Member& member)
  {
    /*
    auto location = loc(member.getLocation());

    // Input values are supposed to be read-only by the Modelica standard,
    // thus they don't need to be copied for local modifications.

    if (member.isInput())
      return;

    const auto& frontendType = member.getType();
    mlir::Type type = lower(frontendType, member.isOutput() ? BufferAllocationScope::heap : BufferAllocationScope::stack);

    llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
    MemberType::Shape shape;

    if (auto arrayType = type.dyn_cast<ArrayType>())
    {
      for (auto dimension : arrayType.getShape())
        shape.push_back(dimension);

      auto expressionsCount = llvm::count_if(
          member.getType().getDimensions(),
          [](const auto& dimension) {
            return dimension.hasExpression();
          });

      // If all the dynamic dimensions have an expression to determine their
      // values, then the member can be instantiated from the beginning.

      bool initialized = expressionsCount == arrayType.getDynamicDimensions();

      if (initialized)
      {
        for (const auto& dimension : member.getType().getDimensions())
        {
          if (dimension.hasExpression())
          {
            mlir::Value size = *lower<Expression>(*dimension.getExpression())[0];
            size = builder.create<CastOp>(location, size, builder.getIndexType());
            dynamicDimensions.push_back(size);
          }
        }
      }
    }

    auto memberType = type.isa<ArrayType>() ?
                      MemberType::get(type.cast<ArrayType>()) :
                      MemberType::get(builder.getContext(), MemberAllocationScope::stack, type);

    mlir::Value var = builder.create<MemberCreateOp>(location, member.getName(), memberType, dynamicDimensions);
    symbolTable.insert(member.getName(), Reference::member(&builder, var));

    if (member.hasInitializer())
    {
      // If the member has an initializer expression, lower and assign it as
      // if it was a regular assignment statement.

      Reference memory = symbolTable.lookup(member.getName());
      mlir::Value value = *lower<Expression>(*member.getInitializer())[0];
      memory.set(value);
    }
     */
  }

  void NewLoweringBridge::lower(const Equation& equation)
  {
    mlir::Location location = loc(equation.getLocation());
    auto equationOp = builder.create<EquationOp>(location);
    mlir::OpBuilder::InsertionGuard guard(builder);
    assert(equationOp.bodyRegion().empty());
    mlir::Block* equationBodyBlock = builder.createBlock(&equationOp.bodyRegion());
    builder.setInsertionPointToStart(equationBodyBlock);

    llvm::SmallVector<mlir::Value, 1> lhs;
    llvm::SmallVector<mlir::Value, 1> rhs;

    {
      // Left-hand side
      const auto* expression = equation.getLhsExpression();
      auto references = lower<Expression>(*expression);

      for (auto& reference : references) {
        lhs.push_back(*reference);
      }
    }

    {
      // Right-hand side
      const auto* expression = equation.getRhsExpression();
      auto references = lower<Expression>(*expression);

      for (auto& reference : references) {
        rhs.push_back(*reference);
      }
    }

    mlir::Value lhsTuple = builder.create<EquationSideOp>(location, lhs);
    mlir::Value rhsTuple = builder.create<EquationSideOp>(location, rhs);
    builder.create<EquationSidesOp>(location, lhsTuple, rhsTuple);
  }

  void NewLoweringBridge::lower(const ForEquation& forEquation)
  {
    /*
    llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
    mlir::Location location = loc(forEquation.getEquation()->getLocation());

    // We need to keep track of the first loop in order to restore
    // the insertion point right after that when we have finished
    // lowering all the nested inductions.
    mlir::Operation* firstOp = nullptr;

    llvm::SmallVector<mlir::Value, 3> inductions;

    for (auto& induction : forEquation.getInductions())
    {
      const auto& startExpression = induction->getBegin();
      assert(startExpression->isa<Constant>());
      long start = startExpression->get<Constant>()->as<BuiltInType::Integer>();

      const auto& endExpression = induction->getEnd();
      assert(endExpression->isa<Constant>());
      long end = endExpression->get<Constant>()->as<BuiltInType::Integer>();

      auto forEquationOp = builder.create<ForEquationOp>(location, start, end);
      builder.setInsertionPointToStart(forEquationOp.body());

      symbolTable.insert(
          induction->getName(),
          Reference::ssa(&builder, forEquationOp.induction()));

      if (firstOp == nullptr)
        firstOp = forEquationOp.getOperation();
    }

    const auto& equation = forEquation.getEquation();

    auto equationOp = builder.create<EquationOp>(location);
    builder.setInsertionPointToStart(equationOp.body());

    llvm::SmallVector<mlir::Value, 1> lhs;
    llvm::SmallVector<mlir::Value, 1> rhs;

    {
      // Left-hand side
      const auto* expression = equation->getLhsExpression();
      auto references = lower<Expression>(*expression);

      for (auto& reference : references)
        lhs.push_back(*reference);
    }

    {
      // Right-hand side
      const auto* expression = equation->getRhsExpression();
      auto references = lower<Expression>(*expression);

      for (auto& reference : references)
        rhs.push_back(*reference);
    }

    builder.create<EquationSidesOp>(location, lhs, rhs);
    builder.setInsertionPointAfter(firstOp);
     */
  }

  void NewLoweringBridge::lower(const Algorithm& algorithm)
  {
    // Lower each statement composing the algorithm
    for (const auto& statement : algorithm) {
      lower(*statement);
    }
  }

  void NewLoweringBridge::lower(const Statement& statement)
  {
    statement.visit([&](const auto& obj) {
      lower(obj);
    });
  }

  void NewLoweringBridge::lower(const AssignmentStatement& statement)
  {
    const auto* destinations = statement.getDestinations();
    auto values = lower<Expression>(*statement.getExpression());

    assert(destinations->isa<Tuple>());
    const auto* destinationsTuple = destinations->get<Tuple>();
    assert(values.size() == destinationsTuple->size() && "Unequal number of destinations and results");

    for (const auto& [ dest, value ] : llvm::zip(*destinationsTuple, values)) {
      auto destination = lower<Expression>(*dest)[0];
      destination.set(*value);
    }
  }

  void NewLoweringBridge::lower(const IfStatement& statement)
  {
    // Each conditional blocks creates an "if" operation, but we need to keep
    // track of the first one in order to restore the insertion point right
    // after that when we have finished lowering all the blocks.
    mlir::Operation* firstOp = nullptr;

    for (size_t i = 0, e = statement.size(); i < e; ++i) {
      llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
      const auto& conditionalBlock = statement[i];
      auto condition = lower<Expression>(*conditionalBlock.getCondition())[0];

      // The last conditional block can be at most an originally equivalent
      // "else" block, and thus doesn't need a lowered "else" block.
      bool elseBlock = i < e - 1;

      auto location = loc(statement.getLocation());
      auto ifOp = builder.create<IfOp>(location, *condition, elseBlock);

      if (firstOp == nullptr) {
        firstOp = ifOp;
      }

      // Populate the "then" block
      builder.setInsertionPointToStart(ifOp.thenBlock());

      for (const auto& bodyStatement : conditionalBlock) {
        lower(*bodyStatement);
      }

      if (i > 0) {
        builder.setInsertionPointAfter(ifOp);
      }

      // The next conditional blocks will be placed as new If operations
      // nested inside the "else" block.
      if (elseBlock) {
        builder.setInsertionPointToStart(ifOp.elseBlock());
      }
    }

    builder.setInsertionPointAfter(firstOp);
  }

  void NewLoweringBridge::lower(const ForStatement& statement)
  {
    /*
    llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
    auto location = loc(statement.getLocation());

    const auto& induction = statement.getInduction();

    mlir::Value lowerBound = *lower<Expression>(*induction->getBegin())[0];
    lowerBound = builder.create<CastOp>(lowerBound.getLoc(), lowerBound, builder.getIndexType());

    auto forOp = builder.create<ForOp>(location, lowerBound);
    mlir::OpBuilder::InsertionGuard guard(builder);

    {
      // Check the loop condition
      llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
      symbolTable.insert(induction->getName(), Reference::ssa(&builder, forOp.condition().front().getArgument(0)));

      builder.setInsertionPointToStart(&forOp.condition().front());

      mlir::Value upperBound = *lower<Expression>(*induction->getEnd())[0];
      upperBound = builder.create<CastOp>(lowerBound.getLoc(), upperBound, builder.getIndexType());

      mlir::Value condition = builder.create<LteOp>(location, builder.getBooleanType(), forOp.condition().front().getArgument(0), upperBound);
      builder.create<ConditionOp>(location, condition, *symbolTable.lookup(induction->getName()));
    }

    {
      // Body
      llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
      symbolTable.insert(induction->getName(), Reference::ssa(&builder, forOp.body().front().getArgument(0)));

      builder.setInsertionPointToStart(&forOp.body().front());

      for (const auto& stmnt : statement)
        lower(*stmnt);

      if (auto& body = forOp.body().back(); body.empty() || !body.back().hasTrait<mlir::OpTrait::IsTerminator>())
      {
        builder.setInsertionPointToEnd(&body);
        builder.create<YieldOp>(location, *symbolTable.lookup(induction->getName()));
      }
    }

    {
      // Step
      llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
      symbolTable.insert(induction->getName(), Reference::ssa(&builder, forOp.step().front().getArgument(0)));

      builder.setInsertionPointToStart(&forOp.step().front());

      mlir::Value step = builder.create<ConstantOp>(location, builder.getIndexAttribute(1));
      mlir::Value incremented = builder.create<AddOp>(location, builder.getIndexType(), *symbolTable.lookup(induction->getName()), step);
      builder.create<YieldOp>(location, incremented);
    }
     */
  }

  void NewLoweringBridge::lower(const WhileStatement& statement)
  {
    auto location = loc(statement.getLocation());

    // Create the operation
    auto whileOp = builder.create<WhileOp>(location);
    mlir::OpBuilder::InsertionGuard guard(builder);

    {
      // Condition
      llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
      assert(whileOp.conditionRegion().empty());
      mlir::Block* conditionBlock = builder.createBlock(&whileOp.conditionRegion());
      builder.setInsertionPointToStart(conditionBlock);
      const auto* condition = statement.getCondition();

      builder.create<ConditionOp>(
          loc(condition->getLocation()),
          *lower<Expression>(*condition)[0]);
    }

    {
      // Body
      llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
      assert(whileOp.bodyRegion().empty());
      mlir::Block* bodyBlock = builder.createBlock(&whileOp.bodyRegion());
      builder.setInsertionPointToStart(bodyBlock);

      for (const auto& stmnt : statement) {
        lower(*stmnt);
      }

      if (auto& body = whileOp.bodyRegion().back(); body.empty() || !body.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder.setInsertionPointToEnd(&body);
        builder.create<YieldOp>(location, llvm::None);
      }
    }
  }

  void NewLoweringBridge::lower(const WhenStatement& statement)
  {
    llvm_unreachable("When statement is not implemented yet");
  }

  void NewLoweringBridge::lower(const BreakStatement& statement)
  {
    auto location = loc(statement.getLocation());
    builder.create<BreakOp>(location);
  }

  void NewLoweringBridge::lower(const ReturnStatement& statement)
  {
    auto location = loc(statement.getLocation());
    builder.create<ReturnOp>(location);
  }

  template<>
  Results NewLoweringBridge::lower<Expression>(const Expression& expression)
  {
    return expression.visit([&](const auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return lower<deconst>(expression);
    });
  }

  template<>
  Results NewLoweringBridge::lower<Operation>(const Expression& expression)
  {
    assert(expression.isa<Operation>());
    const auto* operation = expression.get<Operation>();
    OperationBridge operationBridge(this);

    auto lowererFn = [](OperationKind kind) -> OperationBridge::Lowerer {
      switch (kind) {
        case OperationKind::negate:
          return &OperationBridge::negate;

        case OperationKind::add:
          return &OperationBridge::add;

        case OperationKind::subtract:
          return &OperationBridge::subtract;

        case OperationKind::multiply:
          return &OperationBridge::multiply;

        case OperationKind::divide:
          return &OperationBridge::divide;

        case OperationKind::ifelse:
          return &OperationBridge::ifElse;

        case OperationKind::greater:
          return &OperationBridge::greater;

        case OperationKind::greaterEqual:
          return &OperationBridge::greaterOrEqual;

        case OperationKind::equal:
          return &OperationBridge::equal;

        case OperationKind::different:
          return &OperationBridge::notEqual;

        case OperationKind::lessEqual:
          return &OperationBridge::lessOrEqual;

        case OperationKind::less:
          return &OperationBridge::less;

        case OperationKind::land:
          return &OperationBridge::logicalAnd;

        case OperationKind::lor:
          return &OperationBridge::logicalOr;

        case OperationKind::subscription:
          return &OperationBridge::subscription;

        case OperationKind::memberLookup:
          return &OperationBridge::memberLookup;

        case OperationKind::powerOf:
          return &OperationBridge::powerOf;
      }

      llvm_unreachable("Unknown operation type");
      return nullptr;
    };

    auto lowerer = lowererFn(operation->getOperationKind());
    return lowerer(operationBridge, *operation);
  }

  template<>
  Results NewLoweringBridge::lower<Constant>(const Expression& expression)
  {
    assert(expression.isa<Constant>());
    const auto* constant = expression.get<Constant>();
    const auto& type = constant->getType();

    assert(type.isa<BuiltInType>() && "Constants can be made only of built-in typed values");
    auto builtInType = type.get<BuiltInType>();

    mlir::Attribute attribute;

    if (builtInType == BuiltInType::Boolean) {
      attribute = builder.getBooleanAttribute(constant->as<BuiltInType::Boolean>());
    } else if (builtInType == BuiltInType::Integer) {
      attribute = builder.getIntegerAttribute(constant->as<BuiltInType::Integer>());
    } else if (builtInType == BuiltInType::Float) {
      attribute = builder.getRealAttribute(constant->as<BuiltInType::Float>());
    } else {
      llvm_unreachable("Unsupported constant type");
    }

    auto result = builder.create<ConstantOp>(loc(expression.getLocation()), attribute);
    return Reference::ssa(&builder, result);
  }

  template<>
  Results NewLoweringBridge::lower<ReferenceAccess>(const Expression& expression)
  {
    assert(expression.isa<ReferenceAccess>());
    const auto& reference = expression.get<ReferenceAccess>();
    return symbolTable.lookup(reference->getName());
  }

  template<>
  Results NewLoweringBridge::lower<Call>(const Expression& expression)
  {
    assert(expression.isa<Call>());
    const auto* call = expression.get<Call>();
    const auto* function = call->getFunction();
    const auto& functionName = function->get<ReferenceAccess>()->getName();

    auto lowerer = llvm::StringSwitch<FunctionCallBridge::Lowerer>(functionName)
        .Case("abs", &FunctionCallBridge::abs)
        .Case("acos", &FunctionCallBridge::acos)
        .Case("asin", &FunctionCallBridge::asin)
        .Case("atan", &FunctionCallBridge::atan)
        .Case("atan2", &FunctionCallBridge::atan2)
        .Case("cos", &FunctionCallBridge::cos)
        .Case("cosh", &FunctionCallBridge::cosh)
        .Case("der", &FunctionCallBridge::der)
        .Case("diagonal", &FunctionCallBridge::diagonal)
        .Case("exp", &FunctionCallBridge::exp)
        .Case("identity", &FunctionCallBridge::identity)
        .Case("linspace", &FunctionCallBridge::linspace)
        .Case("log", &FunctionCallBridge::log)
        .Case("log10", &FunctionCallBridge::log10)
        .Case("max", &FunctionCallBridge::max)
        .Case("min", &FunctionCallBridge::min)
        .Case("ndims", &FunctionCallBridge::ndims)
        .Case("ones", &FunctionCallBridge::ones)
        .Case("product", &FunctionCallBridge::product)
        .Case("sign", &FunctionCallBridge::sign)
        .Case("sin", &FunctionCallBridge::sin)
        .Case("sinh", &FunctionCallBridge::sinh)
        .Case("size", &FunctionCallBridge::size)
        .Case("sqrt", &FunctionCallBridge::sqrt)
        .Case("sum", &FunctionCallBridge::sum)
        .Case("symmetric", &FunctionCallBridge::symmetric)
        .Case("tan", &FunctionCallBridge::tan)
        .Case("tanh", &FunctionCallBridge::tanh)
        .Case("transpose", &FunctionCallBridge::transpose)
        .Case("zeros", &FunctionCallBridge::zeros)
        .Default(&FunctionCallBridge::userDefinedFunction);

    FunctionCallBridge functionCallBridge(this);
    return lowerer(functionCallBridge, *call);
  }

  template<>
  Results NewLoweringBridge::lower<Tuple>(const Expression& expression)
  {
    assert(expression.isa<Tuple>());
    const auto* tuple = expression.get<Tuple>();
    Results result;

    for (const auto& exp : *tuple) {
      auto values = lower<Expression>(*exp);

      // The only way to have multiple returns is to call a function, but
      // this is forbidden in a tuple declaration. In fact, a tuple is just
      // a container of references.
      assert(values.size() == 1);
      result.append(values[0]);
    }

    return result;
  }

  template<>
  Results NewLoweringBridge::lower<Array>(const Expression& expression)
  {
    assert(expression.isa<Array>());
    const auto& array = expression.get<Array>();
    mlir::Location location = loc(expression.getLocation());
    auto arrayType = lower(array->getType(), ArrayAllocationScope::stack).cast<ArrayType>();
    auto allocationScope = arrayType.getAllocationScope();

    assert(allocationScope == ArrayAllocationScope::stack || allocationScope == ArrayAllocationScope::heap);

    mlir::Value result = allocationScope == ArrayAllocationScope::stack ?
        builder.create<AllocaOp>(location, arrayType, llvm::None).getResult() :
        builder.create<AllocOp>(location, arrayType, llvm::None).getResult();

    for (const auto& value : llvm::enumerate(*array)) {
      mlir::Value index = builder.create<ConstantOp>(location, builder.getIndexAttr(value.index()));
      mlir::Value slice = builder.create<SubscriptionOp>(location, result, index);
      builder.create<AssignmentOp>(location, *lower<Expression>(*value.value())[0], slice);
    }

    return Reference::ssa(&builder, result);
  }
}
