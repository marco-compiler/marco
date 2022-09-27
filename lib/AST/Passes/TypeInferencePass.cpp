#include "marco/AST/Passes/TypeInferencePass.h"
#include "marco/Diagnostic/Printer.h"
#include <sstream>

using namespace ::marco;
using namespace ::marco::ast;

namespace
{
  template<class T>
  std::string getTemporaryVariableName(T& cls)
  {
    const auto& members = cls.getMembers();
    int counter = 0;

    while (*(members.end()) !=
           *find_if(members.begin(), members.end(), [=](const auto& obj) {
             return obj->getName() == "_temp" + std::to_string(counter);
           }))
      counter++;

    return "_temp" + std::to_string(counter);
  }
}

static bool resolveDummyReferences(StandardFunction& function)
{
  for (auto& algorithm : function.getAlgorithms()) {
    for (auto& statement : algorithm->getBody()) {
      for (auto& assignment : *statement) {
        for (auto& destination : *assignment.getDestinations()->get<Tuple>()) {
          if (!destination->isa<ReferenceAccess>()) {
            continue;
          }

          auto* ref = destination->get<ReferenceAccess>();

          if (!ref->isDummy()) {
            continue;
          }

          std::string name = getTemporaryVariableName(function);
          auto temp = Member::build(destination->getLocation(), name, destination->getType(), TypePrefix::none());
          ref->setName(temp->getName());
          function.addMember(std::move(temp));

          // Note that there is no need to add the dummy variable to the
          // symbol table, because it will never be referenced.
        }
      }
    }
  }

  return true;
}

static bool resolveDummyReferences(Model& model)
{
  auto resolveFn = [&](ast::Expression& lhs) {
    if (auto* lhsTuple = lhs.dyn_get<Tuple>()) {
      for (auto& expression : *lhsTuple) {
        if (!expression->template isa<ReferenceAccess>()) {
          continue;
        }

        auto* ref = expression->template get<ReferenceAccess>();

        if (!ref->isDummy()) {
          continue;
        }

        std::string name = getTemporaryVariableName(model);
        auto temp = Member::build(expression->getLocation(), name, expression->getType(), TypePrefix::none());
        ref->setName(temp->getName());
        model.addMember(std::move(temp));

        // Note that there is no need to add the dummy variable to the
        // symbol table, because it will never be referenced.
      }
    }
  };

  for (auto& equationsBlock : model.getEquationsBlocks()) {
    for (auto& equation : equationsBlock->getEquations()) {
      resolveFn(*equation->getLhsExpression());
    }

    for (auto& forEquation : equationsBlock->getForEquations()) {
      resolveFn(*forEquation->getEquation()->getLhsExpression());
    }
  }

  for (auto& equationsBlock : model.getInitialEquationsBlocks()) {
    for (auto& equation : equationsBlock->getEquations()) {
      resolveFn(*equation->getLhsExpression());
    }

    for (auto& forEquation : equationsBlock->getForEquations()) {
      resolveFn(*forEquation->getEquation()->getLhsExpression());
    }
  }

  for (auto& algorithm : model.getAlgorithms()) {
    for (auto& statement : algorithm->getBody()) {
      for (auto& assignment : *statement) {
        resolveFn(*assignment.getDestinations());
      }
    }
  }

  return true;
}

static llvm::Optional<Type> builtInReferenceType(ReferenceAccess& reference)
{
  assert(!reference.isDummy());
  auto name = reference.getName();

  if (name == "time") {
    return makeType<BuiltInType::Real>();
  }

  return llvm::None;
}

//===----------------------------------------------------------------------===//
// Messages
//===----------------------------------------------------------------------===//

namespace
{
  class IncompatibleShapesMessage : public diagnostic::SourceMessage
  {
    public:
      IncompatibleShapesMessage(SourceRange location)
          : SourceMessage(std::move(location))
      {
      }

      void print(diagnostic::PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();

        auto highlightSourceFn = [&](llvm::raw_ostream& os) {
          printer->setColor(os, printer->diagnosticLevelColor());
        };

        printFileNameAndPosition(os);
        highlightSourceFn(os);
        printDiagnosticLevel(os, printer->diagnosticLevel());
        printer->resetColor(os);
        os << ": ";

        os << "incompatible shapes";
        os << "\n";

        printLines(os, highlightSourceFn);
      }
  };

  class UnvectorizableCallMessage : public diagnostic::SourceMessage
  {
    public:
      UnvectorizableCallMessage(SourceRange location)
          : SourceMessage(std::move(location))
      {
      }

      void print(diagnostic::PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();

        auto highlightSourceFn = [&](llvm::raw_ostream& os) {
          printer->setColor(os, printer->diagnosticLevelColor());
        };

        printFileNameAndPosition(os);
        highlightSourceFn(os);
        printDiagnosticLevel(os, printer->diagnosticLevel());
        printer->resetColor(os);
        os << ": ";

        os << "the function call can't be vectorized";
        os << "\n";

        printLines(os, highlightSourceFn);
      }
  };

  class UnknownSymbolMessage : public diagnostic::SourceMessage
  {
    public:
      UnknownSymbolMessage(SourceRange location, llvm::StringRef symbolName)
          : SourceMessage(std::move(location)),
            symbolName(symbolName.str())
      {
      }

      void print(diagnostic::PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();

        auto highlightSourceFn = [&](llvm::raw_ostream& os) {
          printer->setColor(os, printer->diagnosticLevelColor());
        };

        printFileNameAndPosition(os);
        highlightSourceFn(os);
        printDiagnosticLevel(os, printer->diagnosticLevel());
        printer->resetColor(os);
        os << ": ";

        os << "unknown symbol '";
        printer->setBold(os);
        os << symbolName;
        printer->unsetBold(os);
        os << "'";
        os << "\n";

        printLines(os, highlightSourceFn);
      }

    private:
      std::string symbolName;
  };

  class InvalidTypeMessage : public diagnostic::SourceMessage
  {
    public:
      InvalidTypeMessage(SourceRange location, llvm::StringRef name)
          : SourceMessage(std::move(location)),
          name(name.str())
      {
      }

      void print(diagnostic::PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();

        auto highlightSourceFn = [&](llvm::raw_ostream& os) {
          printer->setColor(os, printer->diagnosticLevelColor());
        };

        printFileNameAndPosition(os);
        highlightSourceFn(os);
        printDiagnosticLevel(os, printer->diagnosticLevel());
        printer->resetColor(os);
        os << ": ";

        os << "invalid type '";
        printer->setBold(os);
        os << name;
        printer->unsetBold(os);
        os << "'";
        os << "\n";

        printLines(os, highlightSourceFn);
      }

    private:
      std::string name;
  };

  class BadSemanticMessage : public diagnostic::SourceMessage
  {
    public:
      BadSemanticMessage(SourceRange location, llvm::StringRef message)
          : SourceMessage(std::move(location)),
            message(message.str())
      {
      }

      void print(diagnostic::PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();

        auto highlightSourceFn = [&](llvm::raw_ostream& os) {
          printer->setColor(os, printer->diagnosticLevelColor());
        };

        printFileNameAndPosition(os);
        highlightSourceFn(os);
        printDiagnosticLevel(os, printer->diagnosticLevel());
        printer->resetColor(os);
        os << ": ";
        os << message;
        os << "\n";

        printLines(os, highlightSourceFn);
      }

    private:
      std::string message;
  };
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace marco::ast
{
  TypeInferencePass::TypeInferencePass(diagnostic::DiagnosticEngine& diagnostics)
      : Pass(diagnostics)
  {
    for (auto& function : getBuiltInFunctions()) {
      builtInFunctions[function->getName()] = std::move(function);
    }
  }

  TypeInferencePass::SymbolTable& TypeInferencePass::getSymbolTable()
  {
    return symbolTable;
  }

  template<>
  bool TypeInferencePass::run<Class>(Class& cls)
  {
    return cls.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(cls);
    });
  }
  
  bool TypeInferencePass::run(std::unique_ptr<Class>& cls)
  {
    return run<Class>(*cls);
  }
  
  template<>
  bool TypeInferencePass::run<PartialDerFunction>(Class& cls)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    SymbolTable::ScopeTy scope(symbolTable);
    auto* derFunction = cls.get<PartialDerFunction>();

    // Populate the symbol table
    symbolTable.insert(derFunction->getName(), Symbol(cls));

    if (auto* derivedFunction = derFunction->getDerivedFunction(); !derivedFunction->isa<ReferenceAccess>()) {
      diagnostics()->emitError<BadSemanticMessage>(
          cls.getLocation(),
          "the derived function must be a reference");

      return numOfErrors == diagnostics()->numOfErrors();
    }

    Class* baseFunction = &cls;

    while (!baseFunction->isa<StandardFunction>()) {
      auto symbol = symbolTable.lookup(derFunction->getDerivedFunction()->get<ReferenceAccess>()->getName());

      if (symbol.isa<Class>()) {
        baseFunction = symbol.get<Class>();
      } else {
        diagnostics()->emitError<BadSemanticMessage>(
            cls.getLocation(),
            "the derived function name must refer to a function");

        return numOfErrors == diagnostics()->numOfErrors();
      }

      if (!cls.isa<StandardFunction>() && !cls.isa<PartialDerFunction>()) {
        diagnostics()->emitError<BadSemanticMessage>(
            cls.getLocation(),
            "the derived function name must refer to a function");

        return numOfErrors == diagnostics()->numOfErrors();
      }
    }

    auto* standardFunction = baseFunction->get<StandardFunction>();
    auto members = standardFunction->getMembers();
    llvm::SmallVector<size_t, 3> independentVariablesIndexes;

    for (auto& independentVariable : derFunction->getIndependentVariables()) {
      auto name = independentVariable->get<ReferenceAccess>()->getName();
      auto membersEnum = llvm::enumerate(members);

      auto member = llvm::find_if(membersEnum, [&name](const auto& obj) {
        return obj.value()->getName() == name;
      });

      if (member == membersEnum.end()) {
        diagnostics()->emitError<BadSemanticMessage>(
            independentVariable->getLocation(),
            "independent variable not found");

        return numOfErrors == diagnostics()->numOfErrors();
      }

      auto type = (*member).value()->getType();

      if (!type.isa<float>()) {
        diagnostics()->emitError<BadSemanticMessage>(
            independentVariable->getLocation(),
            "independent variables must have Real type");

        return numOfErrors == diagnostics()->numOfErrors();
      }

      independentVariable->setType(std::move(type));
      independentVariablesIndexes.push_back((*member).index());
    }

    llvm::SmallVector<Type, 3> argsTypes;
    llvm::SmallVector<Type, 3> resultsTypes;

    for (const auto& arg : standardFunction->getArgs()) {
      argsTypes.push_back(arg->getType());
    }

    for (const auto& result : standardFunction->getResults()) {
      resultsTypes.push_back(result->getType());
    }

    derFunction->setArgsTypes(argsTypes);
    derFunction->setResultsTypes(resultsTypes);

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<StandardFunction>(Class& cls)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    SymbolTable::ScopeTy scope(symbolTable);
    auto* function = cls.get<StandardFunction>();

    // Populate the symbol table
    symbolTable.insert(function->getName(), Symbol(cls));

    for (auto& member : function->getMembers()) {
      symbolTable.insert(member->getName(), Symbol(*member));
    }

    // Process the members
    for (auto& member : function->getMembers()) {
      if (!run(*member)) {
        return false;
      }
    }

    // Process the body
    for (auto& algorithm : function->getAlgorithms()) {
      if (!run(*algorithm)) {
        return false;
      }
    }

    if (!resolveDummyReferences(*function)) {
      return false;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<Model>(Class& cls)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    SymbolTable::ScopeTy scope(symbolTable);
    auto* model = cls.get<Model>();

    // Populate the symbol table
    symbolTable.insert(model->getName(), Symbol(cls));

    for (auto& member : model->getMembers()) {
      symbolTable.insert(member->getName(), Symbol(*member));
    }

    for (auto& innerClass : model->getInnerClasses()) {
      symbolTable.insert(innerClass->getName(), Symbol(*innerClass));
    }

    // Process the body
    for (auto& innerClass : model->getInnerClasses()) {
      if (!run<Class>(*innerClass)) {
        return false;
      }
    }

    for (auto& member : model->getMembers()) {
      if (!run(*member)) {
        return false;
      }
    }

    for (auto& equationsBlock : model->getEquationsBlocks()) {
      for (auto& equation : equationsBlock->getEquations()) {
        if (!run(*equation)) {
          return false;
        }
      }

      for (auto& forEquation : equationsBlock->getForEquations()) {
        if (!run(*forEquation)) {
          return false;
        }
      }
    }

    for (auto& equationsBlock : model->getInitialEquationsBlocks()) {
      for (auto& equation : equationsBlock->getEquations()) {
        if (!run(*equation)) {
          return false;
        }
      }

      for (auto& forEquation : equationsBlock->getForEquations()) {
        if (!run(*forEquation)) {
          return false;
        }
      }
    }

    for (auto& algorithm : model->getAlgorithms()) {
      if (!run(*algorithm)) {
        return false;
      }
    }

    if (!resolveDummyReferences(*model)) {
      return false;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<Package>(Class& cls)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    SymbolTable::ScopeTy scope(symbolTable);
    auto* package = cls.get<Package>();

    // Populate the symbol table
    symbolTable.insert(package->getName(), Symbol(cls));

    for (auto& innerClass : *package) {
      symbolTable.insert(innerClass->getName(), Symbol(*innerClass));
    }

    // Process the inner classes
    for (auto& innerClass : *package) {
      if (!run<Class>(*innerClass)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<Record>(Class& cls)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    SymbolTable::ScopeTy scope(symbolTable);
    auto* record = cls.get<Record>();

    // Populate the symbol table
    symbolTable.insert(record->getName(), Symbol(cls));

    for (auto& member : *record) {
      symbolTable.insert(member->getName(), Symbol(*member));
    }

    // Process the body
    for (auto& member : *record) {
      if (!run(*member)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<Expression>(Expression& expression)
  {
    return expression.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(expression);
    });
  }

  template<>
  bool TypeInferencePass::run<Array>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* array = expression.get<Array>();
    llvm::SmallVector<long, 3> sizes;

    llvm::Optional<Type> resultType;

    for (auto& element : *array) {
      if (!run<Expression>(*element)) {
        return false;
      }

      auto& elementType = element->getType();

      llvm::Optional<BuiltInType> mostGenericType = resultType ? getMostGenericBuiltInType(resultType->get<BuiltInType>(), elementType.get<BuiltInType>()) : elementType.get<BuiltInType>();
      
      assert(mostGenericType.has_value() && "array elements types are incompatible");

      resultType = *mostGenericType;
      auto rank = elementType.dimensionsCount();

      if (!elementType.isScalar()) {
        assert(sizes.empty() || sizes.size() == rank);

        if (sizes.empty()) {
          for (size_t i = 0; i < rank; ++i) {
            assert(!elementType[i].hasExpression());
            sizes.push_back(elementType[i].getNumericSize());
          }
        }
      }
    }

    llvm::SmallVector<ArrayDimension, 3> dimensions;
    dimensions.emplace_back(array->size());

    for (auto size : sizes) {
      dimensions.emplace_back(size);
    }

    resultType->setDimensions(dimensions);
    expression.setType(*resultType);

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<Call>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* call = expression.get<Call>();

    for (auto& arg : *call) {
      if (!run<Expression>(*arg)) {
        return false;
      }
    }

    auto* function = call->getFunction();
    llvm::StringRef functionName = function->get<ReferenceAccess>()->getName();
    bool canBeCalledElementWise = true;
    llvm::SmallVector<long, 3> argsExpectedRanks;

    // If the function name refers to a built-in one, we also need to check
    // whether there is a user defined one that is shadowing it. If this is
    // not the case, then we can fall back to the real built-in function.

    if (symbolTable.count(functionName) == 0 && builtInFunctions.count(functionName) != 0) {
      // Built-in function
      auto& builtInFunction = builtInFunctions[functionName];
      auto resultType = builtInFunction->resultType(call->getArgs());

      if (!resultType.has_value()) {
        diagnostics()->emitError<UnknownSymbolMessage>(function->getLocation(), functionName);
        return numOfErrors == diagnostics()->numOfErrors();
      }

      function->setType(*resultType);
      canBeCalledElementWise = builtInFunction->canBeCalledElementWise();
      builtInFunction->getArgsExpectedRanks(call->argumentsCount(), argsExpectedRanks);
    } else {
      // User defined function

      if (!run<Expression>(*function)) {
        return false;
      }

      auto functionTypeResolver = [&](llvm::StringRef name) -> llvm::Optional<FunctionType> {
        if (symbolTable.count(name) == 0)
          return llvm::None;

        auto symbol = symbolTable.lookup(functionName);

        if (const auto* cls = symbol.dyn_get<Class>()) {
          if (const auto* standardFunction = cls->dyn_get<StandardFunction>()) {
            return standardFunction->getType();
          }

          if (const auto* partialDerFunction = cls->dyn_get<PartialDerFunction>()) {
            return partialDerFunction->getType();
          }

          if (auto* record = cls->dyn_get<Record>()) {
            // Calling the default record constructor
            canBeCalledElementWise = false;
            return record->getDefaultConstructor().getType();
          }
        }

        return llvm::None;
      };

      auto functionType = functionTypeResolver(functionName);

      if (!functionType.has_value()) {
        diagnostics()->emitError<UnknownSymbolMessage>(function->getLocation(), functionName);
        return numOfErrors == diagnostics()->numOfErrors();
      }

      for (const auto& type : functionType->getArgs()) {
        argsExpectedRanks.push_back(type.getRank());
      }
    }

    if (canBeCalledElementWise) {
      llvm::SmallVector<ArrayDimension, 3> dimensions;

      for (const auto& arg : llvm::enumerate(call->getArgs())) {
        unsigned int argActualRank = arg.value()->getType().getRank();
        unsigned int argExpectedRank = argsExpectedRanks[arg.index()];

        if (arg.index() == 0) {
          // If this is the first argument, then it will determine the
          // rank and dimensions of the result array, although the dimensions
          // can be also specialized by the other arguments if initially unknown.

          for (size_t i = 0; i < argActualRank - argExpectedRank; ++i) {
            auto& dimension = arg.value()->getType()[arg.index()];
            dimensions.push_back(dimension.isDynamic() ? -1 : dimension.getNumericSize());
          }
        } else {
          // The rank difference must match with the one given by the first
          // argument, independently of the dimensions sizes.
          if (argActualRank != argExpectedRank + dimensions.size()) {
            diagnostics()->emitError<UnvectorizableCallMessage>(expression.getLocation());
            return numOfErrors == diagnostics()->numOfErrors();
          }

          for (size_t i = 0; i < argActualRank - argExpectedRank; ++i) {
            auto& dimension = arg.value()->getType()[arg.index()];

            // If the dimension is dynamic, then no further checks or
            // specializations are possible.
            if (dimension.isDynamic()) {
              continue;
            }

            // If the dimension determined by the first argument is fixed, then
            // also the dimension of the other arguments must match (when that's
            // fixed too).
            if (!dimensions[i].isDynamic() && dimensions[i] != dimension) {
              diagnostics()->emitError<UnvectorizableCallMessage>(expression.getLocation());
              return numOfErrors == diagnostics()->numOfErrors();
            }

            // If the dimension determined by the first argument is dynamic, then
            // set it to a required size.
            if (dimensions[i].isDynamic()) {
              dimensions[i] = dimension;
            }
          }
        }
      }

      if (dimensions.empty()) {
        expression.setType(call->getFunction()->getType());
      } else {
        expression.setType(call->getFunction()->getType().to(dimensions));
      }
    } else {
      expression.setType(function->getType());
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<Constant>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::add>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto lhsType = operation->getArg(0)->getType();
    auto rhsType = operation->getArg(1)->getType();
    assert(lhsType.isa<BuiltInType>() && rhsType.isa<BuiltInType>());

    auto lhsBaseType = lhsType.get<BuiltInType>();
    auto rhsBaseType = rhsType.get<BuiltInType>();
    auto baseType = *getMostGenericBuiltInType(lhsBaseType, rhsBaseType);

    if (lhsType.getRank() != rhsType.getRank()) {
      diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
      return numOfErrors == diagnostics()->numOfErrors();
    }

    std::vector<ArrayDimension> resultDimensions;

    for (size_t i = 0; i < lhsType.getRank(); ++i) {
      const auto& lhsDim = lhsType[i];
      const auto& rhsDim = rhsType[i];

      if (!lhsDim.isDynamic() && !rhsDim.isDynamic() && lhsDim.getNumericSize() != rhsDim.getNumericSize()) {
        diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
        return numOfErrors == diagnostics()->numOfErrors();
      }

      if (!lhsDim.isDynamic()) {
        resultDimensions.emplace_back(lhsDim);
      } else if (!rhsDim.isDynamic()) {
        resultDimensions.emplace_back(rhsDim);
      } else {
        resultDimensions.emplace_back(ArrayDimension::kDynamicSize);
      }
    }

    expression.setType(Type(baseType, resultDimensions));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::addEW>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto lhsType = operation->getArg(0)->getType();
    auto rhsType = operation->getArg(1)->getType();
    assert(lhsType.isa<BuiltInType>() && rhsType.isa<BuiltInType>());

    auto lhsBaseType = lhsType.get<BuiltInType>();
    auto rhsBaseType = rhsType.get<BuiltInType>();
    auto baseType = *getMostGenericBuiltInType(lhsBaseType, rhsBaseType);

    if (lhsType.getRank() == 0 && rhsType.getRank() == 0) {
      expression.setType(Type(baseType, llvm::None));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() == 0 && rhsType.getRank() != 0) {
      expression.setType(Type(baseType, rhsType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() != 0 && rhsType.getRank() == 0) {
      expression.setType(Type(baseType, lhsType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() != rhsType.getRank()) {
      diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
      return numOfErrors == diagnostics()->numOfErrors();
    }

    std::vector<ArrayDimension> resultDimensions;

    for (size_t i = 0; i < lhsType.getRank(); ++i) {
      const auto& lhsDim = lhsType[i];
      const auto& rhsDim = rhsType[i];

      if (!lhsDim.isDynamic() && !rhsDim.isDynamic() && lhsDim.getNumericSize() != rhsDim.getNumericSize()) {
        diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
        return numOfErrors == diagnostics()->numOfErrors();
      }

      if (!lhsDim.isDynamic()) {
        resultDimensions.emplace_back(lhsDim);
      } else if (!rhsDim.isDynamic()) {
        resultDimensions.emplace_back(rhsDim);
      } else {
        resultDimensions.emplace_back(ArrayDimension::kDynamicSize);
      }
    }

    expression.setType(Type(baseType, resultDimensions));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::different>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    [[maybe_unused]] auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);
    expression.setType(Type(BuiltInType::Boolean, llvm::None));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::divide>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto lhsType = operation->getArg(0)->getType();
    auto rhsType = operation->getArg(1)->getType();
    assert(lhsType.isa<BuiltInType>() && rhsType.isa<BuiltInType>());

    auto lhsBaseType = lhsType.get<BuiltInType>();
    auto rhsBaseType = rhsType.get<BuiltInType>();
    auto baseType = *getMostGenericBuiltInType(lhsBaseType, rhsBaseType);

    if (lhsType.getRank() == 0 && rhsType.getRank() == 0) {
      expression.setType(Type(baseType, llvm::None));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() != 0 && rhsType.getRank() == 0) {
      expression.setType(Type(baseType, lhsType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::divideEW>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto lhsType = operation->getArg(0)->getType();
    auto rhsType = operation->getArg(1)->getType();
    assert(lhsType.isa<BuiltInType>() && rhsType.isa<BuiltInType>());

    auto lhsBaseType = lhsType.get<BuiltInType>();
    auto rhsBaseType = rhsType.get<BuiltInType>();
    auto baseType = *getMostGenericBuiltInType(lhsBaseType, rhsBaseType);

    if (lhsType.getRank() == 0 && rhsType.getRank() == 0) {
      expression.setType(Type(baseType, llvm::None));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() == 0 && rhsType.getRank() != 0) {
      expression.setType(Type(baseType, rhsType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() != 0 && rhsType.getRank() == 0) {
      expression.setType(Type(baseType, lhsType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() != rhsType.getRank()) {
      diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
      return numOfErrors == diagnostics()->numOfErrors();
    }

    std::vector<ArrayDimension> resultDimensions;

    for (size_t i = 0; i < lhsType.getRank(); ++i) {
      const auto& lhsDim = lhsType[i];
      const auto& rhsDim = rhsType[i];

      if (!lhsDim.isDynamic() && !rhsDim.isDynamic() && lhsDim.getNumericSize() != rhsDim.getNumericSize()) {
        diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
        return numOfErrors == diagnostics()->numOfErrors();
      }

      if (!lhsDim.isDynamic()) {
        resultDimensions.emplace_back(lhsDim);
      } else if (!rhsDim.isDynamic()) {
        resultDimensions.emplace_back(rhsDim);
      } else {
        resultDimensions.emplace_back(ArrayDimension::kDynamicSize);
      }
    }

    expression.setType(Type(baseType, resultDimensions));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::equal>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    [[maybe_unused]] auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);
    expression.setType(Type(BuiltInType::Boolean, llvm::None));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::greater>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    [[maybe_unused]] auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);
    expression.setType(Type(BuiltInType::Boolean, llvm::None));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::greaterEqual>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    [[maybe_unused]] auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);
    expression.setType(Type(BuiltInType::Boolean, llvm::None));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::ifelse>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 3);

    auto trueValueType = operation->getArg(1)->getType();
    auto falseValueType = operation->getArg(2)->getType();
    assert(trueValueType.isa<BuiltInType>() && falseValueType.isa<BuiltInType>());

    auto trueValueBaseType = trueValueType.get<BuiltInType>();
    auto falseValueBaseType = falseValueType.get<BuiltInType>();
    auto baseType = *getMostGenericBuiltInType(trueValueBaseType, falseValueBaseType);

    if (trueValueType.getRank() != falseValueType.getRank()) {
      diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
      return numOfErrors == diagnostics()->numOfErrors();
    }

    std::vector<ArrayDimension> resultDimensions;

    for (size_t i = 0; i < trueValueType.getRank(); ++i) {
      const auto& trueValueDim = trueValueType[i];
      const auto& falseValueDim = falseValueType[i];

      if (!trueValueDim.isDynamic() && !falseValueDim.isDynamic() && trueValueDim.getNumericSize() != falseValueDim.getNumericSize()) {
        diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
        return numOfErrors == diagnostics()->numOfErrors();
      }

      if (!trueValueDim.isDynamic()) {
        resultDimensions.emplace_back(trueValueDim);
      } else if (!falseValueDim.isDynamic()) {
        resultDimensions.emplace_back(falseValueDim);
      } else {
        resultDimensions.emplace_back(ArrayDimension::kDynamicSize);
      }
    }

    expression.setType(Type(baseType, resultDimensions));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::land>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto lhsType = operation->getArg(0)->getType();
    auto rhsType = operation->getArg(1)->getType();
    assert(lhsType.isa<BuiltInType>() && rhsType.isa<BuiltInType>());

    if (lhsType.getRank() != rhsType.getRank()) {
      diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
      return numOfErrors == diagnostics()->numOfErrors();
    }

    std::vector<ArrayDimension> resultDimensions;

    for (size_t i = 0; i < lhsType.getRank(); ++i) {
      const auto& lhsDim = lhsType[i];
      const auto& rhsDim = rhsType[i];

      if (!lhsDim.isDynamic() && !rhsDim.isDynamic() && lhsDim.getNumericSize() != rhsDim.getNumericSize()) {
        diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
        return numOfErrors == diagnostics()->numOfErrors();
      }

      if (!lhsDim.isDynamic()) {
        resultDimensions.emplace_back(lhsDim);
      } else if (!rhsDim.isDynamic()) {
        resultDimensions.emplace_back(rhsDim);
      } else {
        resultDimensions.emplace_back(ArrayDimension::kDynamicSize);
      }
    }

    expression.setType(Type(BuiltInType::Boolean, resultDimensions));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::lnot>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 1);
    expression.setType(Type(BuiltInType::Boolean, operation->getArg(0)->getType().getDimensions()));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::lor>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto lhsType = operation->getArg(0)->getType();
    auto rhsType = operation->getArg(1)->getType();
    assert(lhsType.isa<BuiltInType>() && rhsType.isa<BuiltInType>());

    if (lhsType.getRank() != rhsType.getRank()) {
      diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
      return numOfErrors == diagnostics()->numOfErrors();
    }

    std::vector<ArrayDimension> resultDimensions;

    for (size_t i = 0; i < lhsType.getRank(); ++i) {
      const auto& lhsDim = lhsType[i];
      const auto& rhsDim = rhsType[i];

      if (!lhsDim.isDynamic() && !rhsDim.isDynamic() && lhsDim.getNumericSize() != rhsDim.getNumericSize()) {
        diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
        return numOfErrors == diagnostics()->numOfErrors();
      }

      if (!lhsDim.isDynamic()) {
        resultDimensions.emplace_back(lhsDim);
      } else if (!rhsDim.isDynamic()) {
        resultDimensions.emplace_back(rhsDim);
      } else {
        resultDimensions.emplace_back(ArrayDimension::kDynamicSize);
      }
    }

    expression.setType(Type(BuiltInType::Boolean, resultDimensions));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::less>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    [[maybe_unused]] auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);
    expression.setType(Type(BuiltInType::Boolean, llvm::None));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::lessEqual>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    [[maybe_unused]] auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);
    expression.setType(Type(BuiltInType::Boolean, llvm::None));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::memberLookup>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->size() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (!run<Expression>(*lhs)) {
      return false;
    }

    if (!lhs->getType().isa<Record*>()) {
      diagnostics()->emitError<BadSemanticMessage>(
          expression.getLocation(),
          "member lookup is implemented only for records");

      return numOfErrors == diagnostics()->numOfErrors();
    }

    assert(rhs->isa<ReferenceAccess>());

    const auto* record = lhs->getType().get<Record*>();
    const auto memberName = rhs->get<ReferenceAccess>()->getName();
    const auto *member = (*record)[memberName];

    assert(member && "member not found");

    expression.setType(getFlattenedMemberType(lhs->getType(), member->getType()));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::multiply>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto lhsType = operation->getArg(0)->getType();
    auto rhsType = operation->getArg(1)->getType();
    assert(lhsType.isa<BuiltInType>() && rhsType.isa<BuiltInType>());

    auto lhsBaseType = lhsType.get<BuiltInType>();
    auto rhsBaseType = rhsType.get<BuiltInType>();
    auto baseType = *getMostGenericBuiltInType(lhsBaseType, rhsBaseType);

    if (lhsType.getRank() == 0 && rhsType.getRank() == 0) {
      expression.setType(Type(baseType, llvm::None));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() == 0 && rhsType.getRank() != 0) {
      expression.setType(Type(baseType, rhsType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() == 1 && rhsType.getRank() == 1) {
      expression.setType(Type(baseType, llvm::None));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() == 1 && rhsType.getRank() == 2) {
      expression.setType(Type(baseType, rhsType[1]));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() == 2 && rhsType.getRank() == 1) {
      expression.setType(Type(baseType, lhsType[0]));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() == 2 && rhsType.getRank() == 2) {
      expression.setType(Type(baseType, { lhsType[0], rhsType[1] }));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::multiplyEW>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto lhsType = operation->getArg(0)->getType();
    auto rhsType = operation->getArg(1)->getType();
    assert(lhsType.isa<BuiltInType>() && rhsType.isa<BuiltInType>());

    auto lhsBaseType = lhsType.get<BuiltInType>();
    auto rhsBaseType = rhsType.get<BuiltInType>();
    auto baseType = *getMostGenericBuiltInType(lhsBaseType, rhsBaseType);

    if (lhsType.getRank() == 0 && rhsType.getRank() == 0) {
      expression.setType(Type(baseType, llvm::None));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() == 0 && rhsType.getRank() != 0) {
      expression.setType(Type(baseType, rhsType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() != 0 && rhsType.getRank() == 0) {
      expression.setType(Type(baseType, lhsType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() != rhsType.getRank()) {
      diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
      return numOfErrors == diagnostics()->numOfErrors();
    }

    std::vector<ArrayDimension> resultDimensions;

    for (size_t i = 0; i < lhsType.getRank(); ++i) {
      const auto& lhsDim = lhsType[i];
      const auto& rhsDim = rhsType[i];

      if (!lhsDim.isDynamic() && !rhsDim.isDynamic() && lhsDim.getNumericSize() != rhsDim.getNumericSize()) {
        diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
        return numOfErrors == diagnostics()->numOfErrors();
      }

      if (!lhsDim.isDynamic()) {
        resultDimensions.emplace_back(lhsDim);
      } else if (!rhsDim.isDynamic()) {
        resultDimensions.emplace_back(rhsDim);
      } else {
        resultDimensions.emplace_back(ArrayDimension::kDynamicSize);
      }
    }

    expression.setType(Type(baseType, resultDimensions));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::negate>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 1);
    expression.setType(operation->getArg(0)->getType());

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::powerOf>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);
    expression.setType(operation->getArg(0)->getType());

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::powerOfEW>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto xType = operation->getArg(0)->getType();
    auto yType = operation->getArg(1)->getType();
    assert(xType.isa<BuiltInType>() && yType.isa<BuiltInType>());

    auto baseType = xType.get<BuiltInType>();

    if (xType.getRank() == 0 && yType.getRank() == 0) {
      expression.setType(Type(baseType, llvm::None));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (xType.getRank() == 0 && yType.getRank() != 0) {
      expression.setType(Type(baseType, yType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (xType.getRank() != 0 && yType.getRank() == 0) {
      expression.setType(Type(baseType, xType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (xType.getRank() != yType.getRank()) {
      diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
      return numOfErrors == diagnostics()->numOfErrors();
    }

    std::vector<ArrayDimension> resultDimensions;

    for (size_t i = 0; i < xType.getRank(); ++i) {
      const auto& xDim = xType[i];
      const auto& yDim = yType[i];

      if (!xDim.isDynamic() && !yDim.isDynamic() && xDim.getNumericSize() != yDim.getNumericSize()) {
        diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
        return numOfErrors == diagnostics()->numOfErrors();
      }

      if (!xDim.isDynamic()) {
        resultDimensions.emplace_back(xDim);
      } else if (!yDim.isDynamic()) {
        resultDimensions.emplace_back(yDim);
      } else {
        resultDimensions.emplace_back(ArrayDimension::kDynamicSize);
      }
    }

    expression.setType(Type(baseType, resultDimensions));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::range>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();

    if (operation->argumentsCount() == 2) {
      auto* j = operation->getArg(0);
      auto* k = operation->getArg(1);

      auto isInteger = j->getType().isa<int>() && k->getType().isa<int>();

      if (isInteger) {
        auto jValue = j->get<Constant>()->as<BuiltInType::Integer>();
        auto kValue = k->get<Constant>()->as<BuiltInType::Integer>();
        auto n = kValue - jValue;

        expression.setType(Type(BuiltInType::Integer, { 1 + n }));
        return numOfErrors == diagnostics()->numOfErrors();
      } else {
        auto jValue = j->get<Constant>()->as<BuiltInType::Real>();
        auto kValue = k->get<Constant>()->as<BuiltInType::Real>();
        auto n = static_cast<long>(std::floor(kValue - jValue));

        expression.setType(Type(BuiltInType::Real, { 1 + n }));
        return numOfErrors == diagnostics()->numOfErrors();
      }
    }

    if (operation->argumentsCount() == 3) {
      auto* j = operation->getArg(0);
      auto* d = operation->getArg(1);
      auto* k = operation->getArg(2);

      auto isInteger = j->getType().isa<int>() && d->getType().isa<int>() && k->getType().isa<int>();

      if (isInteger) {
        auto jValue = j->get<Constant>()->as<BuiltInType::Integer>();
        auto dValue = d->get<Constant>()->as<BuiltInType::Integer>();
        auto kValue = k->get<Constant>()->as<BuiltInType::Integer>();
        auto n = (kValue - jValue) / dValue;

        expression.setType(Type(BuiltInType::Integer, { 1 + n }));
        return numOfErrors == diagnostics()->numOfErrors();
      } else {
        auto jValue = j->get<Constant>()->as<BuiltInType::Real>();
        auto dValue = d->get<Constant>()->as<BuiltInType::Real>();
        auto kValue = k->get<Constant>()->as<BuiltInType::Real>();
        auto n = static_cast<long>(std::floor((kValue - jValue) / dValue));

        expression.setType(Type(BuiltInType::Real, { 1 + n }));
        return numOfErrors == diagnostics()->numOfErrors();
      }
    }

    llvm_unreachable("Unexpected range");
    return false;
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::subscription>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() >= 2);

    auto arrayType = operation->getArg(0)->getType();

    std::vector<ArrayDimension> resultDimensions;
    auto numOfSubscriptions = operation->argumentsCount() - 1;

    for (size_t i = 0; i < arrayType.getRank() - numOfSubscriptions; ++i) {
      resultDimensions.push_back(arrayType[i]);
    }

    std::vector<bool> unboundedRanges(numOfSubscriptions, false);
    std::vector<ArrayDimension> additionalDimensions;

    for (size_t i = 0; i < numOfSubscriptions; ++i) {
      auto index = operation->getArg(i + 1);

      if (index->isa<Constant>()) {
        assert(index->getType().isa<int>());
        auto value = index->get<Constant>()->as<BuiltInType::Integer>();

        if (value == ArrayDimension::kDynamicSize) {
          unboundedRanges[i] = true;
          additionalDimensions.push_back(arrayType[arrayType.getRank() - numOfSubscriptions + i]);
        }
      } else if (index->isa<Operation>()) {
        auto* op = index->get<Operation>();

        if (op->getOperationKind() == OperationKind::range) {
          for (const auto& dimension : op->getType().getDimensions()) {
            additionalDimensions.push_back(dimension);
          }
        }
      }
    }

    size_t removableDimensions = 0;

    for (size_t i = 0, e = unboundedRanges.size(); i < e; ++i) {
      if (unboundedRanges[e - i - 1]) {
        ++removableDimensions;
      } else {
        break;
      }
    }

    for (size_t i = 0; i < additionalDimensions.size() - removableDimensions; ++i) {
      resultDimensions.push_back(additionalDimensions[i]);
    }

    expression.setType(arrayType.visit([&](const auto& type) {
      if (resultDimensions.empty()) {
        return Type(type);
      } else {
        return Type(type, resultDimensions);
      }
    }));

    // Temporary solution: remove also the subscriptions.
    // In future, a dedicated pass should be created to introduce the additional for equations,
    // instead of erasing some expressions

    std::vector<std::unique_ptr<Expression>> args;

    for (size_t i = 0; i < operation->argumentsCount() - removableDimensions; ++i) {
      args.push_back(operation->getArg(0)->clone());
    }

    if (args.size() == 1) {
      expression = std::move(*args[0]);
    } else {
      auto argsCount = operation->argumentsCount();

      for (size_t i = 0; i < removableDimensions; ++i) {
        operation->removeArg(argsCount - i - 1);
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::subtract>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto lhsType = operation->getArg(0)->getType();
    auto rhsType = operation->getArg(1)->getType();
    assert(lhsType.isa<BuiltInType>() && rhsType.isa<BuiltInType>());

    auto lhsBaseType = lhsType.get<BuiltInType>();
    auto rhsBaseType = rhsType.get<BuiltInType>();
    auto baseType = *getMostGenericBuiltInType(lhsBaseType, rhsBaseType);

    if (lhsType.getRank() != rhsType.getRank()) {
      diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
      return numOfErrors == diagnostics()->numOfErrors();
    }

    std::vector<ArrayDimension> resultDimensions;

    for (size_t i = 0; i < lhsType.getRank(); ++i) {
      const auto& lhsDim = lhsType[i];
      const auto& rhsDim = rhsType[i];

      if (!lhsDim.isDynamic() && !rhsDim.isDynamic() && lhsDim.getNumericSize() != rhsDim.getNumericSize()) {
        diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
        return numOfErrors == diagnostics()->numOfErrors();
      }

      if (!lhsDim.isDynamic()) {
        resultDimensions.emplace_back(lhsDim);
      } else if (!rhsDim.isDynamic()) {
        resultDimensions.emplace_back(rhsDim);
      } else {
        resultDimensions.emplace_back(ArrayDimension::kDynamicSize);
      }
    }

    expression.setType(Type(baseType, resultDimensions));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::processOp<OperationKind::subtractEW>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto lhsType = operation->getArg(0)->getType();
    auto rhsType = operation->getArg(1)->getType();
    assert(lhsType.isa<BuiltInType>() && rhsType.isa<BuiltInType>());

    auto lhsBaseType = lhsType.get<BuiltInType>();
    auto rhsBaseType = rhsType.get<BuiltInType>();
    auto baseType = *getMostGenericBuiltInType(lhsBaseType, rhsBaseType);

    if (lhsType.getRank() == 0 && rhsType.getRank() == 0) {
      expression.setType(Type(baseType, llvm::None));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() == 0 && rhsType.getRank() != 0) {
      expression.setType(Type(baseType, rhsType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() != 0 && rhsType.getRank() == 0) {
      expression.setType(Type(baseType, lhsType.getDimensions()));
      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhsType.getRank() != rhsType.getRank()) {
      diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
      return numOfErrors == diagnostics()->numOfErrors();
    }

    std::vector<ArrayDimension> resultDimensions;

    for (size_t i = 0; i < lhsType.getRank(); ++i) {
      const auto& lhsDim = lhsType[i];
      const auto& rhsDim = rhsType[i];

      if (!lhsDim.isDynamic() && !rhsDim.isDynamic() && lhsDim.getNumericSize() != rhsDim.getNumericSize()) {
        diagnostics()->emitError<IncompatibleShapesMessage>(expression.getLocation());
        return numOfErrors == diagnostics()->numOfErrors();
      }

      if (!lhsDim.isDynamic()) {
        resultDimensions.emplace_back(lhsDim);
      } else if (!rhsDim.isDynamic()) {
        resultDimensions.emplace_back(rhsDim);
      } else {
        resultDimensions.emplace_back(ArrayDimension::kDynamicSize);
      }
    }

    expression.setType(Type(baseType, resultDimensions));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<Operation>(Expression& expression)
  {
    auto* operation = expression.get<Operation>();

    // Process the arguments
    for (size_t i = 0; i < operation->argumentsCount(); ++i) {
      if (!run<Expression>(*operation->getArg(i))) {
        return false;
      }
    }

    // Apply the operation-specific semantics
    switch (operation->getOperationKind()) {
      case OperationKind::add:
        return processOp<OperationKind::add>(expression);

      case OperationKind::addEW:
        return processOp<OperationKind::addEW>(expression);

      case OperationKind::different:
        return processOp<OperationKind::different>(expression);

      case OperationKind::divide:
        return processOp<OperationKind::divide>(expression);

      case OperationKind::divideEW:
        return processOp<OperationKind::divideEW>(expression);

      case OperationKind::equal:
        return processOp<OperationKind::equal>(expression);

      case OperationKind::greater:
        return processOp<OperationKind::greater>(expression);

      case OperationKind::greaterEqual:
        return processOp<OperationKind::greaterEqual>(expression);

      case OperationKind::ifelse:
        return processOp<OperationKind::ifelse>(expression);

      case OperationKind::less:
        return processOp<OperationKind::less>(expression);

      case OperationKind::lessEqual:
        return processOp<OperationKind::lessEqual>(expression);

      case OperationKind::land:
        return processOp<OperationKind::land>(expression);

      case OperationKind::lnot:
        return processOp<OperationKind::lnot>(expression);

      case OperationKind::lor:
        return processOp<OperationKind::lor>(expression);

      case OperationKind::memberLookup:
        return processOp<OperationKind::memberLookup>(expression);

      case OperationKind::multiply:
        return processOp<OperationKind::multiply>(expression);

      case OperationKind::multiplyEW:
        return processOp<OperationKind::multiplyEW>(expression);

      case OperationKind::negate:
        return processOp<OperationKind::negate>(expression);

      case OperationKind::powerOf:
        return processOp<OperationKind::powerOf>(expression);

      case OperationKind::powerOfEW:
        return processOp<OperationKind::powerOfEW>(expression);

      case OperationKind::range:
        return processOp<OperationKind::range>(expression);

      case OperationKind::subscription:
        return processOp<OperationKind::subscription>(expression);

      case OperationKind::subtract:
        return processOp<OperationKind::subtract>(expression);

      case OperationKind::subtractEW:
        return processOp<OperationKind::subtractEW>(expression);
    }

    llvm_unreachable("Unknown operation kind");
    return false;
  }

  template<>
  bool TypeInferencePass::run<ReferenceAccess>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* reference = expression.get<ReferenceAccess>();

    // If the referenced variable is a dummy one (meaning that it is created
    // to store a result value that will never be used), its type is still
    // unknown and will be determined according to the assigned value.

    if (reference->isDummy()) {
      return true;
    }

    auto name = reference->getName();

    if (symbolTable.count(name) == 0) {
      if (auto type = builtInReferenceType(*reference); type.has_value()) {
        expression.setType(type.value());
        return numOfErrors == diagnostics()->numOfErrors();
      }

      // Handle records member lookup chains in the form 'record.member.members_member'.
      // We need to split the identifier in the single parts and check / iterate them.

      if (name.find('.') != std::string::npos) {
        std::stringstream ss(name.str());
        std::string item;

        getline(ss, item, '.');

        if (symbolTable.count(item)) {
          auto symbol = symbolTable.lookup(item);
          const auto *member = symbol.dyn_get<Member>();

          auto loc = expression.getLocation();
          auto new_expression = Expression::reference(loc, member->getType(), item);
          auto t = member->getType();

          while (member && getline(ss, item, '.')) {
            auto memberName = Expression::reference(loc, makeType<std::string>(), item);

            new_expression = Expression::operation(
                loc, t, OperationKind::memberLookup,
                llvm::ArrayRef({ std::move(new_expression), std::move(memberName) })
            );

            if (t.isa<Record*>()){
              member = (*t.get<Record*>())[item];
              t = getFlattenedMemberType(t,member->getType());
            }
          }

          if (member && !ss) {
            new_expression->setType(t);
            expression = *new_expression;

            // The type of the reference is the type of the last accessed member
            return numOfErrors == diagnostics()->numOfErrors();
          }
        }
      }

      diagnostics()->emitError<UnknownSymbolMessage>(expression.getLocation(), name);
      return numOfErrors == diagnostics()->numOfErrors();
    }

    auto symbol = symbolTable.lookup(name);

    auto symbolType = [](Symbol& symbol) -> Type {
      auto* cls = symbol.dyn_get<Class>();

      if (cls && cls->isa<StandardFunction>())
        return cls->get<StandardFunction>()->getType().packResults();

      if (cls && cls->isa<PartialDerFunction>()) {
        auto types = cls->get<PartialDerFunction>()->getResultsTypes();

        if (types.size() == 1) {
          return types[0];
        }

        return Type(PackedType(types));
      }

      if (cls && cls->isa<Record>()) {
        return Type(cls->get<Record>());
      }

      if (symbol.isa<Member>()) {
        return symbol.get<Member>()->getType();
      }

      if (symbol.isa<Induction>()) {
        return makeType<BuiltInType::Integer>();
      }

      llvm_unreachable("Unknown symbol type");
      return Type::unknown();
    };

    expression.setType(symbolType(symbol));
    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<Tuple>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* tuple = expression.get<Tuple>();

    llvm::SmallVector<Type, 3> types;

    for (auto& exp : *tuple) {
      if (!run<Expression>(*exp)) {
        return false;
      }

      types.push_back(exp->getType());
    }

    expression.setType(Type(PackedType(types)));

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<RecordInstance>(Expression& expression)
  {
    return true;
  }

  bool TypeInferencePass::run(Equation& equation)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    if (!run<Expression>(*equation.getLhsExpression())) {
      return false;
    }

    if (!run<Expression>(*equation.getRhsExpression())) {
      return false;
    }

    auto* lhs = equation.getLhsExpression();
    auto* rhs = equation.getRhsExpression();

    const auto& rhsType = rhs->getType();

    if (auto* lhsTuple = lhs->dyn_get<Tuple>()) {
      if (!rhsType.isa<PackedType>() || lhsTuple->size() != rhsType.get<PackedType>().size()) {
        diagnostics()->emitError<BadSemanticMessage>(
            equation.getLocation(),
            "number of results don't match with the destination tuple size");

        return numOfErrors == diagnostics()->numOfErrors();
      }

      assert(rhs->getType().isa<PackedType>());
      auto& rhsTypes = rhs->getType().get<PackedType>();

      // Assign type to dummy variables.
      // The assignment can't be done earlier because the expression type would
      // have not been evaluated yet.

      for (size_t i = 0; i < lhsTuple->size(); ++i) {
        // If it's not a direct reference access, there's no way it can be a
        // dummy variable.

        if (!lhsTuple->getArg(i)->isa<ReferenceAccess>()) {
          continue;
        }

        auto* ref = lhsTuple->getArg(i)->get<ReferenceAccess>();

        if (ref->isDummy()) {
          assert(rhsTypes.size() >= i);
          lhsTuple->getArg(i)->setType(rhsTypes[i]);
        }
      }
    }

    // If the function call has more return values than the provided
    // destinations, then we need to add more dummy references.

    if (rhsType.isa<PackedType>()) {
      const auto& rhsPackedType = rhsType.get<PackedType>();
      size_t returns = rhsPackedType.size();

      llvm::SmallVector<std::unique_ptr<Expression>, 3> newDestinations;
      llvm::SmallVector<Type, 3> destinationsTypes;

      if (auto* lhsTuple = lhs->dyn_get<Tuple>()) {
        for (auto& destination : *lhsTuple) {
          destinationsTypes.push_back(destination->getType());
          newDestinations.push_back(std::move(destination));
        }
      } else {
        destinationsTypes.push_back(lhs->getType());
        newDestinations.push_back(lhs->clone());
      }

      for (size_t i = newDestinations.size(); i < returns; ++i) {
        destinationsTypes.push_back(rhsPackedType[i]);
        newDestinations.push_back(ReferenceAccess::dummy(equation.getLocation(), rhsPackedType[i]));
      }

      equation.setLhsExpression(
          Expression::tuple(lhs->getLocation(), Type(PackedType(destinationsTypes)), newDestinations));
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool TypeInferencePass::run(ForEquation& forEquation)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    SymbolTable::ScopeTy scope(symbolTable);

    for (auto& induction : forEquation.getInductions()) {
      symbolTable.insert(induction->getName(), Symbol(*induction));

      if (!run<Expression>(*induction->getBegin())) {
        return false;
      }

      if (!run<Expression>(*induction->getEnd())) {
        return false;
      }
    }

    if (!run(*forEquation.getEquation())) {
      return false;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool TypeInferencePass::run(Induction& induction)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    if (!run<Expression>(*induction.getBegin())) {
      return false;
    }

    if (!run<Expression>(*induction.getBegin())) {
      return true;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool TypeInferencePass::run(Member& member)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto& type = member.getType();

    if (type.isa<UserDefinedType>()) {
      // Transform the user-defined type to a reference to the record definition
      const auto& userDefinedType = type.get<UserDefinedType>();
      auto name = userDefinedType.getName();

      auto symbol = symbolTable.lookup(name);

      if (symbol.isa<Class>() && symbol.get<Class>()->isa<Record>()) {
        member.setType(Type(symbol.get<Class>()->get<Record>(), type.getDimensions()));
      } else {
        diagnostics()->emitError<InvalidTypeMessage>(member.getLocation(), name);
        return numOfErrors == diagnostics()->numOfErrors();
      }
    }

    for (auto& dimension : type.getDimensions()) {
      if (dimension.hasExpression()) {
        if (!run<Expression>(*dimension.getExpression())) {
          return false;
        }
      }
    }

    if (member.hasModification()) {
      if (!run(*member.getModification())) {
        return false;
      }
    }

    // Ensure that the 'start' value is compatible with the member type
    if (member.hasStartExpression()) {
      auto* startExpression = member.getStartExpression();
      auto memberType = member.getType();
      auto startValueType = startExpression->getType();

      if (memberType.isa<BuiltInType>() && startValueType.isa<BuiltInType>()) {
        auto mostGenericType = *getMostGenericBuiltInType(
            memberType.get<BuiltInType>(),
                startValueType.get<BuiltInType>());

        startExpression->setType(startValueType.to(mostGenericType));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<Statement>(Statement& statement)
  {
    return statement.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(statement);
    });
  }

  bool TypeInferencePass::run(Algorithm& algorithm)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    for (auto& statement : algorithm.getBody()) {
      if (!run<Statement>(*statement)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<AssignmentStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* assignmentStatement = statement.get<AssignmentStatement>();

    auto* destinations = assignmentStatement->getDestinations();
    auto* destinationsTuple = destinations->get<Tuple>();
    auto* expression = assignmentStatement->getExpression();

    if (!run<Expression>(*destinations)) {
      return false;
    }

    if (!run<Expression>(*expression)) {
      return false;
    }

    if (destinationsTuple->size() > 1 && !expression->getType().isa<PackedType>()) {
      diagnostics()->emitError<BadSemanticMessage>(
          expression->getLocation(),
          "the expression must return at least " + std::to_string(destinationsTuple->size()) + "values");

      return numOfErrors == diagnostics()->numOfErrors();
    }

    // Assign type to dummy variables.
    // The assignment can't be done earlier because the expression type would
    // have not been evaluated yet.

    for (size_t i = 0, e = destinationsTuple->size(); i < e; ++i) {
      // If it's not a direct reference access, there's no way it can be a
      // dummy variable.
      if (!destinationsTuple->getArg(i)->isa<ReferenceAccess>()) {
        continue;
      }

      auto* reference = destinationsTuple->getArg(i)->get<ReferenceAccess>();

      if (reference->isDummy()) {
        auto& expressionType = expression->getType();
        auto& packedType = expressionType.get<PackedType>();
        assert(packedType.size() >= i);
        destinationsTuple->getArg(i)->setType(packedType[i]);
      }
    }

    // If the function call has more return values than the provided
    // destinations, then we need to add more dummy references.

    if (expression->getType().isa<PackedType>()) {
      auto& packedType = expression->getType().get<PackedType>();
      size_t returns = packedType.size();

      if (destinationsTuple->size() < returns) {
        llvm::SmallVector<std::unique_ptr<Expression>, 3> newDestinations;
        llvm::SmallVector<Type, 3> destinationsTypes;

        for (auto& destination : *destinationsTuple) {
          destinationsTypes.push_back(destination->getType());
          newDestinations.push_back(std::move(destination));
        }

        for (size_t i = newDestinations.size(); i < returns; ++i) {
          destinationsTypes.push_back(packedType[i]);
          newDestinations.emplace_back(ReferenceAccess::dummy(statement.getLocation(), packedType[i]));
        }

        assignmentStatement->setDestinations(
            Expression::tuple(destinations->getLocation(), Type(PackedType(destinationsTypes)), std::move(newDestinations)));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<BreakStatement>(Statement& statement)
  {
    return true;
  }

  template<>
  bool TypeInferencePass::run<ForStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    SymbolTable::ScopeTy scope(symbolTable);
    auto* forStatement = statement.get<ForStatement>();
    auto* induction = forStatement->getInduction();

    if (!run(*forStatement->getInduction())) {
      return false;
    }

    symbolTable.insert(induction->getName(), Symbol(*induction));

    for (auto& stmnt : forStatement->getBody()) {
      if (!run<Statement>(*stmnt)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<IfStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* ifStatement = statement.get<IfStatement>();

    for (auto& block : *ifStatement) {
      if (!run<Expression>(*block.getCondition())) {
        return false;
      }

      for (auto& stmnt : block) {
        if (!run<Statement>(*stmnt)) {
          return false;
        }
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<ReturnStatement>(Statement& statement)
  {
    return true;
  }

  template<>
  bool TypeInferencePass::run<WhenStatement>(Statement& statement)
  {
    llvm_unreachable("Not implemented");
    return false;
  }

  template<>
  bool TypeInferencePass::run<WhileStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* whileStatement = statement.get<WhileStatement>();

    if (!run<Expression>(*whileStatement->getCondition())) {
      return false;
    }

    for (auto& stmnt : whileStatement->getBody()) {
      if (!run<Statement>(*stmnt)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool TypeInferencePass::run(Modification& modification)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    if (modification.hasClassModification()) {
      if (!run(*modification.getClassModification())) {
        return false;
      }
    }

    if (modification.hasExpression()) {
      if (!run<Expression>(*modification.getExpression())) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<ElementModification>(Argument& argument)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* elementModification = argument.get<ElementModification>();

    if (elementModification->hasModification()) {
      if (!run(*elementModification->getModification())) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeInferencePass::run<ElementRedeclaration>(Argument& argument)
  {
    llvm_unreachable("Not implemented");
    return false;
  }

  template<>
  bool TypeInferencePass::run<ElementReplaceable>(Argument& argument)
  {
    llvm_unreachable("Not implemented");
    return false;
  }

  template<>
  bool TypeInferencePass::run<Argument>(Argument& argument)
  {
    return argument.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(argument);
    });
  }

  bool TypeInferencePass::run(ClassModification& classModification)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    for (auto& argument : classModification) {
      if (!run<Argument>(*argument)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  std::unique_ptr<Pass> createTypeInferencePass(diagnostic::DiagnosticEngine& diagnostics)
  {
    return std::make_unique<TypeInferencePass>(diagnostics);
  }
}
