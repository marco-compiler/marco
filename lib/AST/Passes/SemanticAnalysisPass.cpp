#include "marco/AST/Passes/SemanticAnalysisPass.h"
#include "marco/AST/Analysis/DynamicDimensionsGraph.h"
#include "marco/Diagnostic/Printer.h"

using namespace ::marco;
using namespace ::marco::ast;

//===----------------------------------------------------------------------===//
// Messages
//===----------------------------------------------------------------------===//

namespace
{
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

  class DifferentTypesComparisonMessage : public diagnostic::SourceMessage
  {
    public:
      DifferentTypesComparisonMessage(SourceRange location)
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

        os << "comparison of values with different types";
        os << "\n";

        printLines(os, highlightSourceFn);
      }
  };
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace marco::ast
{
  SemanticAnalysisPass::SemanticAnalysisPass(diagnostic::DiagnosticEngine& diagnostics)
    : Pass(diagnostics)
  {
  }

  SemanticAnalysisPass::SymbolTable& SemanticAnalysisPass::getSymbolTable()
  {
    return symbolTable;
  }

  bool SemanticAnalysisPass::run(std::unique_ptr<Class>& cls)
  {
    return run<Class>(*cls);
  }

  template<>
  bool SemanticAnalysisPass::run<PartialDerFunction>(Class& cls)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::run<StandardFunction>(Class& cls)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    SymbolTable::ScopeTy scope(symbolTable);
    auto* function = cls.get<StandardFunction>();

    // Check that the expression for dynamic dimensions do not create cycles.
    DynamicDimensionsGraph dynamicDimensionsGraph;

    dynamicDimensionsGraph.addMembersGroup(function->getArgs(), true);
    dynamicDimensionsGraph.addMembersGroup(function->getResults(), true);

    dynamicDimensionsGraph.addMembersGroup(
        function->getProtectedMembers(), false);

    if (dynamicDimensionsGraph.hasCycles()) {
      diagnostics()->emitError<BadSemanticMessage>(
          function->getLocation(),
          "cycles detected among the dynamic dimensions of the variables");

      return false;
    }

    // Populate the symbol table
    symbolTable.insert(function->getName(), Symbol(cls));

    for (auto& member : function->getMembers()) {
      symbolTable.insert(member->getName(), Symbol(*member));
    }

    // Check members
    for (auto& member : function->getMembers()) {
      if (!run(*member)) {
        return false;
      }
    }

    for (auto& member : function->getMembers()) {
      if (!run(*member)) {
        return false;
      }
    }

    for (auto& member : function->getMembers()) {
      // From Function reference:
      // "Each input formal parameter of the function must be prefixed by the
      // keyword input, and each result formal parameter by the keyword output.
      // All public variables are formal parameters."

      if (member->isPublic() && !member->isInput() && !member->isOutput()) {
        diagnostics()->emitError<BadSemanticMessage>(
            member->getLocation(),
            "public members of functions must be input or output variables");
      }

      // From Function reference:
      // "Input formal parameters are read-only after being bound to the actual
      // arguments or default values, i.e., they may not be assigned values in
      // the body of the function."

      if (member->isInput() && member->hasExpression() && !function->isCustomRecordConstructor()) {
        diagnostics()->emitError<BadSemanticMessage>(
            member->getLocation(),
            "input member can't receive a new value");
      }
    }

    // Check body
    auto algorithms = function->getAlgorithms();

    // From Function reference:
    // "A function can have at most one algorithm section or one external
    // function interface (not both), which, if present, is the body of the
    // function."

    if (algorithms.size() > 1) {
      diagnostics()->emitError<BadSemanticMessage>(
          function->getLocation(),
          "functions can have at most one 'algorithm' section");
    }

    for (const auto& algorithm : algorithms) {
      for (const auto& statement : *algorithm) {
        for (const auto& assignment : *statement) {
          for (const auto& exp : *assignment.getDestinations()->get<Tuple>()) {
            // From Function reference:
            // "Input formal parameters are read-only after being bound to the
            // actual arguments or default values, i.e., they may not be assigned
            // values in the body of the function."
            const auto* current = exp.get();

            while (current->isa<Operation>()) {
              const auto* operation = current->get<Operation>();
              assert(operation->getOperationKind() == OperationKind::subscription);
              current = operation->getArg(0);
            }

            assert(current->isa<ReferenceAccess>());
            const auto* ref = current->get<ReferenceAccess>();

            if (!ref->isDummy()) {
              const auto& name = ref->getName();

              if (symbolTable.count(name) == 0) {
                diagnostics()->emitError<UnknownSymbolMessage>(ref->getLocation(), name);
              }

              const auto& member = symbolTable.lookup(name).get<Member>();

              if (member->isInput()) {
                diagnostics()->emitError<BadSemanticMessage>(
                    ref->getLocation(),
                    "input members can't receive a new value");
              }
            }
          }

          // From Function reference:
          // "A function cannot contain calls to the Modelica built-in operators
          // der, initial, terminal, sample, pre, edge, change, reinit, delay,
          // cardinality, inStream, actualStream, to the operators of the built-in
          // package Connections, and is not allowed to contain when-statements."

          std::stack<const Expression*> stack;
          stack.push(assignment.getExpression());

          while (!stack.empty()) {
            const auto* expression = stack.top();
            stack.pop();

            if (expression->isa<ReferenceAccess>()) {
              llvm::StringRef name = expression->get<ReferenceAccess>()->getName();

              if (name == "der" || name == "initial" || name == "terminal" || name == "sample" || name == "pre" || name == "edge" || name == "change" || name == "reinit" || name == "delay" || name == "cardinality" || name == "inStream" || name == "actualStream") {
                diagnostics()->emitError<BadSemanticMessage>(
                    expression->getLocation(),
                    "'" + name.str() + "' is not allowed in procedural code");
              }

              // TODO: Connections built-in operators + when statement
            } else if (expression->isa<Operation>()) {
              for (const auto& arg : *expression->get<Operation>()) {
                stack.push(arg.get());
              }
            } else if (expression->isa<Call>()) {
              const auto* call = expression->get<Call>();

              for (const auto& arg : *call) {
                stack.push(arg.get());
              }

              stack.push(call->getFunction());
            }
          }
        }
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool SemanticAnalysisPass::run<Model>(Class& cls)
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

    // Process the members
    for (auto& m : model->getMembers()) {
      if (!run(*m)) {
        return false;
      }
    }

    // Process the body
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

    // Process the inner classes
    for (auto& innerClass : model->getInnerClasses()) {
      if (!run<Class>(*innerClass)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool SemanticAnalysisPass::run<Package>(Class& cls)
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
  bool SemanticAnalysisPass::run<Record>(Class& cls)
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
  bool SemanticAnalysisPass::run<Class>(Class& cls)
  {
    return cls.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(cls);
    });
  }

  template<>
  bool SemanticAnalysisPass::run<Array>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::run<Call>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::run<Constant>(Expression& expression)
  {
    return true;
  }

  bool SemanticAnalysisPass::processComparisonOp(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (auto lhsType = lhs->getType(); !lhsType.isScalar() || !lhsType.isNumeric()) {
      diagnostics()->emitError<BadSemanticMessage>(
          lhs->getLocation(),
          "compared arguments must be scalar numeric values");

      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (auto rhsType = rhs->getType(); !rhsType.isScalar() || !rhsType.isNumeric()) {
      diagnostics()->emitError<BadSemanticMessage>(
          rhs->getLocation(),
          "compared arguments must be scalar numeric values");

      return numOfErrors == diagnostics()->numOfErrors();
    }

    if (lhs->getType() != rhs->getType()) {
      diagnostics()->emitWarning<DifferentTypesComparisonMessage>(expression.getLocation());
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::add>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::addEW>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::different>(Expression& expression)
  {
    return processComparisonOp(expression);
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::divide>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::divideEW>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::equal>(Expression& expression)
  {
    return processComparisonOp(expression);
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::greater>(Expression& expression)
  {
    return processComparisonOp(expression);
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::greaterEqual>(Expression& expression)
  {
    return processComparisonOp(expression);
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::ifelse>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::land>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::lnot>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::lor>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::less>(Expression& expression)
  {
    return processComparisonOp(expression);
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::lessEqual>(Expression& expression)
  {
    return processComparisonOp(expression);
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::memberLookup>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::multiply>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::multiplyEW>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::negate>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::powerOf>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::powerOfEW>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::range>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::subscription>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::subtract>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::processOp<OperationKind::subtractEW>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::run<Operation>(Expression& expression)
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
  bool SemanticAnalysisPass::run<ReferenceAccess>(Expression& expression)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::run<Tuple>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* tuple = expression.get<Tuple>();

    for (auto& exp : *tuple) {
      if (!run<Expression>(*exp)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool SemanticAnalysisPass::run<Expression>(Expression& expression)
  {
    return expression.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(expression);
    });
  }

  bool SemanticAnalysisPass::run(Equation& equation)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    if (!run<Expression>(*equation.getLhsExpression())) {
      return false;
    }

    if (!run<Expression>(*equation.getRhsExpression())) {
      return false;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool SemanticAnalysisPass::run(ForEquation& forEquation)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    SymbolTable::ScopeTy scope(symbolTable);

    for (auto& ind : forEquation.getInductions()) {
      symbolTable.insert(ind->getName(), Symbol(*ind));

      if (!run<Expression>(*ind->getBegin())) {
        return false;
      }

      if (!run<Expression>(*ind->getEnd())) {
        return false;
      }
    }

    if (!run(*forEquation.getEquation())) {
      return false;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool SemanticAnalysisPass::run(Induction& induction)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    if (!run<Expression>(*induction.getBegin())) {
      return false;
    }

    if (!run<Expression>(*induction.getEnd())) {
      return true;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool SemanticAnalysisPass::run(Member& member)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto &type = member.getType();

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

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool SemanticAnalysisPass::run<Statement>(Statement& statement)
  {
    return statement.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(statement);
    });
  }

  template<>
  bool SemanticAnalysisPass::run<AssignmentStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* assignmentStatement = statement.get<AssignmentStatement>();

    auto* destinations = assignmentStatement->getDestinations();
    auto* expression = assignmentStatement->getExpression();

    if (!run<Expression>(*destinations)) {
      return false;
    }

    if (!run<Expression>(*expression)) {
      return false;
    }

    for (auto& destination : *destinations->get<Tuple>()) {
      // The destinations must be l-values.
      // The check can't be enforced at parsing time because the grammar
      // specifies the destinations as expressions.

      if (!destination->isLValue()) {
        diagnostics()->emitError<BadSemanticMessage>(
            destination->getLocation(),
            "destinations of statements must be l-values");
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool SemanticAnalysisPass::run<BreakStatement>(Statement& statement)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::run<ForStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* forStatement = statement.get<ForStatement>();

    if (!run(*forStatement->getInduction())) {
      return false;
    }

    for (auto& stmnt : forStatement->getBody()) {
      if (!run<Statement>(*stmnt)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool SemanticAnalysisPass::run<IfStatement>(Statement& statement)
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
  bool SemanticAnalysisPass::run<ReturnStatement>(Statement& statement)
  {
    return true;
  }

  template<>
  bool SemanticAnalysisPass::run<WhenStatement>(Statement& statement)
  {
    llvm_unreachable("Not implemented");
    return false;
  }

  template<>
  bool SemanticAnalysisPass::run<WhileStatement>(Statement& statement)
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

  bool SemanticAnalysisPass::run(Algorithm& algorithm)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    if (algorithm.empty()) {
      diagnostics()->emitWarning<BadSemanticMessage>(
          algorithm.getLocation(),
          "empty algorithm");
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool SemanticAnalysisPass::run(Modification& modification)
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
  bool SemanticAnalysisPass::run<ElementModification>(Argument& argument)
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
  bool SemanticAnalysisPass::run<ElementRedeclaration>(Argument& argument)
  {
    llvm_unreachable("Not implemented");
    return false;
  }

  template<>
  bool SemanticAnalysisPass::run<ElementReplaceable>(Argument& argument)
  {
    llvm_unreachable("Not implemented");
    return false;
  }

  template<>
  bool SemanticAnalysisPass::run<Argument>(Argument& argument)
  {
    return argument.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(argument);
    });
  }

  bool SemanticAnalysisPass::run(ClassModification& classModification)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    for (auto& argument : classModification) {
      if (!run<Argument>(*argument)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  std::unique_ptr<Pass> createSemanticAnalysisPass(diagnostic::DiagnosticEngine& diagnostics)
  {
    return std::make_unique<SemanticAnalysisPass>(diagnostics);
  }
}
