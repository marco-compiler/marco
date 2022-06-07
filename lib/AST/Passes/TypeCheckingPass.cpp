#include "marco/AST/Passes/TypeCheckingPass.h"
#include "marco/Diagnostic/Printer.h"

using namespace ::marco;
using namespace ::marco::ast;

static bool canBeCastTo(const Type& source, const Type& destination)
{
  if (!source.isa<BuiltInType>() || !destination.isa<BuiltInType>()) {
    return false;
  }

  if (!source.isNumeric() || !destination.isNumeric()) {
    return false;
  }

  // Check if the shapes are compatible
  if (source.getRank() != destination.getRank()) {
    return false;
  }

  auto sourceDimensions = source.getDimensions();
  auto destinationDimensions = destination.getDimensions();

  for (size_t i = 0; i < source.getRank(); ++i) {
    if (!sourceDimensions[i].isDynamic() &&
        !destinationDimensions[i].isDynamic() &&
        sourceDimensions[i].getNumericSize() != destinationDimensions[i].getNumericSize()) {
      return false;
    }
  }

  return true;
}

static bool isLosslessCast(const Type& source, const Type& destination)
{
  assert(source.isNumeric() && destination.isNumeric());

  if (source.isa<float>()) {
    return destination.isa<float>();
  }

  if (source.isa<int>()) {
    return !destination.isa<bool>();
  }

  return true;
}

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

  class ImplicitCastMessage : public diagnostic::SourceMessage
  {
    public:
      ImplicitCastMessage(SourceRange location, Type source, Type destination)
        : SourceMessage(std::move(location)),
          source(std::move(source)),
          destination(std::move(destination))
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
        os << "implicit cast from '";
        printer->setBold(os);
        os << source;
        printer->unsetBold(os);
        os << "' to '";
        printer->setBold(os);
        os << destination;
        printer->unsetBold(os);
        os << "'";
        os << "\n";

        printLines(os, highlightSourceFn);
      }

    private:
      Type source;
      Type destination;
  };
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace marco::ast
{
  TypeCheckingPass::TypeCheckingPass(diagnostic::DiagnosticEngine& diagnostics)
    : Pass(diagnostics)
  {
  }

  TypeCheckingPass::SymbolTable& TypeCheckingPass::getSymbolTable()
  {
    return symbolTable;
  }

  template<>
  bool TypeCheckingPass::run<Class>(Class& cls)
  {
    return cls.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(cls);
    });
  }

  bool TypeCheckingPass::run(std::unique_ptr<Class>& cls)
  {
    return run<Class>(*cls);
  }

  template<>
  bool TypeCheckingPass::run<PartialDerFunction>(Class& cls)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::run<StandardFunction>(Class& cls)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    SymbolTable::ScopeTy scope(symbolTable);
    auto* function = cls.get<StandardFunction>();

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

    // Check body
    for (auto& algorithm : function->getAlgorithms()) {
      if (!run(*algorithm)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeCheckingPass::run<Model>(Class& cls)
  {
    auto* model = cls.get<Model>();

    // Functions type checking must be done before the equations or algorithm
    // ones, because it establishes the result type of the functions that may
    // be invoked elsewhere.
    for (auto& innerClass : model->getInnerClasses()) {
      if (!run<Class>(*innerClass)) {
        return false;
      }
    }

    for (auto& m : model->getMembers()) {
      if (!run(*m)) {
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

    return true;
  }

  template<>
  bool TypeCheckingPass::run<Package>(Class& cls)
  {
    auto* package = cls.get<Package>();

    for (auto& innerClass : *package) {
      if (!run<Class>(*innerClass)) {
        return false;
      }
    }

    return true;
  }

  template<>
  bool TypeCheckingPass::run<Record>(Class& cls)
  {
    auto* record = cls.get<Record>();

    for (auto& member : *record) {
      if (!run(*member)) {
        return false;
      }
    }

    return true;
  }

  template<>
  bool TypeCheckingPass::run<Expression>(Expression& expression)
  {
    return expression.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(expression);
    });
  }

  template<>
  bool TypeCheckingPass::run<Array>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* array = expression.get<Array>();

    for (auto& element : *array) {
      if (!run<Expression>(*element)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeCheckingPass::run<Call>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* call = expression.get<Call>();

    if (!run<Expression>(*call->getFunction())) {
      return false;
    }

    for (auto& arg : *call) {
      if (!run<Expression>(*arg)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeCheckingPass::run<Constant>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::add>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::addEW>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::different>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::divide>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::divideEW>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::equal>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::greater>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::greaterEqual>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::ifelse>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::land>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::lnot>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::lor>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::less>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::lessEqual>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::memberLookup>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::multiply>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::multiplyEW>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::negate>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::powerOf>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::powerOfEW>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::range>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::subscription>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::subtract>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::processOp<OperationKind::subtractEW>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::run<Operation>(Expression& expression)
  {
    auto* operation = expression.get<Operation>();

    // Process the arguments
    for (size_t i = 0; i < operation->argumentsCount(); ++i) {
      if (!run<Expression>(*operation->getArg(i))) {
        return false;
      }
    }

    // Check the operation-specific semantics
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
        return processOp<OperationKind::land>(expression);

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
  bool TypeCheckingPass::run<ReferenceAccess>(Expression& expression)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::run<Tuple>(Expression& expression)
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
  bool TypeCheckingPass::run<RecordInstance>(Expression& expression)
  {
    return true;
  }

  bool TypeCheckingPass::run(Equation& equation)
  {
    if (!run<Expression>(*equation.getLhsExpression())) {
      return false;
    }

    if (!run<Expression>(*equation.getRhsExpression())) {
      return false;
    }

    return true;
  }

  bool TypeCheckingPass::run(ForEquation& forEquation)
  {
    for (auto& induction : forEquation.getInductions()) {
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

    return true;
  }

  bool TypeCheckingPass::run(Induction& induction)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* beginExp = induction.getBegin();

    if (!run<Expression>(*beginExp)) {
      return false;
    }

    if (const auto& type = beginExp->getType(); !type.isa<int>()) {
      if (canBeCastTo(type, Type(BuiltInType::Integer))) {
        if (!isLosslessCast(type, type.to(BuiltInType::Integer))) {
          diagnostics()->emitWarning<ImplicitCastMessage>(beginExp->getLocation(), type, Type(BuiltInType::Integer));
        }
      } else {
        diagnostics()->emitError<BadSemanticMessage>(beginExp->getLocation(), "the begin value must be convertible to a scalar integer");
      }
    }

    auto* endExp = induction.getEnd();

    if (!run<Expression>(*endExp)) {
      return false;
    }

    if (const auto& type = endExp->getType(); !type.isa<int>()) {
      if (canBeCastTo(type, Type(BuiltInType::Integer))) {
        if (!isLosslessCast(type, type.to(BuiltInType::Integer))) {
          diagnostics()->emitWarning<ImplicitCastMessage>(endExp->getLocation(), type, Type(BuiltInType::Integer));
        }
      } else {
        diagnostics()->emitError<BadSemanticMessage>(endExp->getLocation(), "the end value must be convertible to a scalar integer");
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool TypeCheckingPass::run(Member& member)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    if (member.hasInitializer()) {
      if (!run<Expression>(*member.getInitializer())) {
        return false;
      }
    }

    if (member.hasStartOverload()) {
      if (!run<Expression>(*member.getStartOverload())) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeCheckingPass::run<Statement>(Statement& statement)
  {
    return statement.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(statement);
    });
  }

  bool TypeCheckingPass::run(Algorithm& algorithm)
  {
    for (auto& statement : algorithm.getBody()) {
      if (!run<Statement>(*statement)) {
        return false;
      }
    }

    return true;
  }

  template<>
  bool TypeCheckingPass::run<AssignmentStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* assignmentStatement = statement.get<AssignmentStatement>();

    if (!run<Expression>(*assignmentStatement->getDestinations())) {
      return false;
    }

    if (!run<Expression>(*assignmentStatement->getExpression())) {
      return false;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool TypeCheckingPass::run<BreakStatement>(Statement& statement)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::run<ForStatement>(Statement& statement)
  {
    auto* forStatement = statement.get<ForStatement>();

    if (!run(*forStatement->getInduction())) {
      return false;
    }

    for (auto& stmnt : forStatement->getBody()) {
      if (!run<Statement>(*stmnt)) {
        return false;
      }
    }

    return true;
  }

  template<>
  bool TypeCheckingPass::run<IfStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* ifStatement = statement.get<IfStatement>();

    for (auto& block : *ifStatement) {
      if (!run<Expression>(*block.getCondition())) {
        return false;
      }

      auto* condition = block.getCondition();

      if (const auto& type = condition->getType(); !type.isa<bool>()) {
        if (canBeCastTo(type, Type(BuiltInType::Boolean))) {
          if (!isLosslessCast(type, type.to(BuiltInType::Boolean))) {
            diagnostics()->emitWarning<ImplicitCastMessage>(condition->getLocation(), type, Type(BuiltInType::Boolean));
          }
        } else {
          diagnostics()->emitError<BadSemanticMessage>(condition->getLocation(), "the condition must be a scalar numeric value");
        }
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
  bool TypeCheckingPass::run<ReturnStatement>(Statement& statement)
  {
    return true;
  }

  template<>
  bool TypeCheckingPass::run<WhenStatement>(Statement& statement)
  {
    llvm_unreachable("Not implemented");
    return false;
  }

  template<>
  bool TypeCheckingPass::run<WhileStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* whileStatement = statement.get<WhileStatement>();

    auto* condition = whileStatement->getCondition();

    if (!run<Expression>(*condition)) {
      return false;
    }

    if (const auto& type = condition->getType(); !type.isa<bool>()) {
      if (canBeCastTo(type, Type(BuiltInType::Boolean))) {
        if (!isLosslessCast(type, type.to(BuiltInType::Boolean))) {
          diagnostics()->emitWarning<ImplicitCastMessage>(condition->getLocation(), type, Type(BuiltInType::Boolean));
        }
      } else {
        diagnostics()->emitError<BadSemanticMessage>(condition->getLocation(), "the condition must be a scalar numeric value");
      }
    }

    for (auto& stmnt : whileStatement->getBody()) {
      if (!run<Statement>(*stmnt)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  std::unique_ptr<Pass> createTypeCheckingPass(diagnostic::DiagnosticEngine& diagnostics)
  {
    return std::make_unique<TypeCheckingPass>(diagnostics);
  }
}
