#include "marco/Codegen/Lowering/Bridge.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/LoweringContext.h"
#include "marco/Codegen/Lowering/AlgorithmLowerer.h"
#include "marco/Codegen/Lowering/ArrayGeneratorLowerer.h"
#include "marco/Codegen/Lowering/AssignmentStatementLowerer.h"
#include "marco/Codegen/Lowering/BreakStatementLowerer.h"
#include "marco/Codegen/Lowering/CallLowerer.h"
#include "marco/Codegen/Lowering/ClassLowerer.h"
#include "marco/Codegen/Lowering/ComponentReferenceLowerer.h"
#include "marco/Codegen/Lowering/ConstantLowerer.h"
#include "marco/Codegen/Lowering/EquationLowerer.h"
#include "marco/Codegen/Lowering/EqualityEquationLowerer.h"
#include "marco/Codegen/Lowering/EquationSectionLowerer.h"
#include "marco/Codegen/Lowering/ExpressionLowerer.h"
#include "marco/Codegen/Lowering/ForEquationLowerer.h"
#include "marco/Codegen/Lowering/ForStatementLowerer.h"
#include "marco/Codegen/Lowering/IfEquationLowerer.h"
#include "marco/Codegen/Lowering/IfStatementLowerer.h"
#include "marco/Codegen/Lowering/ModelLowerer.h"
#include "marco/Codegen/Lowering/OperationLowerer.h"
#include "marco/Codegen/Lowering/PackageLowerer.h"
#include "marco/Codegen/Lowering/PartialDerFunctionLowerer.h"
#include "marco/Codegen/Lowering/RecordLowerer.h"
#include "marco/Codegen/Lowering/ReturnStatementLowerer.h"
#include "marco/Codegen/Lowering/StandardFunctionLowerer.h"
#include "marco/Codegen/Lowering/StatementLowerer.h"
#include "marco/Codegen/Lowering/SubscriptLowerer.h"
#include "marco/Codegen/Lowering/TupleLowerer.h"
#include "marco/Codegen/Lowering/WhenEquationLowerer.h"
#include "marco/Codegen/Lowering/WhenStatementLowerer.h"
#include "marco/Codegen/Lowering/WhileStatementLowerer.h"
#include <memory>

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering
{
  BridgeInterface::~BridgeInterface() = default;

  class Bridge::Impl : public BridgeInterface
  {
    public:
      Impl(mlir::MLIRContext& context, clang::DiagnosticsEngine &diag);

      ~Impl() override;

      LoweringContext& getContext() override;

      const LoweringContext& getContext() const override;

      mlir::Operation* getRoot() const override;

      std::unique_ptr<mlir::ModuleOp>& getMLIRModule();

      [[nodiscard]] bool convert(const ast::Root& root);

      void declare(const ast::Class& node) override;

      void declare(const ast::Model& node) override;

      void declare(const ast::Package& node) override;

      void declare(const ast::PartialDerFunction& node) override;

      void declare(const ast::Record& node) override;

      void declare(const ast::StandardFunction& node) override;

      [[nodiscard]] bool declareVariables(const ast::Class& node) override;

      [[nodiscard]] bool declareVariables(const ast::Model& model) override;

      [[nodiscard]] bool declareVariables(const ast::Package& package) override;

      void declareVariables(
          const ast::PartialDerFunction& function) override;

      [[nodiscard]] bool declareVariables(const ast::Record& record) override;

      [[nodiscard]] bool declareVariables(
          const ast::StandardFunction& function) override;

      [[nodiscard]] bool declare(const ast::Member& node) override;

      [[nodiscard]] bool lower(const ast::Class& node) override;

      [[nodiscard]] bool lower(const ast::Model& node) override;

      [[nodiscard]] bool lower(const ast::Package& node) override;

      void lower(const ast::PartialDerFunction& node) override;

      [[nodiscard]] bool lower(const ast::Record& node) override;

      [[nodiscard]] bool lower(const ast::StandardFunction& node) override;

      [[nodiscard]] bool lowerClassBody(const ast::Class& node) override;

      [[nodiscard]] bool createBindingEquation(
          const ast::Member& variable,
          const ast::Expression& expression) override;

      [[nodiscard]] bool lowerStartAttribute(
          mlir::SymbolRefAttr variable,
          const ast::Expression& expression,
          bool fixed,
          bool each) override;

      std::optional<Results> lower(const ast::Expression& expression) override;

      std::optional<Results> lower(const ast::ArrayGenerator& node) override;

      std::optional<Results> lower(const ast::Call& node) override;

      Results lower(const ast::Constant& constant) override;

      std::optional<Results> lower(const ast::Operation& operation) override;

      std::optional<Results> lower(
          const ast::ComponentReference& componentReference) override;

      std::optional<Results> lower(const ast::Tuple& tuple) override;

      std::optional<Results> lower(const ast::Subscript& subscript) override;

      [[nodiscard]] bool lower(const ast::EquationSection& node) override;

      [[nodiscard]] bool lower(const ast::Equation& node) override;

      [[nodiscard]] bool lower(const ast::EqualityEquation& node) override;

      void lower(const ast::IfEquation& node) override;

      [[nodiscard]] bool lower(const ast::ForEquation& node) override;

      void lower(const ast::WhenEquation& node) override;

      [[nodiscard]] bool lower(const ast::Algorithm& node) override;

      [[nodiscard]] bool lower(const ast::Statement& node) override;

      [[nodiscard]] bool lower(const ast::AssignmentStatement& statement) override;

      void lower(const ast::BreakStatement& statement) override;

      [[nodiscard]] bool lower(const ast::ForStatement& statement) override;

      [[nodiscard]] bool lower(const ast::IfStatement& statement) override;

      void lower(const ast::ReturnStatement& statement) override;

      void lower(const ast::WhenStatement& statement) override;

      [[nodiscard]] bool lower(const ast::WhileStatement& statement) override;

      void emitIdentifierError(IdentifierError::IdentifierType identifierType, 
                               std::string name, const std::set<std::string> &declaredIdentifiers,
                               unsigned int line, unsigned int column) override;
      void emitError(const std::string &error) override;

    private:
      std::unique_ptr<LoweringContext> context;

      // The module that is populated while converting the AST.
      std::unique_ptr<mlir::ModuleOp> module;

      clang::DiagnosticsEngine *diag;

      // Lowerers.
      std::unique_ptr<ClassLowerer> classLowerer;
      std::unique_ptr<ModelLowerer> modelLowerer;
      std::unique_ptr<PackageLowerer> packageLowerer;
      std::unique_ptr<RecordLowerer> recordLowerer;
      std::unique_ptr<PartialDerFunctionLowerer> partialDerFunctionLowerer;
      std::unique_ptr<StandardFunctionLowerer> standardFunctionLowerer;
      std::unique_ptr<ExpressionLowerer> expressionLowerer;
      std::unique_ptr<ArrayGeneratorLowerer> arrayLowerer;
      std::unique_ptr<CallLowerer> callLowerer;
      std::unique_ptr<ConstantLowerer> constantLowerer;
      std::unique_ptr<OperationLowerer> operationLowerer;
      std::unique_ptr<ComponentReferenceLowerer> componentReferenceLowerer;
      std::unique_ptr<TupleLowerer> tupleLowerer;
      std::unique_ptr<SubscriptLowerer> subscriptLowerer;
      std::unique_ptr<EquationSectionLowerer> equationSectionLowerer;
      std::unique_ptr<EquationLowerer> equationLowerer;
      std::unique_ptr<EqualityEquationLowerer> equalityEquationLowerer;
      std::unique_ptr<ForEquationLowerer> forEquationLowerer;
      std::unique_ptr<IfEquationLowerer> ifEquationLowerer;
      std::unique_ptr<WhenEquationLowerer> whenEquationLowerer;
      std::unique_ptr<AlgorithmLowerer> algorithmLowerer;
      std::unique_ptr<StatementLowerer> statementLowerer;
      std::unique_ptr<AssignmentStatementLowerer> assignmentStatementLowerer;
      std::unique_ptr<BreakStatementLowerer> breakStatementLowerer;
      std::unique_ptr<ForStatementLowerer> forStatementLowerer;
      std::unique_ptr<IfStatementLowerer> ifStatementLowerer;
      std::unique_ptr<ReturnStatementLowerer> returnStatementLowerer;
      std::unique_ptr<WhenStatementLowerer> whenStatementLowerer;
      std::unique_ptr<WhileStatementLowerer> whileStatementLowerer;
  };

  Bridge::Impl::Impl(mlir::MLIRContext& context, clang::DiagnosticsEngine &diag)
  {
    this->context = std::make_unique<LoweringContext>(context);

    this->module = std::make_unique<mlir::ModuleOp>(
        mlir::ModuleOp::create(this->context->getBuilder().getUnknownLoc()));

    this->diag = &diag;

    // Initialize the lowerers.
    this->classLowerer = std::make_unique<ClassLowerer>(this);
    this->modelLowerer = std::make_unique<ModelLowerer>(this);
    this->packageLowerer = std::make_unique<PackageLowerer>(this);

    this->partialDerFunctionLowerer =
        std::make_unique<PartialDerFunctionLowerer>(this);

    this->recordLowerer = std::make_unique<RecordLowerer>(this);

    this->standardFunctionLowerer =
        std::make_unique<StandardFunctionLowerer>(this);

    this->expressionLowerer = std::make_unique<ExpressionLowerer>(this);
    this->arrayLowerer = std::make_unique<ArrayGeneratorLowerer>(this);
    this->callLowerer = std::make_unique<CallLowerer>(this);

    this->componentReferenceLowerer =
        std::make_unique<ComponentReferenceLowerer>(this);

    this->constantLowerer = std::make_unique<ConstantLowerer>(this);
    this->operationLowerer = std::make_unique<OperationLowerer>(this);
    this->tupleLowerer = std::make_unique<TupleLowerer>(this);
    this->subscriptLowerer = std::make_unique<SubscriptLowerer>(this);

    this->equationSectionLowerer =
        std::make_unique<EquationSectionLowerer>(this);

    this->equationLowerer = std::make_unique<EquationLowerer>(this);

    this->equalityEquationLowerer =
        std::make_unique<EqualityEquationLowerer>(this);

    this->forEquationLowerer = std::make_unique<ForEquationLowerer>(this);
    this->ifEquationLowerer = std::make_unique<IfEquationLowerer>(this);
    this->whenEquationLowerer = std::make_unique<WhenEquationLowerer>(this);

    this->algorithmLowerer = std::make_unique<AlgorithmLowerer>(this);
    this->statementLowerer = std::make_unique<StatementLowerer>(this);

    this->assignmentStatementLowerer =
        std::make_unique<AssignmentStatementLowerer>(this);

    this->breakStatementLowerer =
        std::make_unique<BreakStatementLowerer>(this);

    this->forStatementLowerer = std::make_unique<ForStatementLowerer>(this);
    this->ifStatementLowerer = std::make_unique<IfStatementLowerer>(this);

    this->returnStatementLowerer =
        std::make_unique<ReturnStatementLowerer>(this);

    this->whenStatementLowerer = std::make_unique<WhenStatementLowerer>(this);

    this->whileStatementLowerer =
        std::make_unique<WhileStatementLowerer>(this);
  }

  Bridge::Impl::~Impl()
  {
    if (module != nullptr) {
      module->erase();
    }
  }

  LoweringContext& Bridge::Impl::getContext()
  {
    assert(context != nullptr && "Lowering context not set");
    return *context;
  }

  const LoweringContext& Bridge::Impl::getContext() const
  {
    assert(context != nullptr && "Lowering context not set");
    return *context;
  }

  mlir::Operation* Bridge::Impl::getRoot() const
  {
    assert(module != nullptr && "MLIR module not created");
    return module->getOperation();
  }

  bool Bridge::Impl::convert(const ast::Root& root)
  {
    mlir::OpBuilder::InsertionGuard guard(context->getBuilder());
    context->pushLookupScope(module->getOperation());
    context->getBuilder().setInsertionPointToStart(module->getBody());

    for (const auto& cls : root.getInnerClasses()) {
      classLowerer->declare(*cls->cast<ast::Class>());
    }

    for (const auto& cls : root.getInnerClasses()) {
      const bool outcome = classLowerer->declareVariables(*cls->cast<ast::Class>());
      if (!outcome) {
        return false;
      }
    }

    for (const auto& cls : root.getInnerClasses()) {
      const bool outcome = classLowerer->lower(*cls->cast<ast::Class>());
      if (!outcome) {
        return false;
      }
    }

    return true;
  }

  std::unique_ptr<mlir::ModuleOp>& Bridge::Impl::getMLIRModule()
  {
    assert(module != nullptr);
    return module;
  }

  void Bridge::Impl::declare(const ast::Class& cls)
  {
    assert(classLowerer != nullptr);
    return classLowerer->declare(cls);
  }

  void Bridge::Impl::declare(const ast::Model& model)
  {
    assert(modelLowerer != nullptr);
    return modelLowerer->declare(model);
  }

  void Bridge::Impl::declare(const ast::Package& package)
  {
    assert(packageLowerer != nullptr);
    return packageLowerer->declare(package);
  }

  void Bridge::Impl::declare(const ast::PartialDerFunction& function)
  {
    assert(partialDerFunctionLowerer != nullptr);
    return partialDerFunctionLowerer->declare(function);
  }

  void Bridge::Impl::declare(const ast::Record& record)
  {
    assert(recordLowerer != nullptr);
    return recordLowerer->declare(record);
  }

  void Bridge::Impl::declare(const ast::StandardFunction& function)
  {
    assert(standardFunctionLowerer != nullptr);
    return standardFunctionLowerer->declare(function);
  }

  bool Bridge::Impl::declareVariables(const ast::Class& cls)
  {
    assert(classLowerer != nullptr);
    return classLowerer->declareVariables(cls);
  }

  bool Bridge::Impl::declareVariables(const ast::Model& model)
  {
    assert(modelLowerer != nullptr);
    return modelLowerer->declareVariables(model);
  }

  bool Bridge::Impl::declareVariables(const ast::Package& package)
  {
    assert(packageLowerer != nullptr);
    return packageLowerer->declareVariables(package);
  }

  void Bridge::Impl::declareVariables(
      const ast::PartialDerFunction& function)
  {
    assert(partialDerFunctionLowerer != nullptr);
    return partialDerFunctionLowerer->declareVariables(function);
  }

  bool Bridge::Impl::declareVariables(const ast::Record& record)
  {
    assert(recordLowerer != nullptr);
    return recordLowerer->declareVariables(record);
  }

  bool Bridge::Impl::declareVariables(
      const ast::StandardFunction& function)
  {
    assert(standardFunctionLowerer != nullptr);
    return standardFunctionLowerer->declareVariables(function);
  }

  bool Bridge::Impl::declare(const ast::Member& variable)
  {
    assert(classLowerer != nullptr);
    return classLowerer->declare(variable);
  }

  bool Bridge::Impl::lower(const ast::Class& cls)
  {
    assert(classLowerer != nullptr);
    return classLowerer->lower(cls);
  }

  bool
  Bridge::Impl::lower(const ast::Model& model)
  {
    assert(modelLowerer != nullptr);
    return modelLowerer->lower(model);
  }

  bool
  Bridge::Impl::lower(const ast::Package& package)
  {
    assert(packageLowerer != nullptr);
    return packageLowerer->lower(package);
  }

  void Bridge::Impl::lower(const ast::PartialDerFunction& function)
  {
    assert(partialDerFunctionLowerer != nullptr);
    return partialDerFunctionLowerer->lower(function);
  }

  bool
  Bridge::Impl::lower(const ast::Record& record)
  {
    assert(recordLowerer != nullptr);
    return recordLowerer->lower(record);
  }

  bool 
  Bridge::Impl::lower(const ast::StandardFunction& function)
  {
    assert(standardFunctionLowerer != nullptr);
    return standardFunctionLowerer->lower(function);
  }

  bool Bridge::Impl::lowerClassBody(const ast::Class& cls)
  {
    assert(classLowerer != nullptr);
    return classLowerer->lowerClassBody(cls);
  }

  bool Bridge::Impl::createBindingEquation(
      const ast::Member& variable,
      const ast::Expression& expression)
  {
    assert(classLowerer != nullptr);
    return classLowerer->createBindingEquation(variable, expression);
  }

  bool Bridge::Impl::lowerStartAttribute(
      mlir::SymbolRefAttr variable,
      const ast::Expression& expression,
      bool fixed,
      bool each)
  {
    assert(classLowerer != nullptr);

    return classLowerer->lowerStartAttribute(
        variable, expression, fixed, each);
  }

  std::optional<Results> Bridge::Impl::lower(const ast::Expression& expression)
  {
    assert(expressionLowerer != nullptr);
    return expressionLowerer->lower(expression);
  }

  std::optional<Results> Bridge::Impl::lower(const ast::ArrayGenerator& array)
  {
    assert(arrayLowerer != nullptr);
    return arrayLowerer->lower(array);
  }

  std::optional<Results> Bridge::Impl::lower(const ast::Call& call)
  {
    assert(callLowerer != nullptr);
    return callLowerer->lower(call);
  }

  Results Bridge::Impl::lower(const ast::Constant& constant)
  {
    assert(constantLowerer != nullptr);
    return constantLowerer->lower(constant);
  }

  std::optional<Results> Bridge::Impl::lower(const ast::Operation& operation)
  {
    assert(operationLowerer != nullptr);
    return operationLowerer->lower(operation);
  }

  std::optional<Results> Bridge::Impl::lower(
      const ast::ComponentReference& componentReference)
  {
    assert(componentReferenceLowerer != nullptr);
    return componentReferenceLowerer->lower(componentReference);
  }

  std::optional<Results> Bridge::Impl::lower(const ast::Tuple& tuple)
  {
    assert(tupleLowerer != nullptr);
    return tupleLowerer->lower(tuple);
  }

  std::optional<Results> Bridge::Impl::lower(const ast::Subscript& subscript)
  {
    assert(subscriptLowerer != nullptr);
    return subscriptLowerer->lower(subscript);
  }

  bool
  Bridge::Impl::lower(const ast::EquationSection& equationSection)
  {
    assert(equationLowerer != nullptr);
    return equationSectionLowerer->lower(equationSection);
  }

  bool
  Bridge::Impl::lower(const ast::Equation& equation)
  {
    assert(equationLowerer != nullptr);
    return equationLowerer->lower(equation);
  }

  bool
  Bridge::Impl::lower(const ast::EqualityEquation& equation)
  {
    assert(equationLowerer != nullptr);
    return equalityEquationLowerer->lower(equation);
  }

  bool
  Bridge::Impl::lower(const ast::ForEquation& forEquation)
  {
    assert(equationLowerer != nullptr);
    return forEquationLowerer->lower(forEquation);
  }

  void Bridge::Impl::lower(const ast::IfEquation& equation)
  {
    assert(equationLowerer != nullptr);
    return ifEquationLowerer->lower(equation);
  }

  void Bridge::Impl::lower(const ast::WhenEquation& equation)
  {
    assert(equationLowerer != nullptr);
    return whenEquationLowerer->lower(equation);
  }

  bool 
  Bridge::Impl::lower(const ast::Algorithm& algorithm)
  {
    assert(algorithmLowerer != nullptr);
    return algorithmLowerer->lower(algorithm);
  }

  bool 
  Bridge::Impl::lower(const ast::Statement& statement)
  {
    assert(statementLowerer != nullptr);
    return statementLowerer->lower(statement);
  }

  bool
  Bridge::Impl::lower(const ast::AssignmentStatement& statement)
  {
    assert(assignmentStatementLowerer != nullptr);
    return assignmentStatementLowerer->lower(statement);
  }

  void Bridge::Impl::lower(const ast::BreakStatement& statement)
  {
    assert(breakStatementLowerer != nullptr);
    return breakStatementLowerer->lower(statement);
  }

  bool 
  Bridge::Impl::lower(const ast::ForStatement& statement)
  {
    assert(forStatementLowerer != nullptr);
    return forStatementLowerer->lower(statement);
  }

  bool
  Bridge::Impl::lower(const ast::IfStatement& statement)
  {
    assert(ifStatementLowerer != nullptr);
    return ifStatementLowerer->lower(statement);
  }

  void Bridge::Impl::lower(const ast::ReturnStatement& statement)
  {
    assert(returnStatementLowerer != nullptr);
    return returnStatementLowerer->lower(statement);
  }

  void Bridge::Impl::lower(const ast::WhenStatement& statement)
  {
    assert(whenStatementLowerer != nullptr);
    return whenStatementLowerer->lower(statement);
  }

  bool 
  Bridge::Impl::lower(const ast::WhileStatement& statement)
  {
    assert(whileStatementLowerer != nullptr);
    return whileStatementLowerer->lower(statement);
  }

  void Bridge::Impl::emitIdentifierError(IdentifierError::IdentifierType identifierType, 
       std::string name, const std::set<std::string> &declaredIdentifiers,
       unsigned int line, unsigned int column)
  {
    const IdentifierError error(identifierType, name, declaredIdentifiers, line, column);
    const std::string actual = error.getActual();
    const std::string predicted = error.getPredicted();

    std::string errorString = "Error in AST to MLIR conversion. Unknown ";
    switch (identifierType) {
      case marco::codegen::lowering::IdentifierError::IdentifierType::FUNCTION: {
        errorString += "function";
        break;
      }
      case marco::codegen::lowering::IdentifierError::IdentifierType::VARIABLE: {
        errorString += "variable";
        break;
      }
      case marco::codegen::lowering::IdentifierError::IdentifierType::TYPE: {
        errorString += "type or class";
        break;
      }
      case marco::codegen::lowering::IdentifierError::IdentifierType::FIELD: {
        errorString += "field";
        break;
      }
      default: {
        llvm_unreachable("Unkown error type.");
        break;
      }
    }
    errorString += " identifier " + actual + " at line " + std::to_string(line) + 
                   ", column " + std::to_string(column) + ".";

    if (predicted != "") {
      errorString += " Did you mean " + predicted + "?";
    }
    
    emitError(errorString);
  }

  void Bridge::Impl::emitError(const std::string &error) {
    diag->Report(diag->getCustomDiagID(clang::DiagnosticsEngine::Fatal, "%0")) << error;
  }

  Bridge::Bridge(mlir::MLIRContext& context, clang::DiagnosticsEngine &diag)
    : impl(std::make_unique<Impl>(context, diag))
  {
  }

  Bridge::~Bridge() = default;

  bool
  Bridge::lower(const ast::Root& root)
  {
    return impl->convert(root);
  }

  std::unique_ptr<mlir::ModuleOp>& Bridge::getMLIRModule()
  {
    return impl->getMLIRModule();
  }
}
