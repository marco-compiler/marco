#include "marco/Codegen/Bridge.h"
#include "marco/Codegen/BridgeInterface.h"
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
#include "marco/Codegen/Lowering/ExpressionLowerer.h"
#include "marco/Codegen/Lowering/ForStatementLowerer.h"
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
#include "marco/Codegen/Lowering/WhenStatementLowerer.h"
#include "marco/Codegen/Lowering/WhileStatementLowerer.h"
#include <memory>

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  BridgeInterface::~BridgeInterface() = default;

  class Bridge::Impl : public BridgeInterface
  {
    public:
      Impl(mlir::MLIRContext& context, CodegenOptions options);

      ~Impl();

      LoweringContext& getContext() override;

      const LoweringContext& getContext() const override;

      mlir::Operation* getRoot() const override;

      std::unique_ptr<mlir::ModuleOp>& getMLIRModule();

      void convert(const ast::Root& root);

      void declare(const ast::Class& node) override;

      void declare(const ast::Model& node) override;

      void declare(const ast::Package& node) override;

      void declare(const ast::PartialDerFunction& node) override;

      void declare(const ast::Record& node) override;

      void declare(const ast::StandardFunction& node) override;

      void declareVariables(const ast::Class& node) override;

      void declareVariables(const ast::Model& model) override;

      void declareVariables(const ast::Package& package) override;

      void declareVariables(
          const ast::PartialDerFunction& function) override;

      void declareVariables(const ast::Record& record) override;

      void declareVariables(
          const ast::StandardFunction& function) override;

      void declare(const ast::Member& node) override;

      void lower(const ast::Class& node) override;

      void lower(const ast::Model& node) override;

      void lower(const ast::Package& node) override;

      void lower(const ast::PartialDerFunction& node) override;

      void lower(const ast::Record& node) override;

      void lower(const ast::StandardFunction& node) override;

      void lowerClassBody(const ast::Class& node) override;

      void createBindingEquation(
          const ast::Member& variable,
          const ast::Expression& expression) override;

      void lowerStartAttribute(
          const ast::Member& variable,
          const ast::Expression& expression,
          bool fixed,
          bool each) override;

      Results lower(const ast::Expression& expression) override;

      Results lower(const ast::ArrayGenerator& node) override;

      Results lower(const ast::Call& node) override;

      Results lower(const ast::Constant& constant) override;

      Results lower(const ast::Operation& operation) override;

      Results lower(
          const ast::ComponentReference& componentReference) override;

      Results lower(const ast::Tuple& tuple) override;

      Results lower(const ast::Subscript& subscript) override;

      void lower(const ast::Algorithm& node) override;

      void lower(const ast::Statement& node) override;

      void lower(const ast::AssignmentStatement& statement) override;

      void lower(const ast::BreakStatement& statement) override;

      void lower(const ast::ForStatement& statement) override;

      void lower(const ast::IfStatement& statement) override;

      void lower(const ast::ReturnStatement& statement) override;

      void lower(const ast::WhenStatement& statement) override;

      void lower(const ast::WhileStatement& statement) override;

      virtual void lower(
          const ast::Equation& equation,
          bool initialEquation) override;

      virtual void lower(
          const ast::ForEquation& forEquation,
          bool initialEquation) override;

    private:
      std::unique_ptr<LoweringContext> context;

      // The module that is populated while converting the AST.
      std::unique_ptr<mlir::ModuleOp> module;

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
      std::unique_ptr<AlgorithmLowerer> algorithmLowerer;
      std::unique_ptr<StatementLowerer> statementLowerer;
      std::unique_ptr<AssignmentStatementLowerer> assignmentStatementLowerer;
      std::unique_ptr<BreakStatementLowerer> breakStatementLowerer;
      std::unique_ptr<ForStatementLowerer> forStatementLowerer;
      std::unique_ptr<IfStatementLowerer> ifStatementLowerer;
      std::unique_ptr<ReturnStatementLowerer> returnStatementLowerer;
      std::unique_ptr<WhenStatementLowerer> whenStatementLowerer;
      std::unique_ptr<WhileStatementLowerer> whileStatementLowerer;
      std::unique_ptr<EquationLowerer> equationLowerer;
  };

  Bridge::Impl::Impl(mlir::MLIRContext& context, CodegenOptions options)
  {
    this->context = std::make_unique<LoweringContext>(
        context, std::move(options));

    this->module = std::make_unique<mlir::ModuleOp>(
        mlir::ModuleOp::create(this->context->getBuilder().getUnknownLoc()));

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

    this->equationLowerer = std::make_unique<EquationLowerer>(this);
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

  void Bridge::Impl::convert(const ast::Root& root)
  {
    mlir::OpBuilder::InsertionGuard guard(context->getBuilder());
    context->pushLookupScope(module->getOperation());
    context->getBuilder().setInsertionPointToStart(module->getBody());

    for (const auto& cls : root.getInnerClasses()) {
      classLowerer->declare(*cls->cast<ast::Class>());
    }

    for (const auto& cls : root.getInnerClasses()) {
      classLowerer->declareVariables(*cls->cast<ast::Class>());
    }

    for (const auto& cls : root.getInnerClasses()) {
      classLowerer->lower(*cls->cast<ast::Class>());
    }
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

  void Bridge::Impl::declareVariables(const ast::Class& cls)
  {
    assert(classLowerer != nullptr);
    return classLowerer->declareVariables(cls);
  }

  void Bridge::Impl::declareVariables(const ast::Model& model)
  {
    assert(modelLowerer != nullptr);
    return modelLowerer->declareVariables(model);
  }

  void Bridge::Impl::declareVariables(const ast::Package& package)
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

  void Bridge::Impl::declareVariables(const ast::Record& record)
  {
    assert(recordLowerer != nullptr);
    return recordLowerer->declareVariables(record);
  }

  void Bridge::Impl::declareVariables(
      const ast::StandardFunction& function)
  {
    assert(standardFunctionLowerer != nullptr);
    return standardFunctionLowerer->declareVariables(function);
  }

  void Bridge::Impl::declare(const ast::Member& variable)
  {
    assert(classLowerer != nullptr);
    return classLowerer->declare(variable);
  }

  void Bridge::Impl::lower(const ast::Class& cls)
  {
    assert(classLowerer != nullptr);
    return classLowerer->lower(cls);
  }

  void Bridge::Impl::lower(const ast::Model& model)
  {
    assert(modelLowerer != nullptr);
    return modelLowerer->lower(model);
  }

  void Bridge::Impl::lower(const ast::Package& package)
  {
    assert(packageLowerer != nullptr);
    return packageLowerer->lower(package);
  }

  void Bridge::Impl::lower(const ast::PartialDerFunction& function)
  {
    assert(partialDerFunctionLowerer != nullptr);
    return partialDerFunctionLowerer->lower(function);
  }

  void Bridge::Impl::lower(const ast::Record& record)
  {
    assert(recordLowerer != nullptr);
    return recordLowerer->lower(record);
  }

  void Bridge::Impl::lower(const ast::StandardFunction& function)
  {
    assert(standardFunctionLowerer != nullptr);
    return standardFunctionLowerer->lower(function);
  }

  void Bridge::Impl::lowerClassBody(const ast::Class& cls)
  {
    assert(classLowerer != nullptr);
    return classLowerer->lowerClassBody(cls);
  }

  void Bridge::Impl::createBindingEquation(
      const ast::Member& variable,
      const ast::Expression& expression)
  {
    assert(classLowerer != nullptr);
    return classLowerer->createBindingEquation(variable, expression);
  }

  void Bridge::Impl::lowerStartAttribute(
      const ast::Member& variable,
      const ast::Expression& expression,
      bool fixed,
      bool each)
  {
    assert(classLowerer != nullptr);

    return classLowerer->lowerStartAttribute(
        variable, expression, fixed, each);
  }

  Results Bridge::Impl::lower(const ast::Expression& expression)
  {
    assert(expressionLowerer != nullptr);
    return expressionLowerer->lower(expression);
  }

  Results Bridge::Impl::lower(const ast::ArrayGenerator& array)
  {
    assert(arrayLowerer != nullptr);
    return arrayLowerer->lower(array);
  }

  Results Bridge::Impl::lower(const ast::Call& call)
  {
    assert(callLowerer != nullptr);
    return callLowerer->lower(call);
  }

  Results Bridge::Impl::lower(const ast::Constant& constant)
  {
    assert(constantLowerer != nullptr);
    return constantLowerer->lower(constant);
  }

  Results Bridge::Impl::lower(const ast::Operation& operation)
  {
    assert(operationLowerer != nullptr);
    return operationLowerer->lower(operation);
  }

  Results Bridge::Impl::lower(
      const ast::ComponentReference& componentReference)
  {
    assert(componentReferenceLowerer != nullptr);
    return componentReferenceLowerer->lower(componentReference);
  }

  Results Bridge::Impl::lower(const ast::Tuple& tuple)
  {
    assert(tupleLowerer != nullptr);
    return tupleLowerer->lower(tuple);
  }

  Results Bridge::Impl::lower(const ast::Subscript& subscript)
  {
    assert(subscriptLowerer != nullptr);
    return subscriptLowerer->lower(subscript);
  }

  void Bridge::Impl::lower(const ast::Algorithm& algorithm)
  {
    assert(algorithmLowerer != nullptr);
    return algorithmLowerer->lower(algorithm);
  }

  void Bridge::Impl::lower(const ast::Statement& statement)
  {
    assert(statementLowerer != nullptr);
    return statementLowerer->lower(statement);
  }

  void Bridge::Impl::lower(const ast::AssignmentStatement& statement)
  {
    assert(assignmentStatementLowerer != nullptr);
    return assignmentStatementLowerer->lower(statement);
  }

  void Bridge::Impl::lower(const ast::BreakStatement& statement)
  {
    assert(breakStatementLowerer != nullptr);
    return breakStatementLowerer->lower(statement);
  }

  void Bridge::Impl::lower(const ast::ForStatement& statement)
  {
    assert(forStatementLowerer != nullptr);
    return forStatementLowerer->lower(statement);
  }

  void Bridge::Impl::lower(const ast::IfStatement& statement)
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

  void Bridge::Impl::lower(const ast::WhileStatement& statement)
  {
    assert(whileStatementLowerer != nullptr);
    return whileStatementLowerer->lower(statement);
  }

  void Bridge::Impl::lower(
      const ast::Equation& equation,
      bool initialEquation)
  {
    assert(equationLowerer != nullptr);
    return equationLowerer->lower(equation, initialEquation);
  }

  void Bridge::Impl::lower(
      const ast::ForEquation& forEquation,
      bool initialEquation)
  {
    assert(equationLowerer != nullptr);
    return equationLowerer->lower(forEquation, initialEquation);
  }


  Bridge::Bridge(mlir::MLIRContext& context, CodegenOptions options)
    : impl(std::make_unique<Impl>(context, std::move(options)))
  {
  }

  Bridge::~Bridge() = default;

  void Bridge::lower(const ast::Root& root)
  {
    impl->convert(root);
  }

  std::unique_ptr<mlir::ModuleOp>& Bridge::getMLIRModule()
  {
    return impl->getMLIRModule();
  }
}
