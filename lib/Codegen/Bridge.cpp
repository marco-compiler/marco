#include "marco/Codegen/Bridge.h"
#include "marco/Codegen/BridgeInterface.h"
#include "marco/Codegen/Lowering/LoweringContext.h"
#include "marco/Codegen/Lowering/AlgorithmLowerer.h"
#include "marco/Codegen/Lowering/ClassLowerer.h"
#include "marco/Codegen/Lowering/EquationLowerer.h"
#include "marco/Codegen/Lowering/ExpressionLowerer.h"
#include "marco/Codegen/Lowering/StatementLowerer.h"
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

      std::vector<mlir::Operation*> lower(const ast::Class& cls) override;

      Results lower(const ast::Expression& expression) override;

      void lower(const ast::Algorithm& algorithm) override;

      void lower(const ast::Statement& statement) override;

      void lower(const ast::Equation& equation, bool initialEquation) override;

      void lower(const ast::ForEquation& forEquation, bool initialEquation) override;

      std::unique_ptr<mlir::ModuleOp>& getMLIRModule();

    private:
      std::unique_ptr<LoweringContext> context;

      std::unique_ptr<ClassLowerer> classLowerer;
      std::unique_ptr<ExpressionLowerer> expressionLowerer;
      std::unique_ptr<AlgorithmLowerer> algorithmLowerer;
      std::unique_ptr<StatementLowerer> statementLowerer;
      std::unique_ptr<EquationLowerer> equationLowerer;

      // The module that is populated while converting the AST
      std::unique_ptr<mlir::ModuleOp> module;
  };

  Bridge::Impl::Impl(mlir::MLIRContext& context, CodegenOptions options)
  {
    this->context = std::make_unique<LoweringContext>(context, std::move(options));

    this->classLowerer = std::make_unique<ClassLowerer>(this->context.get(), this);
    this->expressionLowerer = std::make_unique<ExpressionLowerer>(this->context.get(), this);
    this->algorithmLowerer = std::make_unique<AlgorithmLowerer>(this->context.get(), this);
    this->statementLowerer = std::make_unique<StatementLowerer>(this->context.get(), this);
    this->equationLowerer = std::make_unique<EquationLowerer>(this->context.get(), this);

    this->module = std::make_unique<mlir::ModuleOp>(mlir::ModuleOp::create(this->context->builder.getUnknownLoc()));
  }

  Bridge::Impl::~Impl()
  {
    if (module != nullptr) {
      module->erase();
    }
  }

  std::vector<mlir::Operation*> Bridge::Impl::lower(const ast::Class& cls)
  {
    assert(classLowerer != nullptr);
    return cls.visit(*classLowerer);
  }

  Results Bridge::Impl::lower(const ast::Expression& expression)
  {
    assert(expressionLowerer != nullptr);
    return expression.visit(*expressionLowerer);
  }

  void Bridge::Impl::lower(const ast::Algorithm& algorithm)
  {
    assert(algorithmLowerer != nullptr);
    algorithmLowerer->lower(algorithm);
  }

  void Bridge::Impl::lower(const ast::Statement& statement)
  {
    assert(statementLowerer != nullptr);
    statement.visit(*statementLowerer);
  }

  void Bridge::Impl::lower(const ast::Equation& equation, bool initialEquation)
  {
    assert(equationLowerer != nullptr);
    equationLowerer->lower(equation, initialEquation);
  }

  void Bridge::Impl::lower(const ast::ForEquation& forEquation, bool initialEquation)
  {
    assert(equationLowerer != nullptr);
    equationLowerer->lower(forEquation, initialEquation);
  }

  std::unique_ptr<mlir::ModuleOp>& Bridge::Impl::getMLIRModule()
  {
    assert(module != nullptr);
    return module;
  }

  Bridge::Bridge(mlir::MLIRContext& context, CodegenOptions options)
    : impl(std::make_unique<Impl>(context, std::move(options)))
  {
  }

  Bridge::~Bridge() = default;

  void Bridge::lower(const ast::Class& cls)
  {
    for (auto* op : impl->lower(cls)) {
      if (op != nullptr) {
        getMLIRModule()->push_back(op);
      }
    }
  }

  std::unique_ptr<mlir::ModuleOp>& Bridge::getMLIRModule()
  {
    return impl->getMLIRModule();
  }
}
