#include "marco/Codegen/NewBridge.h"
#include "marco/Codegen/BridgeInterface.h"
#include "marco/Codegen/Lowering/LoweringContext.h"
#include "marco/Codegen/Lowering/ClassLowerer.h"
#include "marco/Codegen/Lowering/ExpressionLowerer.h"
#include "marco/Codegen/Lowering/EquationLowerer.h"
#include "marco/Codegen/Lowering/StatementLowerer.h"
#include <functional>
#include <memory>

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  class Bridge::Impl : public BridgeInterface
  {
    public:
      Impl(mlir::MLIRContext& context, CodegenOptions options);

      std::vector<mlir::Operation*> lower(const ast::Class& cls) override;

      Results lower(const ast::Expression& expression) override;

      void lower(const ast::Statement& statement) override;

      void lower(const ast::Equation& equation) override;

      void lower(const ast::ForEquation& forEquation) override;

      std::unique_ptr<mlir::ModuleOp>& getMLIRModule();

    private:
      std::unique_ptr<LoweringContext> context;

      std::unique_ptr<ClassLowerer> classLowerer;
      std::unique_ptr<ExpressionLowerer> expressionLowerer;
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
    this->statementLowerer = std::make_unique<StatementLowerer>(this->context.get(), this);
    this->equationLowerer = std::make_unique<EquationLowerer>(this->context.get(), this);

    this->module = std::make_unique<mlir::ModuleOp>(mlir::ModuleOp::create(this->context->builder.getUnknownLoc()));
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

  void Bridge::Impl::lower(const ast::Statement& statement)
  {
    assert(statementLowerer != nullptr);
    statement.visit(*statementLowerer);
  }

  void Bridge::Impl::lower(const ast::Equation& equation)
  {
    assert(equationLowerer != nullptr);
    equationLowerer->lower(equation);
  }

  void Bridge::Impl::lower(const ast::ForEquation& forEquation)
  {
    assert(equationLowerer != nullptr);
    equationLowerer->lower(forEquation);
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
