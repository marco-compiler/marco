#ifndef MARCO_CODEGEN_LOWERING_LOWERER_H
#define MARCO_CODEGEN_LOWERING_LOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/BridgeInterface.h"
#include "marco/Codegen/Lowering/LoweringContext.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/Builders.h"

namespace marco::codegen::lowering
{
  class Lowerer : public BridgeInterface
  {
    public:
      using SymbolScope = LoweringContext::SymbolScope;

      Lowerer(LoweringContext* context, BridgeInterface* bridge);

      virtual ~Lowerer();

    protected:
      /// Helper to convert an AST location to a MLIR location.
      mlir::Location loc(const SourcePosition& location);

      /// Helper to convert an AST location range to a MLIR location.
      mlir::Location loc(const SourceRange& location);

      /// @name Utility getters
      /// {

      LoweringContext* context();

      mlir::OpBuilder& builder();

      LoweringContext::SymbolTable& symbolTable();

      /// }
      /// @name Forwarded methods
      /// {

      virtual std::vector<mlir::Operation*> lower(const ast::Class& cls) override;

      virtual Results lower(const ast::Expression& expression) override;

      virtual void lower(const ast::Algorithm& algorithm) override;

      virtual void lower(const ast::Statement& statement) override;

      virtual void lower(const ast::Equation& equation, bool initialEquation) override;

      virtual void lower(const ast::ForEquation& forEquation, bool initialEquation) override;

      /// }

      virtual mlir::Type lower(const ast::Type& type);

      virtual mlir::Type lower(const ast::BuiltInType& type);

      virtual mlir::Type lower(const ast::PackedType& type);

      virtual mlir::Type lower(const ast::UserDefinedType& type);

    private:
      LoweringContext* context_;

      BridgeInterface* bridge_;
  };
}

#endif // MARCO_CODEGEN_LOWERING_LOWERER_H
