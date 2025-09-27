#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_BRIDGEINTERFACE_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_BRIDGEINTERFACE_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/IdentifierError.h"
#include "marco/Codegen/Lowering/BaseModelica/LoweringContext.h"
#include "marco/Codegen/Lowering/BaseModelica/Results.h"
#include <optional>

namespace marco::codegen::lowering::bmodelica {
class BridgeInterface {
public:
  virtual ~BridgeInterface();

  virtual LoweringContext &getContext() = 0;

  virtual const LoweringContext &getContext() const = 0;

  virtual mlir::Operation *getRoot() const = 0;

  virtual void declare(const ast::bmodelica::Class &cls) = 0;

  virtual void declare(const ast::bmodelica::Model &model) = 0;

  virtual void declare(const ast::bmodelica::Package &package) = 0;

  virtual void declare(const ast::bmodelica::PartialDerFunction &function) = 0;

  virtual void declare(const ast::bmodelica::Record &record) = 0;

  virtual void declare(const ast::bmodelica::StandardFunction &function) = 0;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::Class &cls) = 0;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::Model &model) = 0;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::Package &package) = 0;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::PartialDerFunction &function) = 0;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::Record &record) = 0;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::StandardFunction &function) = 0;

  [[nodiscard]] virtual bool
  declare(const ast::bmodelica::Member &variable) = 0;

  [[nodiscard]] virtual bool lower(const ast::bmodelica::Class &cls) = 0;

  [[nodiscard]] virtual bool lower(const ast::bmodelica::Model &model) = 0;

  [[nodiscard]] virtual bool lower(const ast::bmodelica::Package &package) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::PartialDerFunction &function) = 0;

  [[nodiscard]] virtual bool lower(const ast::bmodelica::Record &record) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::StandardFunction &function) = 0;

  [[nodiscard]] virtual bool
  lowerClassBody(const ast::bmodelica::Class &cls) = 0;

  [[nodiscard]] virtual bool
  createBindingEquation(const ast::bmodelica::Member &variable,
                        const ast::bmodelica::Expression &expression) = 0;

  [[nodiscard]] virtual bool
  lowerStartAttribute(mlir::SymbolRefAttr variable,
                      const ast::bmodelica::Expression &expression, bool fixed,
                      bool each) = 0;

  virtual std::optional<Results>
  lower(const ast::bmodelica::Expression &expression) = 0;

  virtual std::optional<Results>
  lower(const ast::bmodelica::ArrayGenerator &array) = 0;

  virtual std::optional<Results> lower(const ast::bmodelica::Call &call) = 0;

  virtual std::optional<Results>
  lower(const ast::bmodelica::Constant &constant) = 0;

  virtual std::optional<Results>
  lower(const ast::bmodelica::Operation &operation) = 0;

  virtual std::optional<Results>
  lower(const ast::bmodelica::ComponentReference &componentReference) = 0;

  virtual std::optional<Results> lower(const ast::bmodelica::Tuple &tuple) = 0;

  virtual std::optional<Results>
  lower(const ast::bmodelica::Subscript &subscript) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::EquationSection &node) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::Equation &equation) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::EqualityEquation &equation) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::ForEquation &equation) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::IfEquation &equation) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::WhenEquation &equation) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::Algorithm &algorithm) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::Statement &statement) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::AssignmentStatement &statement) = 0;

  [[nodiscard]] virtual bool lowerAssignmentToComponentReference(
      mlir::Location assignmentLoc,
      const ast::bmodelica::ComponentReference &destination,
      mlir::Value value) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::BreakStatement &statement) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::CallStatement &statement) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::ForStatement &statement) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::IfStatement &statement) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::ReturnStatement &statement) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::WhenStatement &statement) = 0;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::WhileStatement &statement) = 0;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_BRIDGEINTERFACE_H
