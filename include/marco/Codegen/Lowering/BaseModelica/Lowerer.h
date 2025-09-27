#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_LOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_LOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/LoweringContext.h"
#include "marco/Codegen/Lowering/BaseModelica/Results.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/IR/Builders.h"

namespace marco::codegen::lowering::bmodelica {
class Lowerer : public BridgeInterface {
public:
  using LookupScopeGuard = LoweringContext::LookupScopeGuard;

  explicit Lowerer(BridgeInterface *bridge);

  ~Lowerer() override;

protected:
  /// Helper to convert an AST location to a MLIR location.
  mlir::Location loc(const SourcePosition &location);

  /// Helper to convert an AST location range to a MLIR location.
  mlir::Location loc(const SourceRange &location);

  /// Helper function to initialize visibleSymbols, adding all the declared
  /// symbols of type T that are visible from the given scope.
  template <typename... T>
  void getVisibleSymbols(mlir::Operation *scope,
                         std::set<std::string> &visibleSymbols) {
    return getVisibleSymbols(scope, visibleSymbols, [](mlir::Operation *op) {
      return mlir::isa<T...>(op);
    });
  }

  /// Helper function to initialize visibleSymbols, adding all the declared
  /// symbols that are visible from the given scope.
  void getVisibleSymbols(mlir::Operation *scope,
                         std::set<std::string> &visibleSymbols);

  /// @name Utility getters.
  /// {

  mlir::OpBuilder &builder();

  mlir::SymbolTableCollection &getSymbolTable();

  VariablesSymbolTable &getVariablesSymbolTable();

  mlir::Operation *getLookupScope();

  void pushLookupScope(mlir::Operation *lookupScope);

  mlir::Operation *getClass(const ast::bmodelica::Class &cls);

  mlir::SymbolRefAttr getSymbolRefFromRoot(mlir::Operation *symbol);

  mlir::Operation *resolveClassName(llvm::StringRef name,
                                    mlir::Operation *currentScope);

  std::optional<mlir::Operation *>
  resolveType(const ast::bmodelica::UserDefinedType &type,
              mlir::Operation *lookupScope);

  mlir::Operation *resolveTypeFromRoot(mlir::SymbolRefAttr name);

  mlir::Operation *resolveSymbolName(llvm::StringRef name,
                                     mlir::Operation *currentScope);

  template <typename... T>
  mlir::Operation *resolveSymbolName(llvm::StringRef name,
                                     mlir::Operation *currentScope) {
    return resolveSymbolName(name, currentScope, [](mlir::Operation *op) {
      return mlir::isa<T...>(op);
    });
  }

  std::optional<Reference> lookupVariable(llvm::StringRef name);

  void insertVariable(llvm::StringRef name, Reference reference);

  bool isScalarType(mlir::Type type);

  /// }
  /// @name Forwarded methods.
  /// {

  LoweringContext &getContext() override;

  const LoweringContext &getContext() const override;

  mlir::Operation *getRoot() const override;

  virtual void declare(const ast::bmodelica::Class &node) override;

  virtual void declare(const ast::bmodelica::Model &node) override;

  virtual void declare(const ast::bmodelica::Package &node) override;

  virtual void declare(const ast::bmodelica::PartialDerFunction &node) override;

  virtual void declare(const ast::bmodelica::Record &node) override;

  virtual void declare(const ast::bmodelica::StandardFunction &node) override;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::Class &node) override;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::Model &model) override;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::Package &package) override;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::PartialDerFunction &function) override;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::Record &record) override;

  [[nodiscard]] virtual bool
  declareVariables(const ast::bmodelica::StandardFunction &function) override;

  [[nodiscard]] virtual bool
  declare(const ast::bmodelica::Member &node) override;

  [[nodiscard]] virtual bool lower(const ast::bmodelica::Class &node) override;

  [[nodiscard]] virtual bool lower(const ast::bmodelica::Model &node) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::Package &node) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::PartialDerFunction &node) override;

  [[nodiscard]] virtual bool lower(const ast::bmodelica::Record &node) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::StandardFunction &node) override;

  [[nodiscard]] virtual bool
  lowerClassBody(const ast::bmodelica::Class &node) override;

  [[nodiscard]] virtual bool
  createBindingEquation(const ast::bmodelica::Member &variable,
                        const ast::bmodelica::Expression &expression) override;

  [[nodiscard]] virtual bool
  lowerStartAttribute(mlir::SymbolRefAttr variable,
                      const ast::bmodelica::Expression &expression, bool fixed,
                      bool each) override;

  virtual std::optional<Results>
  lower(const ast::bmodelica::Expression &expression) override;

  virtual std::optional<Results>
  lower(const ast::bmodelica::ArrayGenerator &node) override;

  virtual std::optional<Results>
  lower(const ast::bmodelica::Call &node) override;

  virtual std::optional<Results>
  lower(const ast::bmodelica::Constant &constant) override;

  virtual std::optional<Results>
  lower(const ast::bmodelica::Operation &operation) override;

  virtual std::optional<Results>
  lower(const ast::bmodelica::ComponentReference &componentReference) override;

  virtual std::optional<Results>
  lower(const ast::bmodelica::Tuple &tuple) override;

  virtual std::optional<Results>
  lower(const ast::bmodelica::Subscript &subscript) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::EquationSection &node) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::Equation &equation) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::EqualityEquation &equation) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::ForEquation &equation) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::IfEquation &equation) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::WhenEquation &equation) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::Algorithm &node) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::Statement &node) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::AssignmentStatement &statement) override;

  [[nodiscard]] virtual bool lowerAssignmentToComponentReference(
      mlir::Location assignmentLoc,
      const ast::bmodelica::ComponentReference &destination,
      mlir::Value value) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::BreakStatement &statement) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::CallStatement &statement) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::ForStatement &statement) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::IfStatement &statement) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::ReturnStatement &statement) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::WhenStatement &statement) override;

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::WhileStatement &statement) override;

  virtual void
  emitIdentifierError(IdentifierError::IdentifierType identifierType,
                      llvm::StringRef name,
                      const std::set<std::string> &declaredIdentifiers,
                      const marco::SourceRange &location);

  /// }

private:
  mlir::Operation *
  resolveSymbolName(llvm::StringRef name, mlir::Operation *currentScope,
                    llvm::function_ref<bool(mlir::Operation *)> filterFn);

  void getVisibleSymbols(mlir::Operation *scope,
                         std::set<std::string> &visibleSymbols,
                         llvm::function_ref<bool(mlir::Operation *)> filterFn);

private:
  BridgeInterface *bridge;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_LOWERER_H
