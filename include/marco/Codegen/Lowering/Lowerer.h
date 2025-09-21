#ifndef MARCO_CODEGEN_LOWERING_LOWERER_H
#define MARCO_CODEGEN_LOWERING_LOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/LoweringContext.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/IR/Builders.h"

namespace marco::codegen::lowering {
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

  mlir::Operation *getClass(const ast::Class &cls);

  mlir::SymbolRefAttr getSymbolRefFromRoot(mlir::Operation *symbol);

  mlir::Operation *resolveClassName(llvm::StringRef name,
                                    mlir::Operation *currentScope);

  std::optional<mlir::Operation *> resolveType(const ast::UserDefinedType &type,
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

  virtual void declare(const ast::Class &node) override;

  virtual void declare(const ast::Model &node) override;

  virtual void declare(const ast::Package &node) override;

  virtual void declare(const ast::PartialDerFunction &node) override;

  virtual void declare(const ast::Record &node) override;

  virtual void declare(const ast::StandardFunction &node) override;

  [[nodiscard]] virtual bool declareVariables(const ast::Class &node) override;

  [[nodiscard]] virtual bool declareVariables(const ast::Model &model) override;

  [[nodiscard]] virtual bool
  declareVariables(const ast::Package &package) override;

  [[nodiscard]] virtual bool
  declareVariables(const ast::PartialDerFunction &function) override;

  [[nodiscard]] virtual bool
  declareVariables(const ast::Record &record) override;

  [[nodiscard]] virtual bool
  declareVariables(const ast::StandardFunction &function) override;

  [[nodiscard]] virtual bool declare(const ast::Member &node) override;

  [[nodiscard]] virtual bool lower(const ast::Class &node) override;

  [[nodiscard]] virtual bool lower(const ast::Model &node) override;

  [[nodiscard]] virtual bool lower(const ast::Package &node) override;

  [[nodiscard]] virtual bool
  lower(const ast::PartialDerFunction &node) override;

  [[nodiscard]] virtual bool lower(const ast::Record &node) override;

  [[nodiscard]] virtual bool lower(const ast::StandardFunction &node) override;

  [[nodiscard]] virtual bool lowerClassBody(const ast::Class &node) override;

  [[nodiscard]] virtual bool
  createBindingEquation(const ast::Member &variable,
                        const ast::Expression &expression) override;

  [[nodiscard]] virtual bool
  lowerStartAttribute(mlir::SymbolRefAttr variable,
                      const ast::Expression &expression, bool fixed,
                      bool each) override;

  virtual std::optional<Results>
  lower(const ast::Expression &expression) override;

  virtual std::optional<Results>
  lower(const ast::ArrayGenerator &node) override;

  virtual std::optional<Results> lower(const ast::Call &node) override;

  virtual std::optional<Results> lower(const ast::Constant &constant) override;

  virtual std::optional<Results>
  lower(const ast::Operation &operation) override;

  virtual std::optional<Results>
  lower(const ast::ComponentReference &componentReference) override;

  virtual std::optional<Results> lower(const ast::Tuple &tuple) override;

  virtual std::optional<Results>
  lower(const ast::Subscript &subscript) override;

  [[nodiscard]] virtual bool lower(const ast::EquationSection &node) override;

  [[nodiscard]] virtual bool lower(const ast::Equation &equation) override;

  [[nodiscard]] virtual bool
  lower(const ast::EqualityEquation &equation) override;

  [[nodiscard]] virtual bool lower(const ast::ForEquation &equation) override;

  [[nodiscard]] virtual bool lower(const ast::IfEquation &equation) override;

  [[nodiscard]] virtual bool lower(const ast::WhenEquation &equation) override;

  [[nodiscard]] virtual bool lower(const ast::Algorithm &node) override;

  [[nodiscard]] virtual bool lower(const ast::Statement &node) override;

  [[nodiscard]] virtual bool
  lower(const ast::AssignmentStatement &statement) override;

  [[nodiscard]] virtual bool lowerAssignmentToComponentReference(
      mlir::Location assignmentLoc, const ast::ComponentReference &destination,
      mlir::Value value) override;

  [[nodiscard]] virtual bool
  lower(const ast::BreakStatement &statement) override;

  [[nodiscard]] virtual bool
  lower(const ast::CallStatement &statement) override;

  [[nodiscard]] virtual bool lower(const ast::ForStatement &statement) override;

  [[nodiscard]] virtual bool lower(const ast::IfStatement &statement) override;

  [[nodiscard]] virtual bool
  lower(const ast::ReturnStatement &statement) override;

  [[nodiscard]] virtual bool
  lower(const ast::WhenStatement &statement) override;

  [[nodiscard]] virtual bool
  lower(const ast::WhileStatement &statement) override;

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
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_LOWERER_H
