#ifndef MARCO_AST_BASEMODELICA_FUNCTION_H
#define MARCO_AST_BASEMODELICA_FUNCTION_H

#include "marco/AST/BaseModelica/Annotation.h"
#include "marco/AST/BaseModelica/Class.h"
#include "marco/AST/BaseModelica/Expression.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

namespace marco::ast::bmodelica {
class Algorithm;
class Annotation;
class Member;
class VariableType;

class FunctionType {
public:
  FunctionType(llvm::ArrayRef<std::unique_ptr<ASTNode>> args,
               llvm::ArrayRef<std::unique_ptr<ASTNode>> results);

  size_t getNumOfArgs() const;

  const VariableType *getArg(size_t index) const;

  size_t getNumOfResults() const;

  const VariableType *getResult(size_t index) const;

private:
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> args;
  llvm::SmallVector<std::unique_ptr<ASTNode>, 1> results;
};

class Function : public Class {
public:
  using Class::Class;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() >= ASTNodeKind::Class_Function &&
           node->getKind<ASTNodeKind>() <=
               ASTNodeKind::Class_Function_LastFunction;
  }
};

class PartialDerFunction : public Function {
public:
  explicit PartialDerFunction(SourceRange location);

  PartialDerFunction(const PartialDerFunction &other);

  ~PartialDerFunction() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() ==
           ASTNodeKind::Class_Function_PartialDerFunction;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getDerivedFunction() const;

  void setDerivedFunction(std::unique_ptr<ASTNode> node);

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getIndependentVariables() const;

  void setIndependentVariables(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  std::unique_ptr<ASTNode> derivedFunction;
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> independentVariables;
};

class StandardFunction : public Function {
public:
  explicit StandardFunction(SourceRange location);

  StandardFunction(const StandardFunction &other);

  ~StandardFunction() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() ==
           ASTNodeKind::Class_Function_StandardFunction;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  bool isPure() const;

  void setPure(bool value);

  bool shouldBeInlined() const;

  FunctionType getType() const;

private:
  bool pure{true};
};

class DerivativeAnnotation {
public:
  DerivativeAnnotation(llvm::StringRef name, unsigned int order = 1);

  [[nodiscard]] llvm::StringRef getName() const;
  [[nodiscard]] unsigned int getOrder() const;

private:
  std::string name;
  unsigned int order;
};

class InverseFunctionAnnotation {
private:
  template <typename T>
  using Container = llvm::SmallVector<T, 3>;

public:
  InverseFunctionAnnotation();

  [[nodiscard]] bool isInvertible(llvm::StringRef arg) const;
  [[nodiscard]] llvm::StringRef
  getInverseFunction(llvm::StringRef invertibleArg) const;
  [[nodiscard]] llvm::ArrayRef<std::string>
  getInverseArgs(llvm::StringRef invertibleArg) const;
  void addInverse(llvm::StringRef invertedArg,
                  llvm::StringRef inverseFunctionName,
                  llvm::ArrayRef<std::string> args);

private:
  llvm::StringMap<std::pair<std::string, Container<std::string>>> map;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_FUNCTION_H
