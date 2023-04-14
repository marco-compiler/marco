#include "marco/AST/Node/Class.h"
#include "marco/AST/Node/Algorithm.h"
#include "marco/AST/Node/EquationsBlock.h"
#include "marco/AST/Node/Member.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Class::Class(const Class& other)
      : ASTNode(other),
        name(other.name)
  {
    setVariables(other.variables);
    setEquationsBlocks(other.equationsBlocks);
    setInitialEquationsBlocks(other.initialEquationsBlocks);
    setAlgorithms(other.algorithms);
    setInnerClasses(other.innerClasses);
  }

  Class::~Class() = default;

  void Class::addJSONProperties(llvm::json::Object& obj) const
  {
    obj["name"] = getName();

    llvm::SmallVector<llvm::json::Value> variablesJson;

    for (const auto& variable : variables) {
      variablesJson.push_back(variable->toJSON());
    }

    obj["variables"] = llvm::json::Array(variablesJson);

    llvm::SmallVector<llvm::json::Value> equationsBlocksJson;

    for (const auto& block : equationsBlocks) {
      equationsBlocksJson.push_back(block->toJSON());
    }

    obj["equations"] = llvm::json::Array(equationsBlocksJson);

    llvm::SmallVector<llvm::json::Value> initialEquationsBlocksJson;

    for (const auto& block : initialEquationsBlocks) {
      initialEquationsBlocksJson.push_back(block->toJSON());
    }

    obj["initial_equations"] = llvm::json::Array(initialEquationsBlocksJson);

    llvm::SmallVector<llvm::json::Value> algorithmsJson;

    for (const auto& algorithm : algorithms) {
      algorithmsJson.push_back(algorithm->toJSON());
    }

    obj["algorithms"] = llvm::json::Array(algorithmsJson);

    llvm::SmallVector<llvm::json::Value> innerClassesJson;

    for (const auto& innerClass : innerClasses) {
      innerClassesJson.push_back(innerClass->toJSON());
    }

    obj["inner_classes"] = llvm::json::Array(innerClassesJson);

    ASTNode::addJSONProperties(obj);
  }

  llvm::StringRef Class::getName() const
  {
    return name;
  }

  void Class::setName(llvm::StringRef newName)
  {
    name = newName.str();
  }

  llvm::ArrayRef<std::unique_ptr<ASTNode>> Class::getVariables() const
  {
    return variables;
  }

  void Class::setVariables(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> newVariables)
  {
    variables.clear();

    for (const auto& variable : newVariables) {
      assert(variable->isa<Member>());
      auto& clone = variables.emplace_back(variable->clone());
      clone->setParent(this);
    }
  }

  llvm::ArrayRef<std::unique_ptr<ASTNode>> Class::getEquationsBlocks() const
  {
    return equationsBlocks;
  }

  void Class::setEquationsBlocks(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> newBlocks)
  {
    equationsBlocks.clear();

    for (const auto& block : newBlocks) {
      assert(block->isa<EquationsBlock>());
      auto& clone = equationsBlocks.emplace_back(block->clone());
      clone->setParent(this);
    }
  }

  llvm::ArrayRef<std::unique_ptr<ASTNode>>
  Class::getInitialEquationsBlocks() const
  {
    return initialEquationsBlocks;
  }

  void Class::setInitialEquationsBlocks(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> newBlocks)
  {
    initialEquationsBlocks.clear();

    for (const auto& block : newBlocks) {
      assert(block->isa<EquationsBlock>());
      auto& clone = initialEquationsBlocks.emplace_back(block->clone());
      clone->setParent(this);
    }
  }

  llvm::ArrayRef<std::unique_ptr<ASTNode>> Class::getAlgorithms() const
  {
    return algorithms;
  }

  void Class::setAlgorithms(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> newAlgorithms)
  {
    algorithms.clear();

    for (const auto& algorithm : newAlgorithms) {
      assert(algorithm->isa<Algorithm>());
      auto& clone = algorithms.emplace_back(algorithm->clone());
      clone->setParent(this);
    }
  }

  llvm::ArrayRef<std::unique_ptr<ASTNode>> Class::getInnerClasses() const
  {
    return innerClasses;
  }

  void Class::setInnerClasses(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> newInnerClasses)
  {
    innerClasses.clear();

    for (const auto& cls : newInnerClasses) {
      assert(cls->isa<Class>());
      auto& clone = innerClasses.emplace_back(cls->clone());
      clone->setParent(this);
    }
  }
}
