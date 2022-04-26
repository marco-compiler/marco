#ifndef MARCO_AST_NODE_MODEL_H
#define MARCO_AST_NODE_MODEL_H

#include "marco/AST/Node/Class.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <string>

namespace marco::ast
{
  class Algorithm;
  class Class;
  class EquationsBlock;
  class Equation;
  class ForEquation;
  class Member;

  class Model
    : public ASTNode,
      public impl::Dumpable<Model>
  {
    private:
      template<typename T> using Container = llvm::SmallVector<T, 3>;

    public:
      Model(const Model& other);
      Model(Model&& other);
      ~Model() override;

      Model& operator=(const Model& other);
      Model& operator=(Model&& other);

      friend void swap(Model& first, Model& second);

      void print(llvm::raw_ostream& os, size_t indents) const override;

      [[nodiscard]] llvm::StringRef getName() const;

      [[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Member>> getMembers();
      [[nodiscard]] llvm::ArrayRef<std::unique_ptr<Member>> getMembers() const;

      void addMember(std::unique_ptr<Member> member);

      [[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<EquationsBlock>> getEquationsBlocks();
      [[nodiscard]] llvm::ArrayRef<std::unique_ptr<EquationsBlock>> getEquationsBlocks() const;

      [[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<EquationsBlock>> getInitialEquationsBlocks();
      [[nodiscard]] llvm::ArrayRef<std::unique_ptr<EquationsBlock>> getInitialEquationsBlocks() const;

      [[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Algorithm>> getAlgorithms();
      [[nodiscard]] llvm::ArrayRef<std::unique_ptr<Algorithm>> getAlgorithms() const;

      [[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Class>> getInnerClasses();
      [[nodiscard]] llvm::ArrayRef<std::unique_ptr<Class>> getInnerClasses() const;

    private:
      friend class Class;

      Model(SourceRange location,
            llvm::StringRef name,
            llvm::ArrayRef<std::unique_ptr<Member>> members,
            llvm::ArrayRef<std::unique_ptr<EquationsBlock>> equationsBlocks,
            llvm::ArrayRef<std::unique_ptr<EquationsBlock>> initialEquationsBlocks,
            llvm::ArrayRef<std::unique_ptr<Algorithm>> algorithms,
            llvm::ArrayRef<std::unique_ptr<Class>> innerClasses);

    private:
      std::string name;
      Container<std::unique_ptr<Member>> members;
      Container<std::unique_ptr<EquationsBlock>> equationsBlocks;
      Container<std::unique_ptr<EquationsBlock>> initialEquationsBlocks;
      Container<std::unique_ptr<Algorithm>> algorithms;
      Container<std::unique_ptr<Class>> innerClasses;
  };
}

#endif // MARCO_AST_NODE_MODEL_H
