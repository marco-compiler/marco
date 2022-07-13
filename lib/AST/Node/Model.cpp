#include "marco/AST/AST.h"

using namespace marco::ast;

namespace marco::ast
{
  Model::Model(SourceRange location,
               llvm::StringRef name,
               llvm::ArrayRef<std::unique_ptr<Member>> members,
               llvm::ArrayRef<std::unique_ptr<EquationsBlock>> equationsBlocks,
               llvm::ArrayRef<std::unique_ptr<EquationsBlock>> initialEquationsBlocks,
               llvm::ArrayRef<std::unique_ptr<Algorithm>> algorithms,
               llvm::ArrayRef<std::unique_ptr<Class>> innerClasses)
      : ASTNode(std::move(location)),
        name(name.str())
  {
    for (const auto& member : members) {
      this->members.push_back(member->clone());
    }

    for (const auto& equationsBlock : equationsBlocks) {
      this->equationsBlocks.push_back(equationsBlock->clone());
    }

    for (const auto& initialEquationsBlock : initialEquationsBlocks) {
      this->initialEquationsBlocks.push_back(initialEquationsBlock->clone());
    }

    for (const auto& algorithm : algorithms) {
      this->algorithms.push_back(algorithm->clone());
    }

    for (const auto& cls : innerClasses) {
      this->innerClasses.push_back(cls->clone());
    }
  }

  Model::Model(const Model& other)
      : ASTNode(other),
        name(other.name)
  {
    for (const auto& member : other.members) {
      this->members.push_back(member->clone());
    }

    for (const auto& equationsBlock : other.equationsBlocks) {
      this->equationsBlocks.push_back(equationsBlock->clone());
    }

    for (const auto& initialEquationsBlock : other.initialEquationsBlocks) {
      this->initialEquationsBlocks.push_back(initialEquationsBlock->clone());
    }

    for (const auto& algorithm : other.algorithms) {
      this->algorithms.push_back(algorithm->clone());
    }

    for (const auto& cls : other.innerClasses) {
      this->innerClasses.push_back(cls->clone());
    }
  }

  Model::Model(Model&& other) = default;

  Model::~Model() = default;

  Model& Model::operator=(const Model& other)
  {
    Model result(other);
    swap(*this, result);
    return *this;
  }

  Model& Model::operator=(Model&& other) = default;

  void swap(Model& first, Model& second)
  {
    using std::swap;
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    impl::swap(first.members, second.members);
    impl::swap(first.equationsBlocks, second.equationsBlocks);
    impl::swap(first.initialEquationsBlocks, second.initialEquationsBlocks);
    impl::swap(first.algorithms, second.algorithms);
    impl::swap(first.innerClasses, second.innerClasses);
  }

  void Model::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents);
    os << "model " << getName() << "\n";

    for (const auto& member : members) {
      member->print(os, indents + 1);
    }

    os << "equations\n";

    for (const auto& equationsBlock : equationsBlocks) {
      equationsBlock->print(os, indents + 1);
    }

    os << "initial equations\n";

    for (const auto& initialEquationsBlock : initialEquationsBlocks) {
      initialEquationsBlock->print(os, indents + 1);
    }

    for (const auto& algorithm : algorithms) {
      algorithm->print(os, indents + 1);
    }

    for (const auto& cls : innerClasses) {
      cls->print(os, indents + 1);
    }
  }

  llvm::StringRef Model::getName() const
  {
    return name;
  }

  llvm::MutableArrayRef<std::unique_ptr<Member>> Model::getMembers()
  {
    return members;
  }

  llvm::ArrayRef<std::unique_ptr<Member>> Model::getMembers() const
  {
    return members;
  }

  llvm::SmallVectorImpl<std::unique_ptr<Member>>& Model::getMembers_mut()
  {
    return members;
  }

  Member* Model::getMember(llvm::StringRef name) const
  {
    for(auto &m : members){
      if(m->getName() == name){
        return m.get();
      }
    }
    return nullptr;
  }

  void Model::addMember(std::unique_ptr<Member> member)
  {
    members.push_back(std::move(member));
  }

  llvm::MutableArrayRef<std::unique_ptr<EquationsBlock>> Model::getEquationsBlocks()
  {
    return equationsBlocks;
  }

  llvm::ArrayRef<std::unique_ptr<EquationsBlock>> Model::getEquationsBlocks() const
  {
    return equationsBlocks;
  }

  llvm::MutableArrayRef<std::unique_ptr<EquationsBlock>> Model::getInitialEquationsBlocks()
  {
    return initialEquationsBlocks;
  }

  llvm::ArrayRef<std::unique_ptr<EquationsBlock>> Model::getInitialEquationsBlocks() const
  {
    return initialEquationsBlocks;
  }

  llvm::MutableArrayRef<std::unique_ptr<Algorithm>> Model::getAlgorithms()
  {
    return algorithms;
  }

  llvm::ArrayRef<std::unique_ptr<Algorithm>> Model::getAlgorithms() const
  {
    return algorithms;
  }

  llvm::MutableArrayRef<std::unique_ptr<Class>> Model::getInnerClasses()
  {
    return innerClasses;
  }

  llvm::ArrayRef<std::unique_ptr<Class>> Model::getInnerClasses() const
  {
    return innerClasses;
  }
}
