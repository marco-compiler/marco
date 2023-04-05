#ifndef MARCO_CODEGEN_LOWERING_CLASSPATH_H
#define MARCO_CODEGEN_LOWERING_CLASSPATH_H

#include "marco/AST/AST.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace marco::codegen::lowering
{
  class ClassPath
  {
    public:
      class Scope
      {
        public:
          Scope(ClassPath& path);

          ~Scope();

        private:
          ClassPath* path;
          size_t size;
      };

      bool operator==(const ClassPath& other) const;

      bool operator!=(const ClassPath& other) const;

      bool operator<(const ClassPath& other) const;

      llvm::ArrayRef<const ast::Class*> get() const;

      size_t size() const;

      void append(const ast::Class& cls);

      const ast::Class& getLeaf() const;

    private:
      std::vector<const ast::Class*> path;
  };
}

namespace llvm
{
  template<>
  struct DenseMapInfo<marco::codegen::lowering::ClassPath>
  {
    static inline marco::codegen::lowering::ClassPath getEmptyKey()
    {
      return {};
    }

    static inline marco::codegen::lowering::ClassPath getTombstoneKey()
    {
      return {};
    }

    static unsigned getHashValue(
        const marco::codegen::lowering::ClassPath& val)
    {
      return llvm::hash_value(val.get());
    }

    static bool isEqual(
        const marco::codegen::lowering::ClassPath& lhs,
        const marco::codegen::lowering::ClassPath& rhs)
    {
      return lhs == rhs;
    }
  };
}

#endif // MARCO_CODEGEN_LOWERING_CLASSPATH_H
