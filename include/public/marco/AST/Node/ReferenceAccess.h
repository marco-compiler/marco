#ifndef MARCO_AST_NODE_REFERENCEACCESS_H
#define MARCO_AST_NODE_REFERENCEACCESS_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Type.h"
#include <string>

namespace marco::ast
{
  class Expression;

	/// A reference access is pretty much any use of a variable at the moment.
	class ReferenceAccess
			: public ASTNode,
				public impl::Dumpable<ReferenceAccess>
	{
		public:
      ReferenceAccess(const ReferenceAccess& other);
      ReferenceAccess(ReferenceAccess&& other);
      ~ReferenceAccess() override;

      ReferenceAccess& operator=(const ReferenceAccess& other);
      ReferenceAccess& operator=(ReferenceAccess&& other);

      friend void swap(ReferenceAccess& first, ReferenceAccess& second);

      void print(llvm::raw_ostream& os, size_t indents = 0) const override;

      [[nodiscard]] bool isLValue() const;

      [[nodiscard]] bool operator==(const ReferenceAccess& other) const;
      [[nodiscard]] bool operator!=(const ReferenceAccess& other) const;

      [[nodiscard]] Type& getType();
      [[nodiscard]] const Type& getType() const;
      void setType(Type tp);

      [[nodiscard]] llvm::StringRef getName() const;
      void setName(llvm::StringRef name);

      [[nodiscard]] bool hasGlobalLookup() const;

      /// Get whether the referenced variable is created just for temporary
      /// use (such as a function output that is then discarded) and thus the
      /// reference points to a not already existing variable.
      [[nodiscard]] bool isDummy() const;

      static std::unique_ptr<Expression> dummy(SourceRange location, Type type);

		private:
      friend class Expression;

      ReferenceAccess(SourceRange location,
                      Type type,
                      llvm::StringRef name,
                      bool globalLookup = false,
                      bool dummy = false);

    private:
      Type type;
      std::string name;
      bool globalLookup;
      bool dummyVar;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const ReferenceAccess& obj);

	std::string toString(const ReferenceAccess& obj);
}

#endif // MARCO_AST_NODE_REFERENCEACCESS_H
