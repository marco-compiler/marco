#ifndef MARCO_AST_NODE_TYPEPREFIX_H
#define MARCO_AST_NODE_TYPEPREFIX_H

#include "marco/AST/Node/ASTNode.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <type_traits>

namespace marco::ast
{
	enum class VariabilityQualifier
	{
		discrete,
		parameter,
		constant,
		none
	};

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const VariabilityQualifier& obj);

	std::string toString(VariabilityQualifier qualifier);

	enum class IOQualifier
	{
		input,
		output,
		none
	};

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const IOQualifier& obj);

	std::string toString(IOQualifier qualifier);

	class TypePrefix : public impl::Dumpable<TypePrefix>
	{
		public:
      TypePrefix(
        VariabilityQualifier variabilityQualifier,
        IOQualifier ioQualifier);

      void print(llvm::raw_ostream& os, size_t indents = 0) const override;

      [[nodiscard]] bool isDiscrete() const;
      [[nodiscard]] bool isParameter() const;
      [[nodiscard]] bool isConstant() const;

      [[nodiscard]] bool isInput() const;
      [[nodiscard]] bool isOutput() const;

      static TypePrefix none();

		private:
      VariabilityQualifier variabilityQualifier;
      IOQualifier ioQualifier;
	};
}

#endif // MARCO_AST_NODE_TYPEPREFIX_H
