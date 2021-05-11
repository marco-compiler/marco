#pragma once

#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourcePosition.h>
#include <string>

#include "Expression.h"

namespace modelica::frontend
{
	/**
	 * A reference access is pretty much any use of a variable at the moment.
	 */
	class ReferenceAccess
			: public impl::ExpressionCRTP<ReferenceAccess>,
				public impl::Cloneable<ReferenceAccess>
	{
		public:
		ReferenceAccess(SourcePosition location,
										Type type,
										llvm::StringRef name,
										bool globalLookup = false,
										bool dummy = false);

		ReferenceAccess(const ReferenceAccess& other);
		ReferenceAccess(ReferenceAccess&& other);
		~ReferenceAccess() override;

		ReferenceAccess& operator=(const ReferenceAccess& other);
		ReferenceAccess& operator=(ReferenceAccess&& other);

		friend void swap(ReferenceAccess& first, ReferenceAccess& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::EXPRESSION_REFERENCE_ACCESS;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool isLValue() const override;

		[[nodiscard]] bool operator==(const ReferenceAccess& other) const;
		[[nodiscard]] bool operator!=(const ReferenceAccess& other) const;

		[[nodiscard]] llvm::StringRef getName() const;
		void setName(llvm::StringRef name);

		[[nodiscard]] bool hasGlobalLookup() const;

		/**
		 * Get whether the referenced variable is created just for temporary
		 * use (such as a function output that is then discarded) and thus the
		 * reference points to a not already existing variable.
		 *
		 * @return true if temporary; false otherwise
		 */
		[[nodiscard]] bool isDummy() const;

		static ReferenceAccess dummy(SourcePosition location);

		private:
		std::string name;
		bool globalLookup;
		bool dummyVar;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const ReferenceAccess& obj);

	std::string toString(const ReferenceAccess& obj);
}
