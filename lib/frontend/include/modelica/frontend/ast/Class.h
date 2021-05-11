#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <string>

#include "ASTNode.h"

namespace modelica::frontend
{
	class Class : public impl::ASTNodeCRTP<Class>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		Class(ASTNodeKind kind,
					SourcePosition location,
					llvm::StringRef name);

		Class(const Class& other);
		Class(Class&& other);

		~Class() override;

		Class& operator=(const Class& other);
		Class& operator=(Class&& other);

		friend void swap(Class& first, Class& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::CLASS &&
						 node->getKind() <= ASTNodeKind::CLASS_LAST;
		}

		[[nodiscard]] virtual std::unique_ptr<Class> cloneClass() const = 0;

		[[nodiscard]] llvm::StringRef getName() const;

		private:
		std::string name;
	};

	namespace impl
	{
		template<typename Derived>
		struct ClassCRTP : public Class
		{
			using Class::Class;

			[[nodiscard]] std::unique_ptr<Class> cloneClass() const override
			{
				return std::make_unique<Derived>(static_cast<const Derived&>(*this));
			}
		};
	}
}
