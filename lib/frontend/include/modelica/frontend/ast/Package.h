#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>

#include "Class.h"

namespace modelica::frontend
{
	class Package
			: public impl::ClassCRTP<Package>,
				public impl::Cloneable<Package>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::unique_ptr<Class>>::iterator;
		using const_iterator = Container<std::unique_ptr<Class>>::const_iterator;

		Package(SourcePosition location,
						llvm::StringRef name,
						llvm::ArrayRef<std::unique_ptr<Class>> innerClasses);

		Package(const Package& other);
		Package(Package&& other);
		~Package() override;

		Package& operator=(const Package& other);
		Package& operator=(Package&& other);

		friend void swap(Package& first, Package& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::CLASS_PACKAGE;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Class>> getInnerClasses();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Class>> getInnerClasses() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		Container<std::unique_ptr<Class>> innerClasses;
	};
}
