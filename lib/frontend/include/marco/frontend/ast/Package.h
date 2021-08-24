#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <string>

#include "ASTNode.h"

namespace marco::frontend
{
	class Class;

	class Package
			: public ASTNode,
				public impl::Dumpable<Package>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::unique_ptr<Class>>::iterator;
		using const_iterator = Container<std::unique_ptr<Class>>::const_iterator;

		Package(const Package& other);
		Package(Package&& other);
		~Package() override;

		Package& operator=(const Package& other);
		Package& operator=(Package&& other);

		friend void swap(Package& first, Package& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] llvm::StringRef getName() const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Class>> getInnerClasses();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Class>> getInnerClasses() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		friend class Class;

		Package(SourceRange location,
						llvm::StringRef name,
						llvm::ArrayRef<std::unique_ptr<Class>> innerClasses);

		std::string name;
		Container<std::unique_ptr<Class>> innerClasses;
	};
}
