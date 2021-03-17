#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourceRange.hpp>
#include <string>

namespace modelica
{
	class ClassContainer;

	class Package
	{
		private:
		template<typename T> using Container = llvm::SmallVector<std::shared_ptr<T>, 3>;

		public:
		Class(
				SourcePosition location,
				std::string name,
				llvm::ArrayRef<ClassContainer> innerClasses);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] const std::string& getName() const;

		[[nodiscard]] Container<ClassContainer>& getInnerClasses();
		[[nodiscard]] const Container<ClassContainer>& getInnerClasses() const;

		private:
		SourcePosition location;
		Container<ClassContainer> innerClasses;
	};
}	 // namespace modelica
