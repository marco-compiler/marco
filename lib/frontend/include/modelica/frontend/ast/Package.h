#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourcePosition.h>
#include <string>

namespace modelica::frontend
{
	class ClassContainer;

	class Package
	{
		private:
		template<typename T> using Container = llvm::SmallVector<std::shared_ptr<T>, 3>;

		public:
		using iterator = boost::indirect_iterator<Container<ClassContainer>::iterator>;
		using const_iterator = boost::indirect_iterator<Container<ClassContainer>::const_iterator>;

		Package(
				SourcePosition location,
				std::string name,
				llvm::ArrayRef<ClassContainer> innerClasses);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] const std::string& getName() const;

		[[nodiscard]] Container<ClassContainer>& getInnerClasses();
		[[nodiscard]] const Container<ClassContainer>& getInnerClasses() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		SourcePosition location;
		std::string name;
		Container<ClassContainer> innerClasses;
	};
}
