#ifndef MARCO_AST_NODE_CLASS_H
#define MARCO_AST_NODE_CLASS_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Function.h"
#include "marco/AST/Node/Model.h"
#include "marco/AST/Node/Package.h"
#include "marco/AST/Node/Record.h"
#include <variant>

namespace marco::ast
{
	class Class
			: public impl::Cloneable<Class>,
				public impl::Dumpable<Class>
	{
		public:
		explicit Class(PartialDerFunction content);
		explicit Class(StandardFunction content);
		explicit Class(Model content);
		explicit Class(Package content);
		explicit Class(Record content);

		Class(const Class& other);
		Class(Class&& other);

		~Class();

		Class& operator=(const Class& other);
		Class& operator=(Class&& other);

		friend void swap(Class& first, Class& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		template<typename T>
		[[nodiscard]] bool isa() const
		{
			return std::holds_alternative<T>(content);
		}

		template<typename T>
		[[nodiscard]] T* get()
		{
			assert(isa<T>());
			return &std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] const T* get() const
		{
			assert(isa<T>());
			return &std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] T* dyn_get()
		{
			if (!isa<T>())
				return nullptr;

			return get<T>();
		}

		template<typename T>
		[[nodiscard]] const T* dyn_get() const
		{
			if (!isa<T>())
				return nullptr;

			return get<T>();
		}

		template<class Visitor>
		auto visit(Visitor&& visitor)
		{
			return std::visit(visitor, content);
		}

		template<class Visitor>
		auto visit(Visitor&& visitor) const
		{
			return std::visit(visitor, content);
		}

		[[nodiscard]] SourceRange getLocation() const;

		[[nodiscard]] llvm::StringRef getName() const;

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Class> partialDerFunction(Args&&... args)
		{
			return std::make_unique<Class>(PartialDerFunction(std::forward<Args>(args)...));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Class> standardFunction(Args&&... args)
		{
			return std::make_unique<Class>(StandardFunction(std::forward<Args>(args)...));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Class> model(Args&&... args)
		{
			return std::make_unique<Class>(Model(std::forward<Args>(args)...));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Class> package(Args&&... args)
		{
			return std::make_unique<Class>(Package(std::forward<Args>(args)...));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Class> record(Args&&... args)
		{
			return std::make_unique<Class>(Record(std::forward<Args>(args)...));
		}

		private:
		std::variant<PartialDerFunction, StandardFunction, Model, Package, Record> content;
	};
}

#endif // MARCO_AST_NODE_CLASS_H
