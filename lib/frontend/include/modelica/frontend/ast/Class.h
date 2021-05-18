#pragma once

#include <variant>

#include "ASTNode.h"
#include "Function.h"
#include "Model.h"
#include "Package.h"
#include "Record.h"

namespace modelica::frontend
{
	class Class
			: public impl::Cloneable<Class>,
				public impl::Dumpable<Class>
	{
		public:
		explicit Class(DerFunction content);
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

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Class> derFunction(Args&&... args)
		{
			return std::make_unique<Class>(DerFunction(std::forward<Args>(args)...));
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
		std::variant<DerFunction, StandardFunction, Model, Package, Record> content;
	};
}
