#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <variant>

namespace marco
{

    class Shape{
		public:

		struct DimensionSize
		{
			template <typename T> using Container = llvm::SmallVector<T, 3>;
			using Undefined = std::monostate;
			using Ragged = std::unique_ptr<Container<DimensionSize>>;
		
            DimensionSize();
            DimensionSize(long int);
            DimensionSize(std::initializer_list<DimensionSize> ragged);
            //DimensionSize(llvm::ArrayRef<DimensionSize> ragged);
			DimensionSize(const Container<DimensionSize> &ragged);
			DimensionSize(std::unique_ptr<Container<DimensionSize>> ragged);

			DimensionSize(const DimensionSize& other);
			DimensionSize(DimensionSize&& other);

			~DimensionSize();

			DimensionSize& operator=(const DimensionSize& other);
			DimensionSize& operator=(DimensionSize&& other);

			friend void swap(DimensionSize& first, DimensionSize& second);

			[[nodiscard]] bool operator==(const DimensionSize& other) const;
			[[nodiscard]] bool operator!=(const DimensionSize& other) const;
            
			bool isRagged() const;
			bool isUndefined() const;
			bool isConstant() const;
			// operator long() const;

			long getNumericValue() const;


            friend llvm::hash_code hash_value(const DimensionSize& d);

			static DimensionSize makeUndefined(){
				return DimensionSize();
			}

			llvm::ArrayRef<DimensionSize> asRagged() const;

			std::variant<Undefined, long, Ragged> value;
            
		};

		using Sizes = llvm::SmallVector<DimensionSize, 3>;
		using iterator = Sizes::iterator;
		using const_iterator = Sizes::const_iterator;

		iterator begin();
		iterator end();

		const_iterator begin() const;
		const_iterator end() const;

        DimensionSize& operator[](int index);
        const DimensionSize& operator[](int index) const;
        
        bool empty() const;
		size_t size() const;

        void push_back(const DimensionSize& d);

        bool operator==(const Shape& other) const;
        bool operator!=(const Shape& other) const;

        llvm::SmallVector<long, 3> to_old() const;

		bool isRagged() const;
		bool isUndefined() const;
		bool isConstant() const;
        
        Shape() = default;
        // Shape(std::initializer_list<long int> values);
        Shape(std::initializer_list<DimensionSize> values);
        Shape(llvm::ArrayRef<long> values);
        Shape(const llvm::SmallVector<long,3> &values);

        Shape(llvm::NoneType);
        Shape(DimensionSize el);
        Shape(DimensionSize el, size_t length);

        //operator llvm::SmallVector<long,3>() const;

		llvm::ArrayRef<DimensionSize> dimensions() const;

		private:

		Sizes sizes;
	};

	extern std::string toString(const Shape &shape);

	extern std::vector<std::vector<long>> generateAllIndexes(Shape shape);
}