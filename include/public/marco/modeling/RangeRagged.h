#pragma once
#include "marco/modeling/Range.h"

#include <variant>
#include <vector>

namespace marco::modeling
{

    struct RaggedValue{
        using data_type = Point::data_type;
        template<typename T>
        using Container=llvm::SmallVector<T, 2>;
        using Ragged = Container<RaggedValue>;
        using RaggedPtr = std::unique_ptr<Ragged>;
        
        RaggedValue(data_type value = 0): value(value){}
        RaggedValue(const Container<RaggedValue> &ragged)
            : value(std::make_unique<Ragged>(ragged)){}

        RaggedValue(std::initializer_list<RaggedValue> list)
            : value(std::make_unique<Ragged>(list.begin(),list.end())){}

        RaggedValue(const RaggedValue& other): value(0U){
            if(other.isRagged()){
                value = std::make_unique<Ragged>(*std::get<RaggedPtr>(other.value));
            }else{
                value = other.asValue();
            }
        }
        RaggedValue(RaggedValue&& other) = default;
        ~RaggedValue() = default;

        RaggedValue& operator=(const RaggedValue& other) {
            if(other.isRagged()){
                value = std::make_unique<Ragged>(*std::get<RaggedPtr>(other.value));
            }else{
                value = other.asValue();
            }
            return *this;
        }

        bool operator==(const RaggedValue& other) const;
        bool operator!=(const RaggedValue& other) const;
        bool operator>(const RaggedValue& other) const;
        bool operator<(const RaggedValue& other) const;
        bool operator>=(const RaggedValue& other) const;
        bool operator<=(const RaggedValue& other) const;
        
        RaggedValue operator*(const RaggedValue& other) const;        RaggedValue operator+(const RaggedValue& other) const;
        RaggedValue operator-(const RaggedValue& other) const;
        RaggedValue operator/(const RaggedValue& other) const;

        RaggedValue& operator+=(const RaggedValue& other);
        RaggedValue& operator-=(const RaggedValue& other);
        RaggedValue& operator*=(const RaggedValue& other);
        RaggedValue& operator/=(const RaggedValue& other);

        data_type min() const;
        data_type max() const;

        RaggedValue& operator++();
        RaggedValue operator++(int);

        bool isRagged() const
        {
            return std::holds_alternative<RaggedPtr>(value);
        }

        [[nodiscard]] llvm::ArrayRef<RaggedValue> asRagged() const
        {
            return *std::get<RaggedPtr>(value);
        }

        data_type asValue() const
        {
            return std::get<data_type>(value);
        }
    private:
        std::variant<data_type, RaggedPtr> value;
    };


    struct RangeRagged{
        using data_type = Range;        
        template<typename T>
        using Container=llvm::SmallVector<T, 2>;
        using Ragged = Container<RangeRagged>;
        using RaggedPtr = std::unique_ptr<Ragged>;
        
        explicit RangeRagged():value(Range(0,1)){}
        
        // RangeRagged(data_type begin, data_type end);
        RangeRagged(Range interval): value(interval){}

        explicit RangeRagged(llvm::ArrayRef<RangeRagged> ragged)
			: value(std::make_unique<Ragged>(ragged.begin(),ragged.end()))
		{
			compact();
		}

        RangeRagged(std::initializer_list<RangeRagged> list):
            value(std::make_unique<Ragged>(list.begin(),list.end()))
		{
			compact();
		}
        
        RangeRagged(const RaggedValue& min,const RaggedValue& max);

        RangeRagged(const RangeRagged& other):value(Range(0,1)){
            if(other.isRagged()){
                value = std::make_unique<Ragged>(*std::get<RaggedPtr>(other.value));
            }else{
                value = other.asValue();
            }
        }

        RangeRagged(RangeRagged&& other):value(std::move(other.value)){}
        ~RangeRagged() = default;

        RangeRagged& operator=(const RangeRagged& other) {
            if(other.isRagged()){
                value = std::make_unique<Ragged>(*std::get<RaggedPtr>(other.value));
            }else{
                value = other.asValue();
            }
            return *this;
        }

        bool operator==(Point::data_type other) const {
            return *this==RangeRagged(other,other+1);
        }
         bool operator!=(Point::data_type other) const {
            return *this!=RangeRagged(other,other+1);
        }
        bool operator==(const RangeRagged& other) const {
            if(isRagged()){
                if(other.isRagged()){
                    auto a=asRagged();
                    auto b=other.asRagged();
                    return a==b;
                }
                return false;
            }
            if(other.isRagged())return false;
            return asValue()==other.asValue();
        }
        bool operator!=(const RangeRagged& other) const {
            return !(*this==other);
        }

        [[nodiscard]] bool operator>(const RangeRagged& other) const
		{
			return min() > other.min();
		}
		[[nodiscard]] bool operator<(const RangeRagged& other) const
		{
			return min() < other.min();
		}
		[[nodiscard]] bool operator>=(const RangeRagged& other) const
		{
			return min() >= other.min();
		}
		[[nodiscard]] bool operator<=(const RangeRagged& other) const
		{
			return min() <= other.min();
		}
        [[nodiscard]] bool operator==(data_type other) const;
        [[nodiscard]] bool operator!=(data_type other) const;

		[[nodiscard]] auto begin() const {
			assert(!isRagged());
			return asValue().begin(); 
		}

		[[nodiscard]] auto end() const {
			assert(!isRagged());
			return asValue().end(); 
		}

        bool isRagged() const
        {
            return std::holds_alternative<RaggedPtr>(value);
        }

        [[nodiscard]] llvm::ArrayRef<RangeRagged> asRagged() const
        {
            assert(isRagged());
            return *std::get<RaggedPtr>(value);
        }

        Range asValue() const
        {
            assert(!isRagged());
            return std::get<Range>(value);
        }

        long min() const;
        long max() const;

        Range getContainingRange() const;

        RaggedValue getBegin() const;
        RaggedValue getEnd() const;
        
        [[nodiscard]] bool contains(long element) const
		{
			assert(!isRagged());
			return asValue().contains(element);
		}

		template<typename... T>
		[[nodiscard]] bool contains(T... coordinates) const
		{
			return contains({ coordinates... });
		}
        
        [[nodiscard]] bool contains(const RangeRagged& other) const;
        [[nodiscard]] bool isContained(const RangeRagged& other) const;
        [[nodiscard]] bool isFullyContained(const RangeRagged& other) const;

        [[nodiscard]] size_t size() const;
        [[nodiscard]] RaggedValue getSize() const;

        bool overlaps(const RangeRagged& other) const;

        RangeRagged intersect(const RangeRagged& other) const;

        /// Check whether the range can be merged with another one.
        /// Two ranges can be merged if they overlap or if they are contiguous.
        bool canBeMerged(const RangeRagged& other) const;

        /// Create a range that is the resulting of merging this one with
        /// another one that can be merged.
        RangeRagged merge(const RangeRagged& other) const;

        /// Subtract a range from the current one.
        /// Multiple results are created if the removed range is fully contained
        /// and does not touch the borders.
        std::vector<RangeRagged> subtract(const RangeRagged& other) const;

        void compact();

        // void dump(llvm::raw_ostream& OS = llvm::outs()) const;
    private:
        std::variant<Range, RaggedPtr> value;
    };

}	 // namespace marco
