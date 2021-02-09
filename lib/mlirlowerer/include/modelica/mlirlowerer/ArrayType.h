#pragma once

#include <mlir/IR/BuiltinTypes.h>

namespace modelica
{
	/**
 * Internal storage of the Modelica "ArrayType".
 */
	struct ArrayTypeStorage : public mlir::TypeStorage {

		/**
		 * The "KeyTy" is a required type that provides an interface for the storage
		 * instance. This type will be used when uniquing an instance of the type
		 * storage.
		 */
		using KeyTy = std::tuple<mlir::MemRefType, bool>;

		/**
		 * Constructor.
		 *
		 * @param memRefType descriptor of the memref
		 * @param heap  		 whether the memref is allocated on the heap or not
		 */
		ArrayTypeStorage(mlir::MemRefType memRefType, bool heap);

		/**
		 * Comparison function for the key type with the current storage instance.
		 * This is used when constructing a new instance to ensure that we
		 * haven't already uniqued an instance of the given key.
		 */
		bool operator==(const KeyTy &key) const;

		/**
		 * Hash function for the key type. This is used when uniquing instances of
		 * the storage.
		 */
		static llvm::hash_code hashKey(const KeyTy& key);

		/**
		 * Define a construction function for the key type from a set of parameters.
		 * These parameters will be provided when constructing the storage instance
		 * itself.
		 */
		static KeyTy getKey(mlir::MemRefType type, bool heap);

		/**
		 * Create a new instance of this storage.
		 * This method takes an instance of a storage allocator, and an instance of a
		 * "KeyTy".
		 */
		static ArrayTypeStorage *construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key);

		mlir::MemRefType memRefType;
		bool heap;
	};

	class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type, ArrayTypeStorage> {
		public:
		using Base::Base;

		static ArrayType get(mlir::MemRefType type, bool heap) {
			mlir::MLIRContext *ctx = type.getContext();
			return Base::get(ctx, type, heap);
		}

		mlir::MemRefType getMemRefType() {
			return getImpl()->memRefType;
		}

		bool isOnHeap() {
			return getImpl()->heap;
		}
	};
}
