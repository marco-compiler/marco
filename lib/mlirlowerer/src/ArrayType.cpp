#include <modelica/mlirlowerer/ArrayType.h>

using namespace mlir;
using namespace modelica;
using namespace std;

ArrayTypeStorage::ArrayTypeStorage(mlir::MemRefType memRefType, bool heap)
		: memRefType(memRefType), heap(heap)
{
}

bool ArrayTypeStorage::operator==(const KeyTy &key) const
{
	return key == std::make_tuple(memRefType, heap);
}

llvm::hash_code ArrayTypeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_value(key);
}

ArrayTypeStorage::KeyTy ArrayTypeStorage::getKey(mlir::MemRefType type, bool heap)
{
	return KeyTy(type, heap);
}

ArrayTypeStorage* ArrayTypeStorage::construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key)
{
	return new (allocator.allocate<ArrayTypeStorage>()) ArrayTypeStorage(std::get<0>(key), std::get<1>(key));
}
