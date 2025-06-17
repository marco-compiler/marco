#ifndef MARCO_MODELING_DUMPABLE_H
#define MARCO_MODELING_DUMPABLE_H

#include <type_traits>

namespace llvm {
class raw_ostream;
}

namespace marco::modeling::internal {
template <typename Stream, typename T, typename = void>
struct is_to_stream_writable : std::false_type {};

template <typename Stream, typename T>
struct is_to_stream_writable<
    Stream, T,
    std::void_t<decltype(std::declval<Stream &>() << std::declval<T>())>>
    : std::true_type {};

template <typename Stream, typename T>
typename std::enable_if<!is_to_stream_writable<Stream, T>::value,
                        Stream &>::type
operator<<(Stream &stream, const T &obj) {
  return stream << "unknown (operator<< not implemented)";
}

class Dumpable {
public:
  virtual ~Dumpable();

  void dump() const;

  virtual void dump(llvm::raw_ostream &os) const = 0;
};
} // namespace marco::modeling::internal

#endif // MARCO_MODELING_DUMPABLE_H
