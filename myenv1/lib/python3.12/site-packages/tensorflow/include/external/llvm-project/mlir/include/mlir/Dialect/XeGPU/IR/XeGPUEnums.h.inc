/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: XeGPUAttrs.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace xegpu {
// Cache policy
enum class CachePolicy : uint32_t {
  CACHED = 0,
  UNCACHED = 1,
  STREAMING = 2,
  READ_INVALIDATE = 3,
  WRITE_BACK = 4,
  WRITE_THROUGH = 5,
};

::std::optional<CachePolicy> symbolizeCachePolicy(uint32_t);
::llvm::StringRef stringifyCachePolicy(CachePolicy);
::std::optional<CachePolicy> symbolizeCachePolicy(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCachePolicy() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(CachePolicy enumValue) {
  return stringifyCachePolicy(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CachePolicy> symbolizeEnum<CachePolicy>(::llvm::StringRef str) {
  return symbolizeCachePolicy(str);
}
} // namespace xegpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::xegpu::CachePolicy, ::mlir::xegpu::CachePolicy> {
  template <typename ParserT>
  static FailureOr<::mlir::xegpu::CachePolicy> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Cache policy");

    // Symbolize the keyword.
    if (::std::optional<::mlir::xegpu::CachePolicy> attr = ::mlir::xegpu::symbolizeEnum<::mlir::xegpu::CachePolicy>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Cache policy specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::xegpu::CachePolicy value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::xegpu::CachePolicy> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::xegpu::CachePolicy getEmptyKey() {
    return static_cast<::mlir::xegpu::CachePolicy>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::xegpu::CachePolicy getTombstoneKey() {
    return static_cast<::mlir::xegpu::CachePolicy>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::xegpu::CachePolicy &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::xegpu::CachePolicy &lhs, const ::mlir::xegpu::CachePolicy &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace xegpu {
// The enumeration for the scope of fence operation.
enum class FenceScope : uint32_t {
  Workgroup = 0,
  GPU = 1,
};

::std::optional<FenceScope> symbolizeFenceScope(uint32_t);
::llvm::StringRef stringifyFenceScope(FenceScope);
::std::optional<FenceScope> symbolizeFenceScope(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFenceScope() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(FenceScope enumValue) {
  return stringifyFenceScope(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<FenceScope> symbolizeEnum<FenceScope>(::llvm::StringRef str) {
  return symbolizeFenceScope(str);
}
} // namespace xegpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::xegpu::FenceScope, ::mlir::xegpu::FenceScope> {
  template <typename ParserT>
  static FailureOr<::mlir::xegpu::FenceScope> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for The enumeration for the scope of fence operation.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::xegpu::FenceScope> attr = ::mlir::xegpu::symbolizeEnum<::mlir::xegpu::FenceScope>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid The enumeration for the scope of fence operation. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::xegpu::FenceScope value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::xegpu::FenceScope> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::xegpu::FenceScope getEmptyKey() {
    return static_cast<::mlir::xegpu::FenceScope>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::xegpu::FenceScope getTombstoneKey() {
    return static_cast<::mlir::xegpu::FenceScope>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::xegpu::FenceScope &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::xegpu::FenceScope &lhs, const ::mlir::xegpu::FenceScope &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace xegpu {
// The address space of the memory the tensor descritor is created for
enum class MemoryScope : uint32_t {
  Global = 0,
  SLM = 1,
};

::std::optional<MemoryScope> symbolizeMemoryScope(uint32_t);
::llvm::StringRef stringifyMemoryScope(MemoryScope);
::std::optional<MemoryScope> symbolizeMemoryScope(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMemoryScope() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(MemoryScope enumValue) {
  return stringifyMemoryScope(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MemoryScope> symbolizeEnum<MemoryScope>(::llvm::StringRef str) {
  return symbolizeMemoryScope(str);
}
} // namespace xegpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::xegpu::MemoryScope, ::mlir::xegpu::MemoryScope> {
  template <typename ParserT>
  static FailureOr<::mlir::xegpu::MemoryScope> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for The address space of the memory the tensor descritor is created for");

    // Symbolize the keyword.
    if (::std::optional<::mlir::xegpu::MemoryScope> attr = ::mlir::xegpu::symbolizeEnum<::mlir::xegpu::MemoryScope>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid The address space of the memory the tensor descritor is created for specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::xegpu::MemoryScope value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::xegpu::MemoryScope> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::xegpu::MemoryScope getEmptyKey() {
    return static_cast<::mlir::xegpu::MemoryScope>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::xegpu::MemoryScope getTombstoneKey() {
    return static_cast<::mlir::xegpu::MemoryScope>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::xegpu::MemoryScope &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::xegpu::MemoryScope &lhs, const ::mlir::xegpu::MemoryScope &rhs) {
    return lhs == rhs;
  }
};
}

