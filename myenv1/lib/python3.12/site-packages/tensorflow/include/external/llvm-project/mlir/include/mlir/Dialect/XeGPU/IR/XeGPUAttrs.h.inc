/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace xegpu {
class TensorDescAttr;
class MemoryScopeAttr;
class CachePolicyAttr;
class FenceScopeAttr;
namespace detail {
struct TensorDescAttrStorage;
} // namespace detail
class TensorDescAttr : public ::mlir::Attribute::AttrBase<TensorDescAttr, ::mlir::Attribute, detail::TensorDescAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.tdesc_attr";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static TensorDescAttr get(::mlir::MLIRContext *context, MemoryScopeAttr memory_scope, IntegerAttr array_length, BoolAttr boundary_check, BoolAttr scattered);
  static TensorDescAttr get(::mlir::MLIRContext *context, xegpu::MemoryScope memory_scope = xegpu::MemoryScope::Global, int array_length = 1, bool boundary_check = true, bool scattered = false);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tdesc_attr"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  MemoryScopeAttr getMemoryScope() const;
  IntegerAttr getArrayLength() const;
  BoolAttr getBoundaryCheck() const;
  BoolAttr getScattered() const;
};
namespace detail {
struct MemoryScopeAttrStorage;
} // namespace detail
class MemoryScopeAttr : public ::mlir::Attribute::AttrBase<MemoryScopeAttr, ::mlir::Attribute, detail::MemoryScopeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.memory_scope";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static MemoryScopeAttr get(::mlir::MLIRContext *context, ::mlir::xegpu::MemoryScope value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"memory_scope"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xegpu::MemoryScope getValue() const;
};
namespace detail {
struct CachePolicyAttrStorage;
} // namespace detail
class CachePolicyAttr : public ::mlir::Attribute::AttrBase<CachePolicyAttr, ::mlir::Attribute, detail::CachePolicyAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.cache_hint";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static CachePolicyAttr get(::mlir::MLIRContext *context, ::mlir::xegpu::CachePolicy value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"cache_hint"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xegpu::CachePolicy getValue() const;
};
namespace detail {
struct FenceScopeAttrStorage;
} // namespace detail
class FenceScopeAttr : public ::mlir::Attribute::AttrBase<FenceScopeAttr, ::mlir::Attribute, detail::FenceScopeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.fence_scope";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static FenceScopeAttr get(::mlir::MLIRContext *context, ::mlir::xegpu::FenceScope value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"fence_scope"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xegpu::FenceScope getValue() const;
};
} // namespace xegpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::TensorDescAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::MemoryScopeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::CachePolicyAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::FenceScopeAttr)

#endif  // GET_ATTRDEF_CLASSES

