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
namespace xla {
namespace gpu {
class IndexingMapAttr;
class LaunchGridAttr;
class MemorySpaceAttr;
class LayoutAttr;
namespace detail {
struct IndexingMapAttrStorage;
} // namespace detail
class IndexingMapAttr : public ::mlir::Attribute::AttrBase<IndexingMapAttr, ::mlir::Attribute, detail::IndexingMapAttrStorage> {
public:
  using Base::Base;
  // Returns the indexing map constructed from IndexingMapAttr.
  xla::gpu::IndexingMap getIndexingMap() const;

  // Returns the number of indexing map results.
  int64_t getNumResults() const;
  static constexpr ::llvm::StringLiteral name = "xla_gpu.indexing_map";
  static constexpr ::llvm::StringLiteral dialectName = "xla_gpu";
  using Base::getChecked;
  static IndexingMapAttr get(::mlir::MLIRContext *context, ::mlir::AffineMap map, ::llvm::ArrayRef<::xla::gpu::DimVar> dim_vars, ::llvm::ArrayRef<::xla::gpu::RangeVar> range_vars, ::llvm::ArrayRef<::std::pair<::mlir::AffineExpr, ::xla::gpu::Interval>> constraints, bool is_simplified);
  static IndexingMapAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::AffineMap map, ::llvm::ArrayRef<::xla::gpu::DimVar> dim_vars, ::llvm::ArrayRef<::xla::gpu::RangeVar> range_vars, ::llvm::ArrayRef<::std::pair<::mlir::AffineExpr, ::xla::gpu::Interval>> constraints, bool is_simplified);
  static IndexingMapAttr get(::mlir::MLIRContext *context, const ::xla::gpu::IndexingMap&indexing_map);
  static IndexingMapAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, const ::xla::gpu::IndexingMap&indexing_map);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::AffineMap map, ::llvm::ArrayRef<::xla::gpu::DimVar> dim_vars, ::llvm::ArrayRef<::xla::gpu::RangeVar> range_vars, ::llvm::ArrayRef<::std::pair<::mlir::AffineExpr, ::xla::gpu::Interval>> constraints, bool is_simplified);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::AffineMap map, ::llvm::ArrayRef<::xla::gpu::DimVar> dim_vars, ::llvm::ArrayRef<::xla::gpu::RangeVar> range_vars, ::llvm::ArrayRef<::std::pair<::mlir::AffineExpr, ::xla::gpu::Interval>> constraints, bool is_simplified);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"indexing_map"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::AffineMap getMap() const;
  ::llvm::ArrayRef<::xla::gpu::DimVar> getDimVars() const;
  ::llvm::ArrayRef<::xla::gpu::RangeVar> getRangeVars() const;
  ::llvm::ArrayRef<::std::pair<::mlir::AffineExpr, ::xla::gpu::Interval>> getConstraints() const;
  bool getIsSimplified() const;
};
namespace detail {
struct LaunchGridAttrStorage;
} // namespace detail
class LaunchGridAttr : public ::mlir::Attribute::AttrBase<LaunchGridAttr, ::mlir::Attribute, detail::LaunchGridAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xla_gpu.launch_grid";
  static constexpr ::llvm::StringLiteral dialectName = "xla_gpu";
  static LaunchGridAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> block_counts, ::llvm::ArrayRef<int64_t> thread_counts);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"launch_grid"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getBlockCounts() const;
  ::llvm::ArrayRef<int64_t> getThreadCounts() const;
};
namespace detail {
struct MemorySpaceAttrStorage;
} // namespace detail
class MemorySpaceAttr : public ::mlir::Attribute::AttrBase<MemorySpaceAttr, ::mlir::Attribute, detail::MemorySpaceAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xla_gpu.memory_space";
  static constexpr ::llvm::StringLiteral dialectName = "xla_gpu";
  static MemorySpaceAttr get(::mlir::MLIRContext *context, ::xla::gpu::MemorySpace value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"memory_space"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::xla::gpu::MemorySpace getValue() const;
};
namespace detail {
struct LayoutAttrStorage;
} // namespace detail
class LayoutAttr : public ::mlir::Attribute::AttrBase<LayoutAttr, ::mlir::Attribute, detail::LayoutAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xla_gpu.layout";
  static constexpr ::llvm::StringLiteral dialectName = "xla_gpu";
  static LayoutAttr get(::mlir::MLIRContext *context, MemorySpaceAttr memory_space, IndexingMapAttr thread_map);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"layout"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  MemorySpaceAttr getMemorySpace() const;
  IndexingMapAttr getThreadMap() const;
};
} // namespace gpu
} // namespace xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::gpu::IndexingMapAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::gpu::LaunchGridAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::gpu::MemorySpaceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::gpu::LayoutAttr)

#endif  // GET_ATTRDEF_CLASSES

