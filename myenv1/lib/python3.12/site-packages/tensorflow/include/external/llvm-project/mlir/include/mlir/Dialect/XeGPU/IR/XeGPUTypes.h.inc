/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace xegpu {
class TensorDescType;
class NbarrierType;
namespace detail {
struct TensorDescTypeStorage;
} // namespace detail
class TensorDescType : public ::mlir::Type::TypeBase<TensorDescType, ::mlir::TensorType, detail::TensorDescTypeStorage, ::mlir::ShapedType::Trait> {
public:
  using Base::Base;
  using TensorType::clone;
  using mlir::ShapedType::Trait<TensorDescType>::getElementTypeBitWidth;
  using mlir::ShapedType::Trait<TensorDescType>::getRank;
  using mlir::ShapedType::Trait<TensorDescType>::getNumElements;
  using mlir::ShapedType::Trait<TensorDescType>::isDynamicDim;
  using mlir::ShapedType::Trait<TensorDescType>::hasStaticShape;
  using mlir::ShapedType::Trait<TensorDescType>::getNumDynamicDims;
  using mlir::ShapedType::Trait<TensorDescType>::getDimSize;
  using mlir::ShapedType::Trait<TensorDescType>::getDynamicDimIndex;

  TensorDescType clone(::mlir::Type elementType) {
    return llvm::cast<TensorDescType>(cloneWith(getShape(), elementType));
  }

  TensorDescAttr getEncodingAsTensorDescAttr() const {
    return llvm::dyn_cast_if_present<TensorDescAttr>(getEncoding());
  }

  xegpu::MemoryScope getMemoryScope() const {
    auto attr = getEncodingAsTensorDescAttr();
    if (attr && attr.getMemoryScope())
      return attr.getMemoryScope().getValue();
    // return default value
    return MemoryScope::Global;
  }

  int getArrayLength() {
    auto attr = getEncodingAsTensorDescAttr();
    if (attr && attr.getArrayLength())
      return attr.getArrayLength().getInt();
    // return default value
    return 1;
  }

  bool getBoundaryCheck() {
    auto attr = getEncodingAsTensorDescAttr();
    if (attr && attr.getBoundaryCheck())
      return attr.getBoundaryCheck().getValue();
    // return default value
    return true;
  }

  bool getScattered() {
    auto attr = getEncodingAsTensorDescAttr();
    if (attr && attr.getScattered())
      return attr.getScattered().getValue();
    // return default value
    return false;
  }
  static constexpr ::llvm::StringLiteral name = "xegpu.tensor_desc";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static TensorDescType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute encoding);
  static TensorDescType get(llvm::ArrayRef<int64_t> shape, mlir::Type elementType, bool scattered = false, int array_length = 1, xegpu::MemoryScope memory_scope = xegpu::MemoryScope::Global, bool boundary_check = true);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_desc"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  mlir::Type getElementType() const;
  mlir::Attribute getEncoding() const;
};
class NbarrierType : public ::mlir::Type::TypeBase<NbarrierType, mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static NbarrierType get(mlir::MLIRContext *context) {
    return Base::get(context);
  };
  static constexpr ::llvm::StringLiteral name = "xegpu.nbarrier";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"nbarrier"};
  }

};
} // namespace xegpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::TensorDescType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::NbarrierType)

#endif  // GET_TYPEDEF_CLASSES

